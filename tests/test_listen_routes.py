"""
Tests for the Listen feature routes in app.py.

Covers (regression guards):
- POST /api/tracks/<id>/rating  — the route fixed in PR #67 (Rekordbox stores 0-5,
  not 0-255). Validates input range, type coercion, auth, and Rekordbox-running guard.
- GET  /api/tracks/<id>/cues    — read cue points; sorted by time_ms.
- POST /api/tracks/<id>/cues    — added in PR #66; validates time_ms, auth, RB guard.
- GET  /api/listen/playlist/<int:pid>/tracks — wraps get_playlist_tracks.py
  (the smart-playlist evaluation path fixed in PR #65).
- GET  /api/listen/all-tracks   — wraps get_playlist_tracks.py with --playlist-id all.
- GET  /api/listen/tree         — wraps get_listen_tree.py.

These routes had ZERO test coverage before this file. Every PR since v1.0.0
that fixed Listen behavior touched untested code paths.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

import app as flask_app


@pytest.fixture
def auth_client(tmp_path):
    """Flask test client with listen_ok=True in session and a configured db_path."""
    flask_app.app.config["TESTING"] = True
    cfg = {
        "db_path": str(tmp_path / "master.db"),
        "music_root": str(tmp_path / "music"),
        "flac_root": str(tmp_path / "flac"),
        "mp3_root": str(tmp_path / "mp3"),
        "delete_dir": str(tmp_path / "DELETE"),
        "watch_dir": str(tmp_path / "watch"),
    }
    with (
        patch.object(flask_app, "load_config", return_value=cfg),
        flask_app.app.test_client() as c,
    ):
        with c.session_transaction() as sess:
            sess["listen_ok"] = True
        yield c


@pytest.fixture
def unauth_client():
    """Flask test client with no listen session — every Listen route should 401."""
    flask_app.app.config["TESTING"] = True
    with flask_app.app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# POST /api/tracks/<id>/rating  — PR #67 regression guards
# ---------------------------------------------------------------------------


def _track_mock():
    """Mock DjmdContent row that the rating route mutates."""
    t = MagicMock()
    t.ID = "123"
    t.Rating = 0
    t.rb_local_deleted = 0
    return t


def _db_mock_with_track(track):
    db = MagicMock()
    db.session.query.return_value.filter_by.return_value.first.return_value = track
    return db


def test_rating_unauthorized_returns_401(unauth_client):
    resp = unauth_client.post("/api/tracks/123/rating", json={"rating": 3})
    assert resp.status_code == 401


def test_rating_blocked_when_rekordbox_running(auth_client):
    with patch.object(flask_app, "rekordbox_is_running", return_value=True):
        resp = auth_client.post("/api/tracks/123/rating", json={"rating": 3})
    assert resp.status_code == 409
    assert "Rekordbox" in resp.get_json()["error"]


@pytest.mark.parametrize("bad_value", [-1, 6, 7, 100, 255])
def test_rating_out_of_range_returns_400(auth_client, bad_value):
    """Regression guard for PR #67: 6+ or negative are not valid stars."""
    with patch.object(flask_app, "rekordbox_is_running", return_value=False):
        resp = auth_client.post("/api/tracks/123/rating", json={"rating": bad_value})
    assert resp.status_code == 400
    assert "0-5" in resp.get_json()["error"]


def test_rating_none_returns_400(auth_client):
    with patch.object(flask_app, "rekordbox_is_running", return_value=False):
        resp = auth_client.post("/api/tracks/123/rating", json={"rating": None})
    assert resp.status_code == 400


def test_rating_non_numeric_returns_400_not_500(auth_client):
    """Bad input must return 400, not crash with a 500."""
    with patch.object(flask_app, "rekordbox_is_running", return_value=False):
        resp = auth_client.post("/api/tracks/123/rating", json={"rating": "abc"})
    assert resp.status_code == 400


def test_rating_missing_body_returns_400(auth_client):
    with patch.object(flask_app, "rekordbox_is_running", return_value=False):
        resp = auth_client.post("/api/tracks/123/rating", json={})
    assert resp.status_code == 400


@pytest.mark.parametrize("stars", [0, 1, 2, 3, 4, 5])
def test_rating_valid_stars_writes_to_track(auth_client, stars):
    """Regression guard for PR #67: 0..5 is the valid range and gets written verbatim."""
    track = _track_mock()
    db = _db_mock_with_track(track)
    with (
        patch.object(flask_app, "rekordbox_is_running", return_value=False),
        patch.object(flask_app, "_open_db", return_value=db),
        patch.object(flask_app, "_ensure_stream_backup"),
    ):
        resp = auth_client.post("/api/tracks/123/rating", json={"rating": stars})
    assert resp.status_code == 200
    assert resp.get_json() == {"ok": True}
    assert track.Rating == stars  # written exactly as-is — no 0-255 conversion
    assert db.commit.called


def test_rating_track_not_found_returns_404(auth_client):
    db = MagicMock()
    db.session.query.return_value.filter_by.return_value.first.return_value = None
    with (
        patch.object(flask_app, "rekordbox_is_running", return_value=False),
        patch.object(flask_app, "_open_db", return_value=db),
        patch.object(flask_app, "_ensure_stream_backup"),
    ):
        resp = auth_client.post("/api/tracks/999/rating", json={"rating": 3})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/tracks/<id>/cues  — added in PR #66
# ---------------------------------------------------------------------------


def _cue_mock(cue_id, time_ms, kind=0, comment=""):
    c = MagicMock()
    c.ID = cue_id
    c.InMsec = time_ms
    c.Kind = kind
    c.Comment = comment
    return c


def test_cues_get_unauthorized_returns_401(unauth_client):
    resp = unauth_client.get("/api/tracks/123/cues")
    assert resp.status_code == 401


def test_cues_get_returns_empty_list_when_no_cues(auth_client):
    db = MagicMock()
    db.session.query.return_value.filter_by.return_value.all.return_value = []
    with patch.object(flask_app, "_open_db", return_value=db):
        resp = auth_client.get("/api/tracks/123/cues")
    assert resp.status_code == 200
    assert resp.get_json() == []


def test_cues_get_returns_sorted_by_time_ms(auth_client):
    """The route must sort cues ascending by time_ms regardless of DB order."""
    db = MagicMock()
    db.session.query.return_value.filter_by.return_value.all.return_value = [
        _cue_mock("c3", 30000),
        _cue_mock("c1", 10000),
        _cue_mock("c2", 20000),
    ]
    with patch.object(flask_app, "_open_db", return_value=db):
        resp = auth_client.get("/api/tracks/123/cues")
    body = resp.get_json()
    assert [c["time_ms"] for c in body] == [10000, 20000, 30000]
    assert [c["id"] for c in body] == ["c1", "c2", "c3"]


def test_cues_get_handles_none_msec(auth_client):
    """InMsec=None should coerce to 0, not crash."""
    db = MagicMock()
    db.session.query.return_value.filter_by.return_value.all.return_value = [
        _cue_mock("c1", None),
    ]
    with patch.object(flask_app, "_open_db", return_value=db):
        resp = auth_client.get("/api/tracks/123/cues")
    assert resp.status_code == 200
    assert resp.get_json()[0]["time_ms"] == 0


# ---------------------------------------------------------------------------
# POST /api/tracks/<id>/cues  — added in PR #66
# ---------------------------------------------------------------------------


def test_cues_post_unauthorized_returns_401(unauth_client):
    resp = unauth_client.post("/api/tracks/123/cues", json={"time_ms": 5000})
    assert resp.status_code == 401


def test_cues_post_blocked_when_rekordbox_running(auth_client):
    with patch.object(flask_app, "rekordbox_is_running", return_value=True):
        resp = auth_client.post("/api/tracks/123/cues", json={"time_ms": 5000})
    assert resp.status_code == 409


def test_cues_post_negative_time_returns_400(auth_client):
    with patch.object(flask_app, "rekordbox_is_running", return_value=False):
        resp = auth_client.post("/api/tracks/123/cues", json={"time_ms": -1})
    assert resp.status_code == 400
    assert ">= 0" in resp.get_json()["error"]


def test_cues_post_missing_time_ms_returns_400(auth_client):
    """Default for missing time_ms is -1, which must be rejected."""
    with patch.object(flask_app, "rekordbox_is_running", return_value=False):
        resp = auth_client.post("/api/tracks/123/cues", json={})
    assert resp.status_code == 400


def test_cues_post_non_int_time_ms_returns_400(auth_client):
    with patch.object(flask_app, "rekordbox_is_running", return_value=False):
        resp = auth_client.post("/api/tracks/123/cues", json={"time_ms": "not-a-number"})
    assert resp.status_code == 400
    assert "integer" in resp.get_json()["error"]


# ---------------------------------------------------------------------------
# POST/DELETE /api/tracks/<id>/playlists/<pid>  — silent-failure regression
# guard. Mobile listen client used to swallow non-2xx responses, leaving
# checkmarks stuck for writes that never persisted. Server must return a
# parseable {"error": "..."} body so the client can surface a toast.
# ---------------------------------------------------------------------------


def test_playlist_add_unauthorized_returns_401(unauth_client):
    resp = unauth_client.post("/api/tracks/123/playlists/456")
    assert resp.status_code == 401


def test_playlist_add_blocked_when_rekordbox_running_returns_parseable_error(auth_client):
    with patch.object(flask_app, "rekordbox_is_running", return_value=True):
        resp = auth_client.post("/api/tracks/123/playlists/456")
    assert resp.status_code == 409
    body = resp.get_json()
    assert "error" in body
    assert "Rekordbox" in body["error"]


def test_playlist_remove_unauthorized_returns_401(unauth_client):
    resp = unauth_client.delete("/api/tracks/123/playlists/456")
    assert resp.status_code == 401


def test_playlist_remove_blocked_when_rekordbox_running_returns_parseable_error(auth_client):
    with patch.object(flask_app, "rekordbox_is_running", return_value=True):
        resp = auth_client.delete("/api/tracks/123/playlists/456")
    assert resp.status_code == 409
    body = resp.get_json()
    assert "error" in body
    assert "Rekordbox" in body["error"]


# ---------------------------------------------------------------------------
# GET /api/listen/tree  — wraps get_listen_tree.py
# ---------------------------------------------------------------------------


def test_listen_tree_unauthorized_returns_401(unauth_client):
    resp = unauth_client.get("/api/listen/tree")
    assert resp.status_code == 401


def test_listen_tree_returns_subprocess_json(auth_client):
    fake_result = MagicMock()
    fake_result.returncode = 0
    fake_result.stdout = '{"tree": [{"id": 1, "name": "Test", "type": "playlist"}]}'
    fake_result.stderr = ""
    with (
        patch("app.subprocess.run", return_value=fake_result),
        patch.object(flask_app, "rekordbox_is_running", return_value=False),
    ):
        resp = auth_client.get("/api/listen/tree")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["tree"][0]["name"] == "Test"
    assert body["rekordbox_running"] is False  # injected for mobile UI


def test_listen_tree_subprocess_failure_returns_500(auth_client):
    fake_result = MagicMock()
    fake_result.returncode = 1
    fake_result.stdout = ""
    fake_result.stderr = "Database not found"
    with patch("app.subprocess.run", return_value=fake_result):
        resp = auth_client.get("/api/listen/tree")
    assert resp.status_code == 500
    assert "Database not found" in resp.get_json()["error"]


def test_listen_tree_passes_db_path_to_script(auth_client, tmp_path):
    fake_result = MagicMock()
    fake_result.returncode = 0
    fake_result.stdout = '{"tree": []}'
    with (
        patch("app.subprocess.run", return_value=fake_result) as mock_run,
        patch.object(flask_app, "rekordbox_is_running", return_value=False),
    ):
        auth_client.get("/api/listen/tree")
    cmd = mock_run.call_args[0][0]
    assert "--db-path" in cmd


# ---------------------------------------------------------------------------
# GET /api/listen/playlist/<int:pid>/tracks  — PR #65 regression guard
# ---------------------------------------------------------------------------


def test_listen_tracks_unauthorized_returns_401(unauth_client):
    resp = unauth_client.get("/api/listen/playlist/42/tracks")
    assert resp.status_code == 401


def test_listen_tracks_returns_subprocess_json_passthrough(auth_client):
    """The route must pass the script's JSON straight through, preserving smart_unavailable etc."""
    fake_result = MagicMock()
    fake_result.returncode = 0
    fake_result.stdout = json.dumps(
        {"playlist_name": "Smart", "playlist_id": 42, "tracks": [], "smart_unavailable": False}
    )
    fake_result.stderr = ""
    with patch("app.subprocess.run", return_value=fake_result):
        resp = auth_client.get("/api/listen/playlist/42/tracks")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["playlist_name"] == "Smart"
    assert body["smart_unavailable"] is False
    # PR #65: smart_unavailable must be present (not stripped) so UI can render correctly


def test_listen_tracks_passes_playlist_id_to_script(auth_client):
    fake_result = MagicMock()
    fake_result.returncode = 0
    fake_result.stdout = '{"tracks": []}'
    with patch("app.subprocess.run", return_value=fake_result) as mock_run:
        auth_client.get("/api/listen/playlist/777/tracks")
    cmd = mock_run.call_args[0][0]
    assert "--playlist-id" in cmd
    assert "777" in cmd


def test_listen_tracks_subprocess_error_returns_500(auth_client):
    fake_result = MagicMock()
    fake_result.returncode = 1
    fake_result.stdout = ""
    fake_result.stderr = "Playlist 42 not found"
    with patch("app.subprocess.run", return_value=fake_result):
        resp = auth_client.get("/api/listen/playlist/42/tracks")
    assert resp.status_code == 500


# ---------------------------------------------------------------------------
# GET /api/listen/all-tracks
# ---------------------------------------------------------------------------


def test_listen_all_tracks_unauthorized_returns_401(unauth_client):
    resp = unauth_client.get("/api/listen/all-tracks")
    assert resp.status_code == 401


def test_listen_all_tracks_passes_special_id_to_script(auth_client):
    """all-tracks must pass --playlist-id all so the script switches to all-tracks mode."""
    fake_result = MagicMock()
    fake_result.returncode = 0
    fake_result.stdout = '{"playlist_name":"All Tracks","playlist_id":"all","tracks":[]}'
    with patch("app.subprocess.run", return_value=fake_result) as mock_run:
        resp = auth_client.get("/api/listen/all-tracks")
    assert resp.status_code == 200
    cmd = mock_run.call_args[0][0]
    assert "--playlist-id" in cmd
    assert "all" in cmd
