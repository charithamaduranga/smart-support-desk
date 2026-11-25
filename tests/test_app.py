import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

import app.main as main_module
from app.main import create_app
from app.models import Ticket

FORM = {"name": "Ana Silva", "email": "ana@example.com", "message": "  I was charged but never got a refund  "}


def _rows(client):
    with client.app.state.session_factory() as s:
        return s.scalars(select(Ticket).order_by(Ticket.id)).all()


# --- startup / classifier lifecycle ------------------------------------------

def test_lifespan_loads_classifier_once_and_reuses_it(settings, monkeypatch, fake_classifier_cls, team_auth):
    loaded = []

    def fake_from_disk(path):
        loaded.append(fake_classifier_cls())
        return loaded[-1]

    monkeypatch.setattr(main_module.TicketClassifier, "from_disk", staticmethod(fake_from_disk))
    app = create_app(settings, drafter=None)
    assert loaded == []  # creating the app does not load the model
    with TestClient(app) as c:
        for i in range(3):
            assert c.post("/", data=FORM | {"message": f"html ticket {i}"}, follow_redirects=False).status_code == 303
            assert c.post("/api/tickets", json=FORM | {"message": f"api ticket {i}"}).status_code == 201
    assert len(loaded) == 1
    assert {instance for instance, _ in loaded[0].calls} == {id(loaded[0])}
    assert len(loaded[0].calls) == 6


def test_startup_fails_clearly_when_model_missing(settings):
    with pytest.raises(FileNotFoundError, match="ml.train"):
        with TestClient(create_app(settings, drafter=None)):
            pass


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_version": "fake-v1", "reply_drafts": False}


# --- customer page (public) --------------------------------------------------

def test_customer_form_renders_publicly(client):
    r = client.get("/")
    assert r.status_code == 200
    for field in ("name", "email", "message"):
        assert f'name="{field}"' in r.text


def test_full_flow_submit_classify_persist_then_team_sees_it(client, fake_classifier, team_auth):
    r = client.post("/", data=FORM)
    assert r.status_code == 303 and r.headers["location"] == "/submitted/1"
    confirmation = client.get(r.headers["location"])
    assert confirmation.status_code == 200 and "#1" in confirmation.text
    assert "REFUND" not in confirmation.text  # customer does not see internal classification
    assert [t for _, t in fake_classifier.calls] == ["I was charged but never got a refund"]
    (row,) = _rows(client)
    assert (row.name, row.email, row.message) == ("Ana Silva", "ana@example.com", "I was charged but never got a refund")
    assert (row.predicted_category, row.confidence, row.priority, row.status) == ("REFUND", 0.87, "normal", "open")
    assert row.reply_status == "disabled" and row.suggested_reply is None
    page = client.get("/team/tickets/1", auth=team_auth)
    assert page.status_code == 200
    for expected in ["Ticket #1", "Ana Silva", "ana@example.com", "REFUND", "87.0%", "drafts are disabled"]:
        assert expected in page.text


def test_refreshing_confirmation_page_does_not_create_duplicate(client, fake_classifier):
    location = client.post("/", data=FORM).headers["location"]
    for _ in range(3):  # what a browser refresh does after the redirect: repeat the GET
        assert client.get(location).status_code == 200
    assert len(_rows(client)) == 1 and len(fake_classifier.calls) == 1


def test_public_confirmation_page_exposes_no_customer_data(client):
    data = {"name": "Zelda Quartermain", "email": "zelda.q@example.org", "message": "Secret parcel 4471 issue"}
    location = client.post("/", data=data).headers["location"]
    page = client.get(location).text
    assert "#1" in page
    for value in data.values():
        assert value not in page
    assert "zelda" not in page.lower()  # not even partially (e.g. the email's local part)


def test_confirmation_page_for_unknown_ticket_is_404(client):
    assert client.get("/submitted/999").status_code == 404


@pytest.mark.parametrize("field,value,error", [
    ("name", "   ", "at least 1 character"),
    ("email", "not-an-email", "valid email address"),
    ("message", "", "at least 1 character"),
    ("message", "x" * 5001, "at most 5000 characters"),
])
def test_invalid_submission_is_rejected_with_field_error(client, fake_classifier, field, value, error):
    r = client.post("/", data=FORM | {field: value})
    assert r.status_code == 400 and error in r.text
    assert fake_classifier.calls == [] and _rows(client) == []


def test_html_is_escaped_everywhere(client, team_auth):
    payload = '<script>alert("x")</script>'
    client.post("/", data=FORM | {"name": payload, "message": payload})
    for url in ("/team/tickets", "/team/tickets/1"):
        page = client.get(url, auth=team_auth).text
        assert "<script>alert" not in page and "&lt;script&gt;" in page
    err = client.post("/", data=FORM | {"email": payload}).text
    assert "<script>alert" not in err


# --- team pages (login) ------------------------------------------------------

BAD_AUTH = [None, ("team", "wrong"), ("intruder", "test-pass")]  # none, wrong password, wrong username


@pytest.mark.parametrize("auth", BAD_AUTH)
@pytest.mark.parametrize("url", ["/team/tickets", "/team/tickets/1", "/api/tickets", "/api/tickets/1"])
def test_team_read_routes_require_valid_login(client, url, auth):
    r = client.get(url, auth=auth)
    assert r.status_code == 401 and r.headers["www-authenticate"] == "Basic"


@pytest.mark.parametrize("auth", BAD_AUTH)
@pytest.mark.parametrize("method,url,kwargs", [
    ("POST", "/team/tickets/1", {"data": {"status": "resolved", "priority": "high"}}),
    ("PATCH", "/api/tickets/1", {"json": {"status": "resolved", "priority": "high"}}),
])
def test_team_update_routes_reject_bad_login_and_change_nothing(client, method, url, kwargs, auth):
    client.post("/", data=FORM)
    r = client.request(method, url, auth=auth, **kwargs)
    assert r.status_code == 401
    row = _rows(client)[0]
    assert (row.status, row.priority) == ("open", "normal")


def test_custom_team_username_is_enforced(settings, fake_classifier):
    from dataclasses import replace
    app = create_app(replace(settings, team_username="agent"), classifier=fake_classifier, drafter=None)
    with TestClient(app) as c:
        assert c.get("/api/tickets", auth=("agent", "test-pass")).status_code == 200
        assert c.get("/api/tickets", auth=("team", "test-pass")).status_code == 401


def test_team_routes_unavailable_without_configured_password(settings, fake_classifier):
    from dataclasses import replace
    app = create_app(replace(settings, team_password=None), classifier=fake_classifier, drafter=None)
    with TestClient(app) as c:
        assert c.get("/team/tickets", auth=("team", "")).status_code == 503


def test_team_list_shows_priority_status_and_filters(client, fake_classifier, team_auth):
    client.post("/", data=FORM | {"message": "where is my parcel"})
    fake_classifier.category = "ACCOUNT"
    client.post("/", data=FORM | {"message": "reset my password"})
    page = client.get("/team/tickets", auth=team_auth).text
    assert page.index("reset my password") < page.index("where is my parcel")  # newest first
    assert "normal" in page and "open" in page
    filtered = client.get("/team/tickets", params={"category": "ACCOUNT"}, auth=team_auth).text
    assert "reset my password" in filtered and "where is my parcel" not in filtered


def test_team_updates_status_and_priority(client, team_auth):
    client.post("/", data=FORM)
    r = client.post("/team/tickets/1", data={"status": "resolved", "priority": "high"}, auth=team_auth)
    assert r.status_code == 303 and r.headers["location"] == "/team/tickets/1"
    assert (_rows(client)[0].status, _rows(client)[0].priority) == ("resolved", "high")


def test_team_update_rejects_invalid_status(client, team_auth):
    client.post("/", data=FORM)
    r = client.post("/team/tickets/1", data={"status": "deleted", "priority": "normal"}, auth=team_auth)
    assert r.status_code == 400 and _rows(client)[0].status == "open"


def test_unknown_ticket_404(client, team_auth):
    assert client.get("/team/tickets/999", auth=team_auth).status_code == 404


# --- REST API ----------------------------------------------------------------

def test_api_create_is_public_and_returns_receipt_only(client):
    r = client.post("/api/tickets", json=FORM)
    assert r.status_code == 201 and r.json() == {"id": 1, "status": "open"}


def test_api_create_returns_clear_validation_errors(client):
    r = client.post("/api/tickets", json={"name": "", "email": "bad", "message": "hi"})
    assert r.status_code == 422
    errors = {e["loc"][-1]: e["msg"] for e in r.json()["detail"]}
    assert "at least 1 character" in errors["name"] and "valid email address" in errors["email"]
    assert _rows(client) == []


def test_api_list_get_and_filter(client, fake_classifier, team_auth):
    client.post("/api/tickets", json=FORM)
    fake_classifier.category = "ACCOUNT"
    client.post("/api/tickets", json=FORM | {"message": "reset my password"})
    assert [t["id"] for t in client.get("/api/tickets", auth=team_auth).json()] == [2, 1]
    only = client.get("/api/tickets", params={"category": "ACCOUNT"}, auth=team_auth).json()
    assert [t["predicted_category"] for t in only] == ["ACCOUNT"]
    one = client.get("/api/tickets/1", auth=team_auth).json()
    assert one["email"] == "ana@example.com" and one["status"] == "open" and one["reply_sources"] == []
    assert client.get("/api/tickets/99", auth=team_auth).json() == {"detail": "Ticket not found"}


def test_api_patch_status_and_priority(client, team_auth):
    client.post("/api/tickets", json=FORM)
    r = client.patch("/api/tickets/1", json={"status": "in_progress"}, auth=team_auth)
    assert r.status_code == 200 and (r.json()["status"], r.json()["priority"]) == ("in_progress", "normal")
    r = client.patch("/api/tickets/1", json={"priority": "high"}, auth=team_auth)
    assert r.json()["priority"] == "high"


@pytest.mark.parametrize("body", [{}, {"status": "deleted"}, {"priority": "urgent"}, {"category": "ORDER"}])
def test_api_patch_rejects_invalid_updates(client, team_auth, body):
    client.post("/api/tickets", json=FORM)
    assert client.patch("/api/tickets/1", json=body, auth=team_auth).status_code == 422
    assert _rows(client)[0].status == "open"


def test_each_test_gets_its_own_database(client, settings, tmp_path):
    assert settings.database_url.endswith(f"{tmp_path.as_posix()}/app.db") and _rows(client) == []


# --- end-to-end with the real trained model ----------------------------------

@pytest.mark.skipif(not main_module.Settings().model_path.exists(), reason="trained model not present; run ml.train")
def test_end_to_end_with_real_model(settings, team_auth):
    from dataclasses import replace
    real = replace(settings, model_path=main_module.Settings().model_path)
    with TestClient(create_app(real, drafter=None)) as c:
        r = c.post("/api/tickets", json=FORM | {"message": "I want my money back for order 123"})
        assert c.get(f"/api/tickets/{r.json()['id']}", auth=team_auth).json()["predicted_category"] == "REFUND"
