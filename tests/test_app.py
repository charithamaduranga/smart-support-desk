import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.main import create_app


def test_root_redirects_to_ticket_list(client):
    r = client.get("/")
    assert r.status_code == 303 and r.headers["location"] == "/tickets"


def test_health_reports_injected_model_version(client):
    assert client.get("/health").json() == {"status": "ok", "model_version": "fake-v1"}


def test_lifespan_loads_classifier_once_for_many_requests(settings, monkeypatch, fake_classifier_cls):
    loaded = []

    def fake_from_disk(path):
        loaded.append(path)
        return fake_classifier_cls()

    monkeypatch.setattr(main_module.TicketClassifier, "from_disk", staticmethod(fake_from_disk))
    app = create_app(settings)  # no injected classifier: the real startup path
    assert loaded == []  # creating the app does not load the model
    with TestClient(app) as c:
        for _ in range(5):
            assert c.get("/health").status_code == 200
        first = app.state.classifier
    assert loaded == [settings.model_path]
    assert app.state.classifier is first


def test_injected_classifier_skips_disk_loading(settings, monkeypatch, fake_classifier):
    def fail(path):
        raise AssertionError("from_disk must not be called when a classifier is injected")

    monkeypatch.setattr(main_module.TicketClassifier, "from_disk", staticmethod(fail))
    fake = fake_classifier
    app = create_app(settings, classifier=fake)
    with TestClient(app):
        assert app.state.classifier is fake


def test_startup_fails_clearly_when_model_missing(settings):
    # settings.model_path points at a file that does not exist
    with pytest.raises(FileNotFoundError, match="ml.train"):
        with TestClient(create_app(settings)):
            pass


# --- submission flow ---------------------------------------------------------

def _rows(client):
    from sqlalchemy import select
    from app.models import Ticket
    with client.app.state.session_factory() as s:
        return s.scalars(select(Ticket).order_by(Ticket.id)).all()


def test_new_ticket_form_renders(client):
    r = client.get("/tickets/new")
    assert r.status_code == 200
    assert 'name="message"' in r.text and 'action="/tickets"' in r.text


def test_full_flow_submit_classify_persist_redirect_display(client, fake_classifier):
    r = client.post("/tickets", data={"message": "  I was charged but never got a refund  "})

    # redirect (Post/Redirect/Get)
    assert r.status_code == 303
    assert r.headers["location"] == "/tickets/1"
    # classified exactly once, with the stripped text
    assert [text for _, text in fake_classifier.calls] == ["I was charged but never got a refund"]
    # persisted
    (row,) = _rows(client)
    assert (row.message, row.predicted_category, row.confidence, row.model_version) == (
        "I was charged but never got a refund", "REFUND", 0.87, "fake-v1")
    assert row.created_at.tzinfo is not None
    assert row.suggested_reply is None and row.reply_error is None
    # displayed
    page = client.get(r.headers["location"])
    assert page.status_code == 200
    for expected in ["Ticket #1", "I was charged but never got a refund", "REFUND", "87.0%", "fake-v1", "UTC"]:
        assert expected in page.text


def test_multiple_requests_reuse_the_same_classifier_instance(settings, monkeypatch, fake_classifier_cls):
    loaded = []

    def fake_from_disk(path):
        clf = fake_classifier_cls()
        loaded.append(clf)
        return clf

    monkeypatch.setattr(main_module.TicketClassifier, "from_disk", staticmethod(fake_from_disk))
    with TestClient(create_app(settings), follow_redirects=False) as c:
        for i in range(5):
            assert c.post("/tickets", data={"message": f"ticket {i}"}).status_code == 303
    assert len(loaded) == 1
    instances_used = {instance_id for instance_id, _ in loaded[0].calls}
    assert instances_used == {id(loaded[0])} and len(loaded[0].calls) == 5


@pytest.mark.parametrize("data", [{"message": ""}, {"message": "   \n\t "}, {}])
def test_blank_or_missing_message_is_rejected(client, fake_classifier, data):
    r = client.post("/tickets", data=data)
    assert r.status_code == 400
    assert "Please describe your issue." in r.text
    assert fake_classifier.calls == [] and _rows(client) == []


def test_too_long_message_is_rejected_and_kept_in_form(client, fake_classifier):
    long_msg = "x" * (main_module.MAX_MESSAGE_LENGTH + 1)
    r = client.post("/tickets", data={"message": long_msg})
    assert r.status_code == 400 and "too long" in r.text
    assert long_msg in r.text  # user's input is not lost
    assert fake_classifier.calls == [] and _rows(client) == []


def test_message_at_max_length_is_accepted(client):
    r = client.post("/tickets", data={"message": "y" * main_module.MAX_MESSAGE_LENGTH})
    assert r.status_code == 303 and len(_rows(client)) == 1


def test_html_in_message_is_escaped(client):
    payload = '<script>alert("x")</script><b>bold</b>'
    r = client.post("/tickets", data={"message": payload})
    page = client.get(r.headers["location"]).text
    assert "<script>alert" not in page and "<b>bold</b>" not in page
    assert "&lt;script&gt;" in page and "&lt;b&gt;bold&lt;/b&gt;" in page
    assert _rows(client)[0].message == payload  # stored as typed, escaped only on output


def test_html_is_escaped_when_redisplayed_after_error(client):
    payload = "<script>x</script>" + "z" * main_module.MAX_MESSAGE_LENGTH
    r = client.post("/tickets", data={"message": payload})
    assert r.status_code == 400 and "<script>x</script>" not in r.text and "&lt;script&gt;" in r.text


def test_unknown_ticket_returns_404(client):
    assert client.get("/tickets/999").status_code == 404


def test_non_integer_ticket_id_is_rejected(client):
    assert client.get("/tickets/abc").status_code == 422


def test_each_test_gets_its_own_database(client, settings, tmp_path):
    assert settings.database_url.endswith(f"{tmp_path.as_posix()}/app.db")
    assert _rows(client) == []  # nothing leaks in from other tests


# --- ticket list -------------------------------------------------------------

def _submit(client, fake_classifier, message, category):
    fake_classifier.category = category
    assert client.post("/tickets", data={"message": message}).status_code == 303


def test_empty_list(client):
    r = client.get("/tickets")
    assert r.status_code == 200 and "No tickets yet." in r.text


def test_list_shows_newest_first_with_category_and_confidence(client, fake_classifier):
    _submit(client, fake_classifier, "first ticket", "ORDER")
    _submit(client, fake_classifier, "second ticket", "REFUND")
    page = client.get("/tickets").text
    assert page.index("second ticket") < page.index("first ticket")
    assert "ORDER" in page and "REFUND" in page and "87.0%" in page
    assert 'href="/tickets/1"' in page and 'href="/tickets/2"' in page


def test_list_filters_by_category(client, fake_classifier):
    _submit(client, fake_classifier, "where is my parcel", "DELIVERY")
    _submit(client, fake_classifier, "refund please", "REFUND")
    page = client.get("/tickets", params={"category": "REFUND"}).text
    assert "refund please" in page and "where is my parcel" not in page
    assert '<option value="REFUND" selected>' in page


def test_list_filter_with_no_matches(client, fake_classifier):
    _submit(client, fake_classifier, "refund please", "REFUND")
    page = client.get("/tickets", params={"category": "ACCOUNT"}).text
    assert "No tickets in category ACCOUNT." in page and "refund please" not in page


def test_list_escapes_html_and_filter_value(client, fake_classifier):
    _submit(client, fake_classifier, "<img src=x onerror=alert(1)>", "REFUND")
    page = client.get("/tickets", params={"category": "<script>"}).text
    assert "<script>" not in page.replace("&lt;script&gt;", "")
    page = client.get("/tickets").text
    assert "<img src=x" not in page and "&lt;img src=x" in page


# --- end-to-end with the real trained model ----------------------------------

@pytest.mark.skipif(not main_module.Settings().model_path.exists(), reason="trained model not present; run ml.train")
def test_end_to_end_with_real_model(settings):
    from dataclasses import replace
    real = replace(settings, model_path=main_module.Settings().model_path)  # temp DB, real model
    with TestClient(create_app(real), follow_redirects=False) as c:
        r = c.post("/tickets", data={"message": "I want my money back for order 123"})
        assert r.status_code == 303
        page = c.get(r.headers["location"]).text
        assert "REFUND" in page and "I want my money back for order 123" in page
        assert c.get("/health").json()["model_version"] != "unknown"
