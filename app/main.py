"""FastAPI application: public customer form, login-protected team pages, REST API.

Run: uv run python -m app
"""
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.orm import Session

from app import services
from app.classifier import TicketClassifier
from app.config import Settings
from app.db import init_db, make_engine, make_session_factory
from app.models import PRIORITIES, STATUSES, Ticket
from app.services import MAX_MESSAGE_LENGTH, TicketCreate, TicketOut, TicketReceipt, TicketUpdate

templates = Jinja2Templates(directory=Path(__file__).resolve().parent / "templates")
_basic = HTTPBasic()
_UNSET = object()


# --- dependencies ------------------------------------------------------------

def get_db(request: Request):
    with request.app.state.session_factory() as session:
        yield session


def get_classifier(request: Request) -> TicketClassifier:
    return request.app.state.classifier


def get_drafter(request: Request):
    return request.app.state.drafter


def require_team(request: Request, credentials: HTTPBasicCredentials = Depends(_basic)) -> str:
    settings: Settings = request.app.state.settings
    if not settings.team_password:
        raise HTTPException(503, "Team login is not configured: set TEAM_PASSWORD")
    user_ok = secrets.compare_digest(credentials.username.encode(), settings.team_username.encode())
    pass_ok = secrets.compare_digest(credentials.password.encode(), settings.team_password.encode())
    if not (user_ok and pass_ok):
        raise HTTPException(401, "Invalid username or password", headers={"WWW-Authenticate": "Basic"})
    return credentials.username


def _get_ticket(db: Session, ticket_id: int) -> Ticket:
    ticket = db.get(Ticket, ticket_id)
    if ticket is None:
        raise HTTPException(404, "Ticket not found")
    return ticket


# --- app factory -------------------------------------------------------------

def create_app(settings: Settings | None = None, classifier: TicketClassifier | None = None,
               drafter=_UNSET) -> FastAPI:
    """Tests pass their own settings (temp DB), classifier and drafter (fake or None)."""
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = make_engine(settings.database_url)
        init_db(engine)
        app.state.settings = settings
        app.state.session_factory = make_session_factory(engine)
        # Loaded once per process and shared by all requests; fails fast if untrained.
        app.state.classifier = classifier or TicketClassifier.from_disk(settings.model_path)
        if drafter is _UNSET:
            from app.replies import build_drafter  # noqa: PLC0415  (optional feature, built from settings)
            app.state.drafter = build_drafter(settings)
        else:
            app.state.drafter = drafter
        yield
        engine.dispose()

    app = FastAPI(title="Smart Support Desk", lifespan=lifespan)
    app.include_router(public)
    app.include_router(team)
    app.include_router(api_public)
    app.include_router(api_team)
    return app


# --- public: customer page ---------------------------------------------------

public = APIRouter()


def _submit_form(request: Request, values: dict | None = None, errors: dict | None = None, status_code=200):
    return templates.TemplateResponse(
        request, "submit.html",
        {"values": values or {}, "errors": errors or {}, "max_length": MAX_MESSAGE_LENGTH},
        status_code=status_code)


@public.get("/")
def submit_page(request: Request):
    return _submit_form(request)


@public.post("/")
def submit(request: Request, name: str = Form(""), email: str = Form(""), message: str = Form(""),
           db: Session = Depends(get_db), clf=Depends(get_classifier), drafter=Depends(get_drafter)):
    values = {"name": name, "email": email, "message": message}
    try:
        data = TicketCreate(**values)
    except ValidationError as exc:
        return _submit_form(request, values, services.error_messages(exc), status_code=400)
    ticket = services.create_ticket(db, clf, data, drafter)
    # Post/Redirect/Get: refreshing the confirmation page must not resubmit the form.
    return RedirectResponse(f"/submitted/{ticket.id}", status_code=303)


@public.get("/submitted/{ticket_id}")
def submitted(request: Request, ticket_id: int, db: Session = Depends(get_db)):
    _get_ticket(db, ticket_id)  # 404 for unknown ids; shows nothing but the ticket number
    return templates.TemplateResponse(request, "submitted.html", {"ticket_id": ticket_id})


@public.get("/health")
def health(clf=Depends(get_classifier), drafter=Depends(get_drafter), db: Session = Depends(get_db)):
    db.execute(text("SELECT 1"))  # proves the database is reachable
    return {"status": "ok", "model_version": clf.model_version, "reply_drafts": drafter is not None}


# --- team pages (login required) ---------------------------------------------

team = APIRouter(prefix="/team", dependencies=[Depends(require_team)])


@team.get("/tickets")
def team_list(request: Request, category: str = "", db: Session = Depends(get_db)):
    return templates.TemplateResponse(request, "team_list.html", {
        "tickets": services.list_tickets(db, category),
        "categories": services.stored_categories(db), "selected": category})


def _detail(request: Request, ticket: Ticket, error: str | None = None, status_code=200):
    return templates.TemplateResponse(request, "team_detail.html", {
        "ticket": ticket, "sources": TicketOut.model_validate(ticket).reply_sources,
        "statuses": STATUSES, "priorities": PRIORITIES, "error": error}, status_code=status_code)


@team.get("/tickets/{ticket_id}")
def team_detail(request: Request, ticket_id: int, db: Session = Depends(get_db)):
    return _detail(request, _get_ticket(db, ticket_id))


@team.post("/tickets/{ticket_id}")
def team_update(request: Request, ticket_id: int, status: str = Form(...), priority: str = Form(...),
                db: Session = Depends(get_db)):
    ticket = _get_ticket(db, ticket_id)
    try:
        update = TicketUpdate(status=status, priority=priority)
    except ValidationError as exc:
        return _detail(request, ticket, "; ".join(services.error_messages(exc).values()), status_code=400)
    services.update_ticket(db, ticket, update)
    return RedirectResponse(f"/team/tickets/{ticket_id}", status_code=303)


# --- REST API ----------------------------------------------------------------

api_public = APIRouter(prefix="/api", tags=["api"])
api_team = APIRouter(prefix="/api", tags=["api"], dependencies=[Depends(require_team)])


@api_public.post("/tickets", status_code=201, response_model=TicketReceipt)
def api_create(data: TicketCreate, db: Session = Depends(get_db), clf=Depends(get_classifier),
               drafter=Depends(get_drafter)):
    return services.create_ticket(db, clf, data, drafter)


@api_team.get("/tickets", response_model=list[TicketOut])
def api_list(category: str = "", db: Session = Depends(get_db)):
    return services.list_tickets(db, category)


@api_team.get("/tickets/{ticket_id}", response_model=TicketOut)
def api_get(ticket_id: int, db: Session = Depends(get_db)):
    return _get_ticket(db, ticket_id)


@api_team.patch("/tickets/{ticket_id}", response_model=TicketOut)
def api_update(ticket_id: int, update: TicketUpdate, db: Session = Depends(get_db)):
    return services.update_ticket(db, _get_ticket(db, ticket_id), update)


app = create_app()  # for uvicorn; nothing is loaded until the lifespan starts
