"""FastAPI application: server-rendered support ticket desk.

Run: uv run uvicorn app.main:app --reload
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.classifier import TicketClassifier
from app.config import Settings
from app.db import init_db, make_engine, make_session_factory
from app.models import Ticket

MAX_MESSAGE_LENGTH = 5000
templates = Jinja2Templates(directory=Path(__file__).resolve().parent / "templates")
router = APIRouter()


def get_db(request: Request):
    with request.app.state.session_factory() as session:
        yield session


def get_classifier(request: Request) -> TicketClassifier:
    return request.app.state.classifier


def create_app(settings: Settings | None = None, classifier: TicketClassifier | None = None) -> FastAPI:
    """Build the app. Tests pass their own settings (temp DB) and/or classifier."""
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = make_engine(settings.database_url)
        init_db(engine)
        app.state.session_factory = make_session_factory(engine)
        # Loaded once per process and shared by all requests; fails fast if untrained.
        app.state.classifier = classifier or TicketClassifier.from_disk(settings.model_path)
        yield
        engine.dispose()

    app = FastAPI(title="Smart Support Desk", lifespan=lifespan)
    app.include_router(router)
    return app


@router.get("/", include_in_schema=False)
def index():
    return RedirectResponse("/tickets", status_code=303)


@router.get("/tickets")
def list_tickets(request: Request, category: str = "", db: Session = Depends(get_db)):
    query = select(Ticket).order_by(Ticket.created_at.desc(), Ticket.id.desc())
    if category:
        query = query.where(Ticket.predicted_category == category)
    # Filter options come from stored tickets, so the page has no hard-coded label list.
    categories = db.scalars(select(Ticket.predicted_category).distinct().order_by(Ticket.predicted_category)).all()
    return templates.TemplateResponse(
        request, "tickets_list.html",
        {"tickets": db.scalars(query).all(), "categories": categories, "selected": category},
    )


@router.get("/tickets/new")
def new_ticket_form(request: Request):
    return _form(request)


@router.post("/tickets")
def create_ticket(
    request: Request,
    message: str = Form(""),
    db: Session = Depends(get_db),
    clf: TicketClassifier = Depends(get_classifier),
):
    message = message.strip()
    if not message:
        return _form(request, message, "Please describe your issue.", status_code=400)
    if len(message) > MAX_MESSAGE_LENGTH:
        return _form(request, message, f"Message is too long (max {MAX_MESSAGE_LENGTH} characters).",
                     status_code=400)
    prediction = clf.classify(message)
    ticket = Ticket(message=message, predicted_category=prediction.category,
                    confidence=prediction.confidence, model_version=prediction.model_version)
    db.add(ticket)
    db.commit()
    # Post/Redirect/Get: refreshing the detail page cannot resubmit the form.
    return RedirectResponse(f"/tickets/{ticket.id}", status_code=303)


@router.get("/tickets/{ticket_id}")
def ticket_detail(request: Request, ticket_id: int, db: Session = Depends(get_db)):
    ticket = db.get(Ticket, ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return templates.TemplateResponse(request, "ticket_detail.html", {"ticket": ticket})


def _form(request: Request, message: str = "", error: str | None = None, status_code: int = 200):
    return templates.TemplateResponse(
        request, "ticket_new.html",
        {"message": message, "error": error, "max_length": MAX_MESSAGE_LENGTH},
        status_code=status_code,
    )


@router.get("/health")
def health(clf: TicketClassifier = Depends(get_classifier), db: Session = Depends(get_db)):
    db.execute(text("SELECT 1"))  # proves the database is reachable
    return {"status": "ok", "model_version": clf.model_version}


app = create_app()  # for uvicorn; nothing is loaded until the lifespan starts
