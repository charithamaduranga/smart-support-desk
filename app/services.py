"""Ticket business logic and validation, shared by the HTML pages and the REST API."""
import json
import re
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints, field_validator, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.classifier import TicketClassifier
from app.models import Ticket

MAX_MESSAGE_LENGTH = 5000
_EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")

Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
Email = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=254)]
Message = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_MESSAGE_LENGTH)]
Status = Literal["open", "in_progress", "resolved"]
Priority = Literal["low", "normal", "high"]


class TicketCreate(BaseModel):
    name: Name
    email: Email
    message: Message

    @field_validator("email")
    @classmethod
    def _valid_email(cls, v: str) -> str:
        if not _EMAIL.fullmatch(v):
            raise ValueError("must be a valid email address")
        return v


class TicketUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Status | None = None
    priority: Priority | None = None

    @model_validator(mode="after")
    def _something_to_change(self):
        if self.status is None and self.priority is None:
            raise ValueError("provide status and/or priority")
        return self


class TicketReceipt(BaseModel):
    """What a customer gets back: no internal classification or draft details."""
    id: int
    status: str


class TicketOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    email: str
    message: str
    predicted_category: str
    confidence: float
    priority: str
    status: str
    created_at: datetime
    model_version: str
    suggested_reply: str | None
    reply_status: str | None
    reply_error: str | None
    reply_sources: list[dict]

    @field_validator("reply_sources", mode="before")
    @classmethod
    def _parse_sources(cls, v):
        return json.loads(v) if isinstance(v, str) else (v or [])


def create_ticket(db: Session, classifier: TicketClassifier, data: TicketCreate, drafter=None) -> Ticket:
    prediction = classifier.classify(data.message)
    ticket = Ticket(name=data.name, email=data.email, message=data.message,
                    predicted_category=prediction.category, confidence=prediction.confidence,
                    model_version=prediction.model_version)
    db.add(ticket)
    db.commit()  # the ticket is saved before any LLM call, so a draft failure cannot lose it
    attach_draft(db, ticket, drafter)
    return ticket


def attach_draft(db: Session, ticket: Ticket, drafter) -> None:
    if drafter is None:
        ticket.reply_status = "disabled"
    else:
        try:
            draft = drafter.draft(ticket.message)  # only the message goes to the LLM
            ticket.suggested_reply = draft.text
            ticket.reply_status = draft.status
            ticket.reply_sources = json.dumps(draft.sources)
        except Exception as exc:  # any provider/parsing failure: keep the ticket, record why
            ticket.reply_status = "error"
            ticket.reply_error = f"{type(exc).__name__}: {exc}"[:500]
    db.commit()


def list_tickets(db: Session, category: str = "") -> list[Ticket]:
    query = select(Ticket).order_by(Ticket.created_at.desc(), Ticket.id.desc())
    if category:
        query = query.where(Ticket.predicted_category == category)
    return list(db.scalars(query))


def stored_categories(db: Session) -> list[str]:
    return list(db.scalars(select(Ticket.predicted_category).distinct().order_by(Ticket.predicted_category)))


def update_ticket(db: Session, ticket: Ticket, update: TicketUpdate) -> Ticket:
    if update.status is not None:
        ticket.status = update.status
    if update.priority is not None:
        ticket.priority = update.priority
    db.commit()
    return ticket


def error_messages(exc) -> dict[str, str]:
    """Pydantic ValidationError -> {field: readable message} for the HTML forms."""
    out = {}
    for err in exc.errors():
        field = str(err["loc"][-1]) if err["loc"] else "form"
        msg = err["msg"].removeprefix("Value error, ")
        out.setdefault(field, msg)
    return out
