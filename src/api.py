"""FastAPI surface for the assistant.

Run:
    uvicorn src.api:app --reload
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from src.graph import ask

app = FastAPI(title="Rigshospitalet Patient Info Assistant", version="0.1.0")


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    citations: list[str]
    intent: str
    trace: list[str]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest) -> AskResponse:
    state = ask(req.question)
    return AskResponse(
        answer=state.get("answer", ""),
        citations=state.get("citations", []),
        intent=state.get("intent", "unknown"),
        trace=state.get("trace", []),
    )
