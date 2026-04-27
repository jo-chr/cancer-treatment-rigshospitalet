"""Input + output guardrails.

The hard rule from the case description: NEVER give medical advice. Only practical
information from the provided documents. It is enforced in two places:

1. ``classify_intent`` — cheap LLM classifier on the *input* so we can refuse
   medical questions before retrieval.
2. ``check_output`` — LLM-as-judge on the *output* to catch leakage where the
   model drifted into clinical territory anyway.

Both rules also catch acute emergencies and route them to 112 / 1813.
"""
from __future__ import annotations

from enum import Enum

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.config import settings


class Intent(str, Enum):
    PRACTICAL = "practical"          # logistics, hotel, food, visiting hours, etc.
    MEDICAL = "medical"              # diagnosis, prognosis, dosing, symptoms interpretation
    EMERGENCY = "emergency"          # acute symptoms, suicidal ideation, severe bleeding
    OUT_OF_SCOPE = "out_of_scope"    # unrelated to Rigshospitalet care


class IntentDecision(BaseModel):
    intent: Intent
    reason: str = Field(description="Short justification, max 1 sentence.")


_INTENT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You triage messages from cancer patients and their relatives writing "
        "to a Rigshospitalet (Copenhagen) information assistant.\n\n"
        "Classify the user's message into exactly ONE category:\n"
        "- practical: logistics, visiting hours, hotel, food, parking, what to "
        "  bring, fasting before surgery, contact info, general what-to-expect.\n"
        "- medical: anything that requires clinical judgement — diagnosis, "
        "  prognosis, medication dosing, symptom interpretation, treatment "
        "  recommendations, lab result meaning.\n"
        "- emergency: acute danger (heavy bleeding, chest pain, suicidal "
        "  thoughts, severe breathing trouble).\n"
        "- out_of_scope: unrelated to Rigshospitalet care.\n\n"
        "Be conservative: when in doubt between practical and medical, choose "
        "medical.",
    ),
    ("human", "{question}"),
])


class OutputCheck(BaseModel):
    is_safe: bool = Field(
        description="False if the answer contains medical advice, diagnosis, "
        "prognosis, dosing, or unsupported clinical claims."
    )
    reason: str


_OUTPUT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You audit answers from a hospital information assistant. Mark the "
        "answer UNSAFE if it: gives a diagnosis, interprets symptoms, "
        "recommends or doses medication, predicts prognosis, or makes "
        "clinical claims not present in the supplied source excerpts. "
        "Practical logistics (hours, addresses, what to bring, who to call) "
        "are always safe.",
    ),
    (
        "human",
        "User question:\n{question}\n\n"
        "Source excerpts:\n{sources}\n\n"
        "Proposed answer:\n{answer}",
    ),
])


def _llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )


def classify_intent(question: str) -> IntentDecision:
    chain = _INTENT_PROMPT | _llm().with_structured_output(IntentDecision)
    return chain.invoke({"question": question})


def check_output(question: str, answer: str, sources: str) -> OutputCheck:
    chain = _OUTPUT_PROMPT | _llm().with_structured_output(OutputCheck)
    return chain.invoke({"question": question, "answer": answer, "sources": sources})


# --- Canned responses ---

REFUSAL_MEDICAL = (
    "Jeg kan desværre ikke svare på medicinske spørgsmål. "
    "Kontakt venligst din afdeling i dagtimerne, eller ring til "
    "Lægevagten på 1813 uden for åbningstid. "
    "Er det livstruende, ring 112."
)

REFUSAL_EMERGENCY = (
    "Hvis dette er akut: ring 112 med det samme. "
    "Er det ikke livstruende men haster, ring til 1813. "
    "Jeg er kun et informationsværktøj og kan ikke hjælpe i en nødsituation."
)

REFUSAL_OUT_OF_SCOPE = (
    "Det spørgsmål ligger uden for, hvad jeg kan hjælpe med. "
    "Jeg kan svare på praktiske spørgsmål om dit forløb på Rigshospitalet "
    "(indlæggelse, patienthotel, besøg, mad, kontakt m.m.)."
)

NO_ANSWER = (
    "Jeg kunne ikke finde svaret i informationsmaterialet. "
    "Kontakt venligst afdelingen i dagtimerne for at få et præcist svar."
)
