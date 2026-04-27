"""LangGraph state machine for the Rigshospitalet assistant.

Flow:

    START
      └─► classify_intent
            ├─ medical      ──► refuse_medical    ──► END
            ├─ emergency    ──► refuse_emergency  ──► END
            ├─ out_of_scope ──► refuse_oos        ──► END
            └─ practical    ──► retrieve
                                  └─► generate
                                        └─► guard_output
                                              ├─ unsafe ──► refuse_medical ──► END
                                              └─ safe   ──► END
"""
from __future__ import annotations

from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from src.config import settings
from src.guards import (
    NO_ANSWER,
    REFUSAL_EMERGENCY,
    REFUSAL_MEDICAL,
    REFUSAL_OUT_OF_SCOPE,
    Intent,
    check_output,
    classify_intent,
)
from src.retriever import annotate_freshness, retrieve


class AgentState(TypedDict, total=False):
    question: str
    intent: str
    intent_reason: str
    sources: list[Document]
    scores: list[float]
    answer: str
    citations: list[str]
    trace: list[str]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Du er en venlig informationsassistent for kræftpatienter og pårørende "
        "på Rigshospitalet. Svar KUN ud fra de medfølgende uddrag fra "
        "hospitalets informationsmateriale.\n\n"
        "Regler:\n"
        "1. Svar på samme sprog som spørgsmålet (typisk dansk).\n"
        "2. Vær kort, rolig og konkret. Patienten er bange.\n"
        "3. Giv ALDRIG medicinske råd, diagnoser, prognoser eller "
        "   medicindosering. Henvis i stedet til afdelingen / 1813 / 112.\n"
        "4. Hvis svaret ikke står i uddragene: sig at du ikke kan finde det, "
        "   og henvis til afdelingen.\n"
        "5. Slut svaret med 'Kilder:' og de filnavne du brugte.\n"
        "6. Hvis et uddrag er markeret som forældet, nævn det kort.",
    ),
    (
        "human",
        "Spørgsmål:\n{question}\n\n"
        "Uddrag fra informationsmaterialet:\n{context}\n\n"
        "{freshness_warnings}",
    ),
])


def _llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        temperature=0.3,                     #Factual Q&A grounded in retrieved docs -> 0–0.3
    )


def node_classify(state: AgentState) -> AgentState:
    decision = classify_intent(state["question"])
    trace = state.get("trace", []) + [f"intent={decision.intent.value} ({decision.reason})"]
    return {"intent": decision.intent.value, "intent_reason": decision.reason, "trace": trace}


def node_retrieve(state: AgentState) -> AgentState:
    hits = retrieve(state["question"])
    docs = [d for d, _ in hits]
    scores = [s for _, s in hits]
    trace = state.get("trace", []) + [f"retrieved={len(docs)} chunks"]
    return {"sources": docs, "scores": scores, "trace": trace}


def _format_context(docs: list[Document]) -> tuple[str, list[str]]:
    blocks: list[str] = []
    warnings: list[str] = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source_file", "unknown")
        blocks.append(f"[{i}] ({src})\n{d.page_content}")
        w = annotate_freshness(d)
        if w:
            warnings.append(w)
    return "\n\n".join(blocks), warnings


def node_generate(state: AgentState) -> AgentState:
    docs = state.get("sources", [])
    if not docs:
        return {
            "answer": NO_ANSWER,
            "citations": [],
            "trace": state.get("trace", []) + ["no_context -> NO_ANSWER"],
        }

    context, warnings = _format_context(docs)
    chain = ANSWER_PROMPT | _llm()
    msg = chain.invoke({
        "question": state["question"],
        "context": context,
        "freshness_warnings": "\n".join(warnings),
    })
    citations = sorted({d.metadata.get("source_file", "?") for d in docs})
    trace = state.get("trace", []) + [f"generated, citations={citations}"]
    return {"answer": msg.content, "citations": citations, "trace": trace}


def node_guard_output(state: AgentState) -> AgentState:
    docs = state.get("sources", [])
    context, _ = _format_context(docs)
    verdict = check_output(state["question"], state.get("answer", ""), context)
    trace = state.get("trace", []) + [f"output_safe={verdict.is_safe} ({verdict.reason})"]
    if not verdict.is_safe:
        return {"answer": REFUSAL_MEDICAL, "citations": [], "trace": trace}
    return {"trace": trace}


def node_refuse_medical(state: AgentState) -> AgentState:
    return {"answer": REFUSAL_MEDICAL, "citations": [],
            "trace": state.get("trace", []) + ["refuse_medical"]}


def node_refuse_emergency(state: AgentState) -> AgentState:
    return {"answer": REFUSAL_EMERGENCY, "citations": [],
            "trace": state.get("trace", []) + ["refuse_emergency"]}


def node_refuse_oos(state: AgentState) -> AgentState:
    return {"answer": REFUSAL_OUT_OF_SCOPE, "citations": [],
            "trace": state.get("trace", []) + ["refuse_oos"]}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_after_intent(state: AgentState) -> str:
    intent = state.get("intent")
    if intent == Intent.MEDICAL.value:
        return "refuse_medical"
    if intent == Intent.EMERGENCY.value:
        return "refuse_emergency"
    if intent == Intent.OUT_OF_SCOPE.value:
        return "refuse_oos"
    return "retrieve"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("classify", node_classify)
    g.add_node("retrieve", node_retrieve)
    g.add_node("generate", node_generate)
    g.add_node("guard_output", node_guard_output)
    g.add_node("refuse_medical", node_refuse_medical)
    g.add_node("refuse_emergency", node_refuse_emergency)
    g.add_node("refuse_oos", node_refuse_oos)

    g.add_edge(START, "classify")
    g.add_conditional_edges(
        "classify",
        route_after_intent,
        {
            "retrieve": "retrieve",
            "refuse_medical": "refuse_medical",
            "refuse_emergency": "refuse_emergency",
            "refuse_oos": "refuse_oos",
        },
    )
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "guard_output")
    g.add_edge("guard_output", END)
    g.add_edge("refuse_medical", END)
    g.add_edge("refuse_emergency", END)
    g.add_edge("refuse_oos", END)

    return g.compile()


# Singleton-ish accessor for the API / CLI
_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def ask(question: str) -> AgentState:
    return get_graph().invoke({"question": question})
