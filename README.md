# Rigshospitalet — Enhancing Cancer Treatment

LLM-powered assistant that helps cancer patients and relatives at Rigshospitalet
get **practical** information from the official patient information package
24/7 — without ever giving medical advice.

## Project Structure

```
hospital_documents/         # 19 source PDFs (thyroid / patient / paaroerende)
src/
  config.py                 # Env-driven settings
  ingest.py                 # PDFs -> chunks -> Chroma vector store
  retriever.py              # Similarity search + freshness annotation
  guards.py                 # Intent classifier + output safety judge
  graph.py                  # LangGraph state machine (the agent)
  api.py                    # FastAPI surface
  cli.py                    # Interactive CLI for the live demo
eval/
  questions.yaml            # 12 hand-crafted test cases
  run.py                    # Scorecard: intent / refusal / citation accuracy
```

## Architecture (1-minute version)

```
user ──► classify_intent ──┬─ medical    ─► refuse (1813)
                           ├─ emergency  ─► refuse (112)
                           ├─ out_scope  ─► refuse (scope message)
                           └─ practical  ─► retrieve ─► generate ─► guard_output ─► answer + citations
```

- **Retrieval**: Chroma over 800-token chunks of the 19 PDFs, with
  `category`, `source_file` and `last_modified` metadata.
- **Generation**: GPT-4o-mini, Danish prompt, must cite source filenames,
  refuses if context is empty.
- **Guardrails**: input-side intent classifier (medical / emergency /
  out-of-scope / practical) **plus** output-side LLM-as-judge that catches
  any clinical leakage that slipped through.
- **Freshness**: documents older than `STALE_DAYS` get a warning attached
  to the answer (the brief calls out conflicting / outdated docs).

## Quickstart

```powershell
# 1. install
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. configure
Copy-Item .env.example .env
# edit .env and put your OPENAI_API_KEY

# 3. build the vector store (one-off)
python -m src.ingest

# 4. talk to it
python -m src.cli

# 5. or expose it as an API
uvicorn src.api:app --reload
# open http://127.0.0.1:8000/docs in your browser

# 6. run the eval
python -m eval.run
```

## Design choices

| Choice | Why |
|---|---|
| RAG, not fine-tuning | Docs change, traceability is mandatory, citations are non-negotiable. |
| LangGraph over plain chains | Explicit, auditable state machine — easy to show the Chief Surgeon where the medical-advice guard sits. |
| Chroma + local files | Live demo runs on a laptop, zero infra. |
| GPT-4o-mini default | Cheap, fast, multilingual. Swap for Azure OpenAI EU for production GDPR. |
| Two-layer guardrails | Input classifier is cheap and catches 95%; output judge catches drift. |

## Path to production

- **Hosting**: Azure OpenAI in EU region; Rigshospitalet AD-integrated.
- **PII**: Presidio-style scrubbing pre-LLM; no raw inputs in logs.
- **Governance**: weekly nurse review of flagged conversations; versioned
  prompts; eval suite gating every release in CI.
- **Rollout**: pilot on thyroid (35 pts/mo) → measure deflection &
  cancellation rate → expand ENT → hospital-wide.
- **Equity v2**: reading-level simplification + English / Arabic / Turkish.

## Limitations

- Only the 19 PDFs in `hospital_documents/` are knowledge.
- Output guardrail uses an LLM judge — adds latency; could be replaced by
  a fine-tuned classifier in production.
