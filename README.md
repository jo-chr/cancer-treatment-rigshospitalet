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
  cli.py                    # Interactive CLI
eval/
  questions.yaml            # 12 hand-crafted test cases
  run.py                    # Scorecard: intent / refusal / citation accuracy
```

## Architecture

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
