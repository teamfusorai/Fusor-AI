# Project structure and naming

## Current layout

```
Fusor AI/
├── main.py                 # FastAPI app, health/ready, KB list/delete, WS chat, QR, root
├── config.py               # Central config (env-based)
├── data_ingestion.py       # Ingest router + sync/async ingestion logic
├── search_engine.py        # Query router + RAG search + answer
├── chatbot_config.py       # Bubble.io config fetch + system prompt builder
├── celery_app.py           # Celery app (broker/backend)
├── test.py                 # Legacy manual query script (see scripts/)
├── requirements.txt
├── .env
├── tasks/
│   ├── __init__.py
│   └── ingest_tasks.py     # Celery task: run_ingest
├── utils/
│   ├── logging_config.py
│   ├── metrics.py
│   └── sitemap.py
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_chatbot_config.py
│   └── test_search_engine.py
└── scripts/
    ├── README.md
    └── query_knowledge_base.py   # Manual query script
```

## What works well

- **config.py** – Single place for env and tuning.
- **utils/** – Shared helpers (logging, metrics, sitemap) are grouped.
- **tests/** – Pytest layout is standard.
- **tasks/** – Celery tasks are isolated and easy to run with a worker.

## Suggested improvements

### 1. File names

- **data_ingestion.py** → **ingestion.py**  
  Shorter and still clear; “ingestion” is the domain.  
  If you rename, update imports in `main.py` and `tasks/ingest_tasks.py`.

- **search_engine.py** → Keep as is, or **retrieval.py** / **rag.py** if you want to stress “RAG” rather than “search”.

- **chatbot_config.py** → Keep, or **chatbot_config_service.py** if you want to stress it’s a service layer.

### 2. Root clutter

- **test.py** – Prefer moving to **scripts/query_knowledge_base.py** (done) and deleting root **test.py** so the root stays minimal.
- **celery_app.py** – Could move to **core/celery_app.py** or **app/celery.py** if you later introduce an `app/` or `core/` package.

### 3. Optional: group API routes

If the codebase grows, you can group route modules under a package, e.g.:

- **api/** or **routes/**
  - **ingestion.py** (router only; logic stays in a service or current module).
  - **search.py**
  - **chatbot_config.py** (or keep at root if it’s small).

Then **main.py** would do `from api.ingestion import router` etc. No need to do this until you have more routes or multiple teams.

### 4. Docs and run instructions

- **README.md** – Add a short “Project structure” section that points to this file.
- **CELERY.md** (optional) – How to set `CELERY_BROKER_URL`, run the worker, and poll `GET /ingest/status/{job_id}`.

## Summary

- Current structure is clear and fine for this size.
- Small wins: move/remove root **test.py** (use **scripts/query_knowledge_base.py**), optionally rename **data_ingestion.py** → **ingestion.py**.
- Larger refactors (e.g. **api/** or **core/**) can wait until the project grows.
