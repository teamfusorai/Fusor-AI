# Background ingestion with Celery

When `CELERY_BROKER_URL` is set, the `/ingest` endpoint queues work to a Celery worker instead of running it in the API process.

## 1. Redis (broker and optional result backend)

Install and start Redis, e.g.:

- **Docker:** `docker run -d -p 6379:6379 redis`
- **Windows:** use WSL or a Redis build for Windows

## 2. Environment

In `.env`:

```env
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
```

- **CELERY_BROKER_URL** – Required for background jobs. If unset, ingestion runs inline in the API.
- **CELERY_RESULT_BACKEND** – Optional. Needed to poll job status with `GET /ingest/status/{job_id}`.

Optional:

```env
INGEST_UPLOAD_DIR=.ingest_uploads
```

The worker must be able to read this directory (same machine or shared volume).

## 3. Run the worker

From the project root (with venv activated):

```bash
celery -A celery_app worker -l info
```

On Windows you may need:

```bash
celery -A celery_app worker -l info -P solo
```

## 4. API behaviour

- **POST /ingest** (file or URL)  
  - With Celery: returns `{"status": "queued", "job_id": "<id>", ...}`.  
  - Without Celery: returns the same as before (ingestion runs in the request).

- **GET /ingest/status/{job_id}**  
  - Returns `{"job_id": "...", "status": "PENDING"|"STARTED"|"SUCCESS"|"FAILURE", "result": {...} }` when `CELERY_RESULT_BACKEND` is set.  
  - Without result backend, status may stay PENDING and result won’t be available.

## 5. File uploads

For file uploads, the API writes the file under `INGEST_UPLOAD_DIR` and passes the path to the worker. The worker deletes the file after processing. Ensure the worker process can read that directory (e.g. same host or shared filesystem).
