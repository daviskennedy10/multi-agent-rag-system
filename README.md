# Developer Documentation Assistant

Multi-agent FastAPI application for grounded technical documentation research, Python code analysis, and document summaries. LangGraph orchestrates Research, Code and Summary workers; Chroma provides FastAPI documentation retrieval; Tavily Remote MCP provides current web search. OpenAI and Gemini generation are selected through an inspectable heuristic router, with fallback on certain provider errors. The frontend is a single vanilla HTML/CSS/JavaScript page receiving SSE token streams.

**Live demo:** https://developer-docs-assistant-let7krnqfa-wl.a.run.app

## Architecture and evaluation

- [`docs/architecture.md`](docs/architecture.md): graph, state, agent tools, routing, RAG, persistence and limitations.
- [`docs/evaluation.md`](docs/evaluation.md): RAGAS scores, latency, TTFT, provider comparison and cost estimates.
- [`PROCESS_LOG.md`](PROCESS_LOG.md): development and verification narrative.

## Requirements

Python 3.12 (tested locally), Docker for containerization, Google Cloud CLI for deployment, and credentials for OpenAI, Google Gemini and Tavily. Install **exact pinned versions** from the project's existing `requirements.txt`. Do not replace that file with this documentation bundle.

## Local setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
# Edit .env locally with real keys; never commit it.
python -m uvicorn main:app --reload --port 8000
```

Open `http://127.0.0.1:8000/`; check `http://127.0.0.1:8000/health`. The packaged Chroma index is expected at `./chroma_index` (override with `CHROMA_PATH`). Indexing a fresh corpus requires `data/raw` and the project's ingestion scripts; it incurs embedding API usage. Do not run the destructive index-reset command on the submission index.

### `.env.example` values

```dotenv
OPENAI_API_KEY=replace-me
OPENAI_MODEL=gpt-4.1-mini
GOOGLE_API_KEY=replace-me
GEMINI_MODEL=gemini-3.6-flash
TAVILY_API_KEY=replace-me
CHROMA_PATH=./chroma_index
```

Keep the real `.env` out of Git and the source ZIP. Provider access, free quotas and pricing can change; Gemini 3.6 Flash encountered 429 daily-quota errors during testing, so OpenAI fallback is important. API costs are separate from Google Cloud infrastructure charges.

## Local container

```bash
docker build -t developer-docs-assistant:local .
docker run --rm --env-file .env -p 8081:8080 developer-docs-assistant:local
curl http://127.0.0.1:8081/health
```

## Cloud Run deployment

Example project: `csci599-docs-assistant-2026`; region: `us-west2`. For a fresh project, create it, link billing, and enable Cloud Run, Cloud Build, Artifact Registry and Secret Manager APIs. Create the Artifact Registry Docker repository `docs-assistant`, configure Docker auth, and provision a dedicated service account with Secret Manager accessor rights for the three specific secrets. Create secrets `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `TAVILY_API_KEY` with populated versions. **Do not upload `.env` into the image.**

```bash
gcloud config set project YOUR_PROJECT_ID
gcloud config set run/region us-west2
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com
gcloud auth configure-docker us-west2-docker.pkg.dev
IMAGE="us-west2-docker.pkg.dev/YOUR_PROJECT_ID/docs-assistant/app:v1"
docker buildx build --platform linux/amd64 --tag "$IMAGE" --push .
gcloud run deploy developer-docs-assistant \
  --image="$IMAGE" --region=us-west2 \
  --service-account="YOUR_SERVICE_ACCOUNT_EMAIL" \
  --set-secrets=OPENAI_API_KEY=OPENAI_API_KEY:latest,GOOGLE_API_KEY=GOOGLE_API_KEY:latest,TAVILY_API_KEY=TAVILY_API_KEY:latest \
  --set-env-vars=OPENAI_MODEL=gpt-4.1-mini,GEMINI_MODEL=gemini-3.6-flash,CHROMA_PATH=/app/chroma_index \
  --memory=2Gi --cpu=2 --concurrency=1 --max-instances=2 --timeout=300 \
  --allow-unauthenticated
```

This recipe assumes you have separately created the repository, service account and secrets and granted permissions. It is **not** a one-command bootstrap. The service is public for coursework demonstration; add authentication and rate limiting before production use. A read-only Chroma snapshot is bundled into the container image; user conversation state is intentionally not persisted between requests.

## Testing

```bash
python -m tests.test_provider_fallback
python -m tests.test_multi_agent
python -m app.rag.evaluate_additional
python -m app.rag.evaluate_recall
python -m tests.benchmark_latency
```

Some tests require external API access and consume quota. Check the deployed dashboard for actual streamed tokens, agent transitions, provider/model/rationale and citation links. Confirm live Tavily usage separately with a current FastAPI release question and a latest-API code analysis request. See `docs/evaluation.md` for observed results and limitations.

## Submission

Include source, pinned `requirements.txt`, `.env.example`, docs, process log, evaluation data and a reproducible index or documented reindexing procedure. Exclude `.env`, API keys, `.venv`, caches, and Git history. Supply the live Cloud Run URL separately in Brightspace.
