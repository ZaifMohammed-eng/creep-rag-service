# Deploying

The image serves the API on `$PORT` (default 8080) over the creep corpus. It builds the
index (2,326 documents) on the first request, about 2 s. `results/creep_config.json`
(tuned fusion weights and the gate) is baked in.

## Local

```bash
docker build -t creep-rag-service:0.1.0 .
docker run -p 8080:8080 creep-rag-service:0.1.0
curl localhost:8080/healthz
```

To answer with an LLM instead of the extractive baseline:

```bash
docker run -p 8080:8080 -e RAG_LLM=anthropic -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY creep-rag-service:0.1.0
```

## Google Cloud Run

```bash
gcloud config set project <PROJECT_ID>
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
gcloud run deploy creep-rag-service --source . --region us-central1 \
  --memory 2Gi --cpu 1 --min-instances 0 --max-instances 2 --allow-unauthenticated
```

For an LLM provider, store the key in Secret Manager and pass
`--set-secrets ANTHROPIC_API_KEY=anthropic-key:latest --set-env-vars RAG_LLM=anthropic`.
`--min-instances 0` keeps the cost at zero when idle; the first request after idle
pays the cold start plus index build.

## AWS App Runner (via ECR)

```bash
aws ecr create-repository --repository-name creep-rag-service
aws ecr get-login-password | docker login --username AWS --password-stdin <ACCOUNT>.dkr.ecr.<REGION>.amazonaws.com
docker buildx build --platform linux/amd64 -t <ACCOUNT>.dkr.ecr.<REGION>.amazonaws.com/creep-rag-service:0.1.0 --push .
# then create an App Runner service from that image, port 8080, 1 vCPU / 2 GB
```

Build for `linux/amd64` when building on an Apple Silicon Mac.

## Operating it

* `GET /metrics` exposes request outcomes (ok, abstained by reason, cache hits,
  errors), a latency histogram and LLM token counters for Prometheus.
* Each answered request is appended to `$RAG_REQUEST_LOG` (JSONL) with the gate
  probability, citations, timings and token usage. Watching the abstention rate and
  the gate-probability distribution over time is the first drift signal.
* Re-run `python eval/run_creep_eval.py` after any change to the corpus, embedder, filter or
  fusion settings; CI fails if retrieval, held-out declines or lookup accuracy drop below
  their floors.
