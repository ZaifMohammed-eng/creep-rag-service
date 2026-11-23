FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RAG_LLM=extractive RAG_CORPUS=creep PORT=8080
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY rag/ rag/
COPY data/creep_tests.csv data/creep_tests.csv
COPY data/notes/ data/notes/
COPY results/creep_config.json results/creep_config.json
RUN useradd -m app && chown -R app /app
USER app
# the index (2,326 documents) is built on the first request, ~2 s
CMD ["sh", "-c", "uvicorn rag.api:app --host 0.0.0.0 --port ${PORT}"]
