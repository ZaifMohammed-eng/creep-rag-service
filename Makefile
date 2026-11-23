.PHONY: test eval squad-data squad-eval gen-eval bench serve docker

test:
	pytest -q
eval:                ## creep corpus: retrieval, metadata filter, gate, answers
	python eval/run_creep_eval.py
squad-data:
	python scripts/get_data.py
squad-eval: squad-data   ## generic benchmark on SQuAD v1.1 dev
	python eval/run_squad_eval.py
gen-eval: squad-data     ## PROVIDER: extractive | anthropic | openai
	python eval/run_generation_eval.py --provider $${PROVIDER:-extractive} --n 300
bench:
	python eval/bench_api.py --requests 1000 --concurrency 16
serve:
	uvicorn rag.api:app --host 0.0.0.0 --port 8000
docker:
	docker build -t creep-rag-service:0.1.0 .
