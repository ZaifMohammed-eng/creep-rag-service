# Creep RAG service: retrieval-augmented answers over superalloy creep records

A retrieval-augmented generation (RAG) service that answers questions about creep
rupture of nickel-based superalloys from three kinds of documents, the way an engineering
team's knowledge base is organised: **test records, alloy data sheets and study notes**.
It finds the passages, answers only from them with a citation, and declines questions
about alloys it has no documents for.

It is the retrieval counterpart to [creep-alloy-agent](https://github.com/ZaifMohammed-eng/creep-alloy-agent),
which answers the same domain through SQL and model tools. The two cover different
questions: this service answers lookups ("how long did Waspaloy last at 552 MPa and
760 °C?"); the agent answers aggregations ("which alloy lasted longest above 1200 K?"),
which retrieval cannot.

```
question ─► BM25 (lexical) ──────┐
         └► embedding ─► FAISS / Qdrant ─┴► reciprocal-rank fusion (top 50)
                                               │
                          metadata filter: parse alloy, MPa, K or °C from the question,
                          re-rank that alloy's test records by distance to those conditions
                                               │
            scope check: names an indexed alloy? → in scope; otherwise learned abstention gate
                                               │ declines
                                               ▼
                          LLM (Anthropic / OpenAI) or offline extractive reader
                                               │
                    citation check: the answer must cite a retrieved passage id
                                               ▼
                FastAPI /v1/query → answer, citations, gate probability, timings, tokens, cost
```

## The corpus

Built from the dataset behind *Interpretable ML for Creep Rupture Life in Ni-Based
Superalloys* ([calphad-ml-creep](https://github.com/ZaifMohammed-eng/calphad-ml-creep)):
2,632 creep rupture tests on 38 alloys, with CALPHAD descriptors computed with Thermo-Calc.

| Document | Count | Content |
|---|---|---|
| Test record | one per test | alloy, temperature (K and °C), stress, time to rupture, Larson-Miller parameter, CALPHAD γ′ fraction, lattice misfit and γ-matrix diffusivity at that temperature |
| Alloy data sheet | one per alloy | composition (atom %), tested temperature and stress ranges, number of tests, rupture-life range, γ′ fraction range |
| Study notes | 7 chunks | the method and findings paragraphs from the two creep-study READMEs |

Records and data sheets are **rendered from the dataset**: the facts are the dataset's,
the sentences are generated (`rag/creep.py`). That is what makes the evaluation possible:
every question has a known gold document and a known answer.

**Evaluation design.** 6 of the 38 alloys are held out of the index entirely; questions
about them must be declined. The other 32 alloys are split by alloy into calibration and
test halves. Fusion weights and the abstention gate are fit on calibration alloys; every
number below is from test alloys (935 answerable questions) and held-out alloys (148).

Question types (all generated from the data, several phrasings each):

| Type | Example | Gold |
|---|---|---|
| exact lookup | How long did Waspaloy last at 552 MPa and 1033.15 K? | every record at that alloy, stress and temperature |
| approximate lookup | Roughly how long does Waspaloy last at about 550 MPa and 760 °C? | every record within 5 MPa and 5 °C |
| composition | How much chromium does Udimet 500 contain? | the alloy's data sheet |
| temperature range | Over what temperature range was Inconel 617 creep tested? | the data sheet |
| CALPHAD descriptor | What γ′ fraction does CALPHAD give for Waspaloy at 1033.15 K? | records at that temperature |
| method (8, hand-written) | By how much were features perturbed in the sensitivity analysis? | the note chunks that say so |
| aggregation (8) | Which alloy lasted longest in tests above 1200 K? | no single passage: out of reach for retrieval |

## Results (test alloys)

Retrieval: gold document ranked first (R@1), in the top 5 (R@5):

| Retriever | All, R@1 | All, R@5 | Exact lookup, R@1 | **Approximate lookup, R@1** |
|---|---|---|---|---|
| BM25 | 0.723 | 0.798 | 0.961 | 0.161 |
| Dense (WordLlama embeddings) | 0.156 | 0.268 | 0.027 | 0.021 |
| Hybrid, standard RRF (k = 60, equal weights) | 0.314 | 0.602 | 0.236 | 0.061 |
| Hybrid, tuned on calibration alloys (dense weight → 0) | 0.723 | 0.798 | 0.961 | 0.161 |
| **Tuned hybrid + metadata filter** | **0.996** | **0.999** | **1.000** | **1.000** |

What this says:

* **Text retrieval cannot match numbers.** To BM25, "550 MPa" and "552 MPa" are different
  tokens, so it finds the right record for only 16% of approximate questions. The small
  offline embedding model is worse still on records that differ only in their numbers;
  tuning drove its fusion weight to zero.
* **Parsing the question fixes it.** The metadata filter reads the alloy name (matched
  against the index's own catalogue), the stress and the temperature (K or °C, never
  confused with names like "Inconel 713C"), then re-ranks that alloy's records by distance.
  The approximate-lookup gold set is defined by the same nearness, so the 100% there mainly
  shows that the parser recovered the alloy and conditions every time.
* **Standard equal-weight fusion hurt** (R@5 0.798 → 0.602) because the dense list is noise
  for these documents. Fusion has to be tuned per corpus.

Scope and abstention:

| | Result |
|---|---|
| Answerable questions answered | 100% (935) |
| Questions about held-out alloys declined | **89.9%** (133 of 148) |
| Learned gate alone, AUROC answerable vs held-out | 0.964 |
| Held-out questions wrongly matched to an indexed alloy name | 0 |

A question naming an alloy that is in the catalogue skips the learned gate; everything
else goes through it. The gate's largest weight is on whether the top document's alloy
name appears in the question.

Answers with the offline extractive reader (returns the best-matching sentence, so these
measure whether an answer-bearing sentence is retrieved and selected):

| Type | n | Correct |
|---|---|---|
| Exact lookup | 440 | 100% |
| Approximate lookup | 280 | 96.4% |
| Composition | 119 | 100% |
| CALPHAD descriptor | 76 | 100% |
| Temperature range | 16 | 93.8% |
| Method notes | 4 | 2 of 4 |
| **Aggregation** | 8 | answered 5, **correct 0** |

The aggregation row is the important negative result: retrieval returns five plausible
records and the reader answers from them, confidently and wrongly. Those questions need
a scan over all 2,632 records, which is what the agent's SQL tool is for. Routing
aggregation questions away from RAG is the next step.

**Conflicting sources.** For "Does a higher γ′ fraction increase or reduce creep life?"
all three relevant note chunks are retrieved in the top 5, and they disagree: two say
higher γ′ improves creep resistance, one says high Xγ′ reduces creep life. The prompt asks
the LLM to cite passages; a reader of the citations sees the conflict.

Service latency (`eval/bench_api.py`, uvicorn, 1 worker, 2 vCPU, no LLM call):
5.4 ms p50 / 7.9 ms p95 sequential, 127 requests/s at 16 concurrent, 0 errors over 1,000
requests. Retrieval, filter and gate take 3.3 ms in process.

## Data issues found while building the corpus

* **215 of 2,632 tests have rupture times above 200,000 hours** (about 23 years); 170 of
  them are Inconel 617, and five Inconel 617 tests at 850 K show more than 10¹⁵ hours. These
  are not physical test durations and look like extrapolated or mis-converted values. They
  are kept as-is so the corpus matches the published dataset, and flagged here.
* **The two study READMEs disagree on the effect of γ′ fraction** (see above), and the
  article README contradicts itself between consecutive paragraphs.

## Generic benchmark: SQuAD v1.1

The same engine, tuned the same way, on 10,570 Wikipedia questions with 8 of 48 articles
held out (`make squad-eval`): tuned hybrid Recall@5 0.919, MRR 0.832; abstention-gate
AUROC 0.965, declining 85.0% of out-of-scope questions while answering 93.9% of in-scope
ones. FAISS HNSW returns 99.4% of exact top-10 neighbours.

## Production features

| Piece | File |
|---|---|
| Hybrid retrieval (BM25 + embeddings, weighted RRF) | `rag/retriever.py` |
| Embeddings: WordLlama (offline, weights in the pip package), spaCy, sentence-transformers, OpenAI | `rag/embeddings.py` |
| Vector stores: FAISS flat / HNSW, Qdrant (local or server) | `rag/vectorstore.py` |
| Metadata filter: entity + unit parsing, numeric re-ranking | `rag/metadata.py` |
| Abstention gate, saved as JSON | `rag/gate.py` |
| LLM providers: Anthropic, OpenAI, offline extractive; retries, token and cost accounting | `rag/llm.py` |
| Grounded prompt and citation check | `rag/pipeline.py` |
| FastAPI app: validation, answer cache, document ingestion, JSONL log, Prometheus metrics | `rag/api.py` |
| Corpus builder and question generator | `rag/creep.py` |
| Evaluation, load test, CI regression gates | `eval/`, `.github/workflows/ci.yml` |

## Run it

```bash
pip install -e ".[dev]"
pytest -q                          # 23 tests
make eval                          # creep evaluation, ~20 s
make serve                         # http://localhost:8000/docs  (RAG_CORPUS=creep by default)
curl -s localhost:8000/v1/query -H 'content-type: application/json' \
     -d '{"query": "Roughly how long does Waspaloy last at about 550 MPa and 760 °C?"}'

# answer with an LLM instead of the extractive reader
export RAG_LLM=anthropic ANTHROPIC_API_KEY=...         # or RAG_LLM=openai OPENAI_API_KEY=...
export RAG_PRICE_IN_PER_MTOK=... RAG_PRICE_OUT_PER_MTOK=...   # optional, cost per request
make serve
```

Docker and cloud deployment (Google Cloud Run, AWS App Runner): [DEPLOY.md](DEPLOY.md).

## Limitations

* Questions are generated from templates, so they share wording with the documents. Real
  engineers phrase things more loosely; the approximate-lookup set is a first step toward
  that, and a hand-written question set from practising engineers is the real test.
* Only 8 method questions and 7 note chunks: the prose side of the corpus is small.
  Adding papers and alloy datasheets (Special Metals, Haynes) is the obvious extension.
* LLM answer quality was not measured here (no API key in the build environment); the
  provider code is tested against mocked APIs.
* Aggregation questions are answered wrongly rather than declined; they need routing to SQL.
