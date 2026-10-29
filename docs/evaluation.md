# Evaluation — Developer Documentation Assistant

## Method and scope

Five fixed FastAPI questions (`q1` dependency injection, `q2` OAuth2, `q3` file uploads, `q4` sync vs async endpoints, `q5` Pydantic request validation). RAGAS 0.4.3 metrics use `gpt-4.1-mini` as evaluator, with `text-embedding-3-small` for answer relevancy. Faithfulness, relevancy and precision were computed on **previously saved answers and contexts**; recall was subsequently computed against authored reference answers. These are model-judged scores, not objective correctness percentages. Reference answers should be reviewed against the official documentation before final submission.

## RAGAS results

| Question | Answer relevancy | Context precision | Context recall |
|---|---:|---:|---:|
| q1 | 0.9703 | 1.0000 | 1.0000 |
| q2 | 0.8917 | 1.0000 | 1.0000 |
| q3 | 0.9045 | 0.9500 | 1.0000 |
| q4 | 0.9533 | 1.0000 | 1.0000 |
| q5 | 0.8353 | 1.0000 | 0.5000 |
| **Mean (5/5)** | **0.9110** | **0.9900** | **0.9000** |

**Mean faithfulness: 0.8681 (5/5).** Individual faithfulness scores are in `data/evaluation_results.json`; this document does not infer their exact values from the aggregate. The lower q5 context recall suggests some information in the reference answer was not supported by retrieved evidence. The small benchmark and LLM-as-judge variability limit generalization.

## Local latency benchmark — RAG generation

An earlier five-query benchmark measured total retrieval-plus-generation latency of **3.34 s on average** (all five requests succeeded through OpenAI fallback). A subsequent controlled streaming provider comparison used the same retrieval context for both providers and measured:

| Question | OpenAI total (s) | OpenAI TTFT (s) |
|---|---:|---:|
| q1 | 3.285 | 1.004 |
| q2 | 3.854 | 0.844 |
| q3 | 2.972 | 0.764 |
| q4 | 2.629 | 0.525 |
| q5 | 2.704 | 0.555 |
| **Mean** | **3.089** | **0.738** |

`data/detailed_benchmark.json` contains **per-query `retrieval_seconds`, `llm_seconds`, and `total_seconds`** for the controlled comparison; include that file in the source archive so the grader can inspect the breakdown. Note that its total is the sum of the measured retrieval and LLM durations; it excludes LangGraph planning, browser/network overhead, and other production workflow steps. TTFT is measured from the start of the model call, whereas the Cloud Run TTFT below is measured from the client request start.

## Cloud Run latency — SSE endpoint

| Request | TTFT (s) | Total (s) | Completed |
|---|---:|---:|---|
| First request after idle (cold **not confirmed**) | 10.967 | 13.191 | Yes |
| Immediate subsequent request (warm) | 1.893 | 4.251 | Yes |

The first request was much slower, consistent with cold-start overhead, but instance creation was not independently confirmed. Label it *first request after idle*, not a proven cold start. The warm measurement is only one sample, not a statistically stable latency distribution.

## Provider comparison and availability

| Provider | Model | Controlled benchmark | Interpretation |
|---|---|---|---|
| OpenAI | `gpt-4.1-mini` | 5/5 successful; mean total 3.089 s; mean TTFT 0.738 s | Usable observed baseline |
| Google | `gemini-3.6-flash` | 0/5 successful in the controlled run (quota errors) | No comparable successful latency in this run |

Gemini returned `429 RESOURCE_EXHAUSTED` due to the observed free-tier limit of 20 requests/day for this model. Its short error-return times are **not** successful generation latency. Separate earlier Gemini streaming smoke test: 0.96 s TTFT and 1.92 s total for one short answer, under different conditions; not an apples-to-apples comparison. The system falls back to OpenAI when Gemini returns 429/503 before emitting text. Both providers have been successfully used during development, but a controlled same-prompt quality comparison remains incomplete.

## Token usage and estimated OpenAI cost

The following estimates use **illustrative, unverified rates** of $0.40 per million input tokens and $1.60 per million output tokens. Verify current prices for the exact model before presenting these as billed costs.

| Question | Input tokens | Output tokens | Estimated USD |
|---|---:|---:|---:|
| q1 | 774 | 189 | 0.000612 |
| q2 | 787 | 500 | 0.001115 |
| q3 | 805 | 383 | 0.000935 |
| q4 | 765 | 299 | 0.000784 |
| q5 | 859 | 367 | 0.000931 |
| **Total** | **3,990** | **1,738** | **0.004377** |

Average estimated cost per successful controlled OpenAI generation: **$0.000875**. These costs **exclude** query embeddings, planner calls, evaluator calls, Tavily, cloud infrastructure, and possible provider retries. They are not actual billed cost per complete end-to-end user turn.

## Verification and limitations

- Local tests verified supervisor delegation, Research→Code and Research→Summary result preservation, worker failure stop, provider 429/503 fallback, and shared Tavily MCP invocation by Research and Code.
- Production traces confirmed Research and Code each invoked `tavily_search`; SSE streaming and provider-routing display worked in Cloud Run.
- Citation validation checks chunk IDs against retrieved chunks, not claim-level factual correctness.
- A small fixed dataset cannot support strong accuracy or latency claims. Add more questions, repeated warm/cold trials, and successful same-prompt comparisons across both providers for a more rigorous evaluation.

**Source artifacts to retain in submission:** `data/evaluation_questions.json`, `data/evaluation_results.json`, `data/latency_results.json`, `data/detailed_benchmark.json`, `data/cloud_run_latency.json` (where present).
