# Supply Chain Disruption Response Desk

CSE476 · Agentic AI and Intelligent Automation · Task 2 (Project 2) · Group 5

A multi-agent system that takes a supply-chain disruption (a port strike, a shortage, a typhoon) and proposes
**ranked action options, each with its cost, its delay and the reason it was ranked there**.

Five agents coordinate through a shared **blackboard** with a manager arbiter. They are
[Semantic Kernel](https://github.com/microsoft/semantic-kernel) `ChatCompletionAgent`s running
`openai/gpt-oss-120b` on [Groq](https://groq.com). Real tools do the arithmetic, so the numbers are exact; the LLMs
interpret text, decide which tools to call, and explain the ranking.

| # | Agent | Tool it calls |
|---|-------|---------------|
| 1 | Event Interpreter: free text to a validated event (refuses off-topic input) | none (validation in code) |
| 2 | Inventory: reads stock levels | `get_stock` |
| 3 | Supplier Alternatives: RAG over the supplier catalogue (TF-IDF cosine) | `search_catalogue` |
| 4 | Cost-Delay Trade-off: cost and delay per option | `evaluate_options` |
| 5 | Operations Manager: ranks every option and explains why | none |

## Quick start

Requires Python 3.10+ (tested on 3.14) and a free [Groq API key](https://console.groq.com).

```bash
pip install -r requirements.txt

# Windows (cmd)
set GROQ_API_KEY=your_key_here
# macOS / Linux
export GROQ_API_KEY=your_key_here

python server.py          # website at http://localhost:8000
```

Other ways to run it:

```bash
python main.py "Port strike at Shanghai port in China hits lithium cells for 21 days"   # one event, terminal
python evaluate.py                                                                      # 8 scenarios, terminal
streamlit run app.py                                                                    # older Streamlit UI
python tests/test_edge_cases.py                                                         # 65 edge-case tests, no key needed
```

The key is read from the environment only. It is never written to disk or sent to the browser.

## The website

Tabs: **Run desk** (live agent pipeline, ranked options with reasons), **Analysis** (cost breakdown, Pareto chart,
weight sensitivity), **Evaluation** (8 scenarios, LLM vs baseline), **Catalogue & stock**, **Blackboard** (the shared
state and write trail).

Backup modes if the network or Groq fails during a demo:
- **Replay saved LLM run**: shows a real earlier LLM run from `data/last_eval.json`, no internet needed.
- **Demo mode (no LLM)**: runs the same tools with a weighted ranking and no LLM. Always labelled on screen.

## Evaluation

8 disruption scenarios with a known best option; metric is top-1 match.

| Method | Top-1 match |
|--------|-------------|
| LLM Operations Manager | 8 / 8 |
| Weighted-score baseline (60% cost, 40% delay, no LLM) | 6 / 8 |

Caveats: the "known best" labels were set by the team (cheapest option with no stockout, or least stockout if none
avoids it), and all supplier and inventory data is synthetic. Results are indicative, not independent proof.

## Guardrails

Four gates sit between a user's text and the plan: a no-LLM pre-check (empty, too long, not about supply chains), an LLM
`is_disruption` classifier, a whitelist rebuild of every field, and tool guarantees with a deterministic fallback
ranking. Only the Event Interpreter ever sees the raw text. See `tests/test_edge_cases.py` (65 tests using a fake LLM).

## Storage

No database. Read-only JSON in `data/` (suppliers, inventory, scenarios); the only file the app writes is
`data/last_eval.json` after a fully successful evaluation. Each run's blackboard lives in memory and is discarded.

## Project layout

```
server.py            web server (Starlette + uvicorn): API, validation, queue, demo/replay modes
web/index.html       the whole front end (HTML, CSS, JS, Chart.js from a CDN)
pipeline.py          runs the five agents in order on a fresh blackboard
agents/              one module per agent, plus tools.py, blackboard.py, llm.py, errors.py
data/                suppliers.json, inventory.json, scenarios.json, last_eval.json
tests/               edge-case tests
app.py, main.py, evaluate.py   Streamlit UI, CLI run, CLI evaluation
docs/                presentation (.pptx) and the presenter's guide (.pdf)
```

## Team

Induri Avinash Reddy (12314362) · Tummuru Varshith Krishna Mohan Reddy (12305637)
