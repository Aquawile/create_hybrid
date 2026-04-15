# AD-RAG: Entropy-Gated Diagnostic Retrieval-Augmented Generation

An autonomous UAV diagnostic system that uses **Shannon entropy as an abstention gate** to prevent hallucinated diagnoses. The system compares two retrieval strategies using a **real local LLM** (Ollama), with **no ground truth leakage during inference**.

## Overview

AD-RAG evaluates diagnostic queries against a knowledge corpus to determine the root cause of UAV failures. Unlike standard RAG systems that always output an answer, this system **abstains when its internal belief distribution is too uncertain** (high Shannon entropy), preventing confident-but-wrong diagnoses.

### Key Differences From Standard Approaches

- **Real LLM generation** — Uses Ollama (local, free) instead of manufactured belief functions
- **No ground truth leakage** — The model diagnoses from retrieved documents alone; accuracy is measured post-hoc
- **TF-IDF retrieval** — Both baseline and hybrid use proper TF-IDF cosine similarity, not hardcoded document selection
- **Structured output** — The model must return JSON with diagnosis, confidence, evidence citations, and alternatives considered
- **Entropy-gated action** — Shannon entropy of the belief distribution determines whether the system ACTs or ABSTAINs

## Project Structure

```
create_hybrid/
├── main.py                      # CLI evaluation script (runs all 25 queries)
├── dashboard/
│   └── app.py                   # Streamlit interactive dashboard
├── data/
│   ├── corpus.json             # Knowledge base (25 diagnostic documents)
│   ├── queries.json             # 25 diagnostic test queries
│   ├── hypotheses.json          # Candidate hypotheses per query
│   └── ground_truth.json        # Correct diagnoses (eval-only, never passed to model)
├── engine/
│   ├── retriever.py             # TF-IDF + keyword overlap retrieval
│   ├── llm_interface.py         # Ollama-based real LLM with structured JSON output
│   └── safety_gate.py           # Shannon entropy abstention mechanism
└── evaluation/
    └── metrics.py               # Performance comparison and reporting
```

## Components

### Retriever ([`engine/retriever.py`](engine/retriever.py))

The `DiagnosticRetriever` class implements two retrieval modes:

- **Baseline** (`hybrid=False`): Pure TF-IDF cosine similarity ranking. Documents are scored by term frequency-inverse document frequency vectors and ranked by cosine distance to the query.
- **Hybrid** (`hybrid=True`): TF-IDF score plus a keyword overlap bonus. Documents sharing exact terms with the query receive a small boost, simulating BM25-style lexical matching layered on top of TF-IDF.

Both modes return the top-k documents by combined score.

### LLM Interface ([`engine/llm_interface.py`](engine/llm_interface.py))

The `DiagnosticLLM` class calls a local Ollama instance to generate diagnostic beliefs:

- **No ground truth is passed** — the model receives only the query, retrieved documents, and candidate hypotheses
- **Constrained prompt** — the model must respond with structured JSON including diagnosis, confidence, evidence, and alternatives
- **Structured output parsing** — handles JSON in code fences, bare JSON, or partial output
- **Abstention handling** — if the model returns no diagnosis, low confidence, or invalid JSON, the system returns a uniform (maximum entropy) distribution
- **Deterministic** — temperature is set to 0.0 for reproducibility

### Safety Gate ([`engine/safety_gate.py`](engine/safety_gate.py))

The `analyze_safety()` function implements the entropy-based abstention mechanism:

- Computes Shannon entropy: `H = -Σ p(x) · log₂ p(x)` over the belief distribution
- If entropy exceeds the configurable threshold (default: 2.0 bits), the system **ABSTAINs**
- Otherwise returns the highest-confidence hypothesis for action

A lower threshold makes the system more cautious. The default (2.0 bits) allows moderate uncertainty while catching genuinely confused states.

### Metrics ([`evaluation/metrics.py`](evaluation/metrics.py))

The `print_comparison_row()` function displays side-by-side results:

- Query text and ground truth
- Baseline vs Hybrid predictions with confidence scores and entropy values
- Decision status (ACT or ABSTAIN)
- Outcome assessment (whether hybrid corrected, abstained, or still missed)

## Quick Start

### Prerequisites

1. **Python 3.8+** with NumPy
2. **Ollama** installed and running locally:

```bash
# macOS
brew install ollama
ollama serve &
ollama pull llama3  # or any other model
```

3. **Streamlit** (for the dashboard):

```bash
pip install streamlit numpy requests
```

### Run CLI Evaluation

```bash
python main.py
```

This runs all 25 diagnostic queries through both retrieval methods and prints a detailed comparison with final statistics.

### Run Interactive Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard supports three modes:
- **Single Query** — run one scenario at a time, see full reasoning and retrieved documents
- **Batch Run** — execute all 25 queries with a progress bar and results table
- **Custom Query** — type your own symptom description and candidate diagnoses

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SAFETY_THRESHOLD` | `2.0` | Max entropy (bits) before abstention |
| `OLLAMA_MODEL` | `llama3` | Ollama model name to use |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API endpoint |
| `top_k` | `2` | Number of documents to retrieve |

## How It Works (Pipeline)

```
Query
  ↓
Retrieve Documents (TF-IDF or TF-IDF + keywords)
  ↓
Prompt LLM: "Based on these docs, which diagnosis fits?"
  ↓
LLM returns structured JSON: {diagnosis, confidence, evidence, alternatives}
  ↓
Build belief distribution from LLM output
  ↓
Compute Shannon entropy of belief distribution
  ↓
Entropy > threshold? → ABSTAIN (escalate to human)
Entropy ≤ threshold? → ACT (execute diagnosis)
```

## What Makes This Different

| Aspect | Standard RAG | AD-RAG |
|--------|-------------|--------|
| Generation | Always outputs an answer | Abstains when uncertain |
| Retrieval | Often naive | TF-IDF with proper ranking |
| Ground truth | Sometimes leaked into prompt | Never passed to model at inference |
| Output format | Free text | Structured JSON with citations |
| Uncertainty | Hidden or post-hoc | Shannon entropy as explicit gate |
| Model | Closed API | Local, free (Ollama) |

## Evaluation Metrics

The system reports:
- **Accuracy**: % of queries where the ACT decision matches ground truth
- **Abstention rate**: % of queries where the system declined to decide
- **Wrong rate**: % of queries where the system acted but was incorrect
- **Improvement**: Accuracy difference between hybrid and baseline

A well-tuned system should show that hybrid retrieval produces higher accuracy and/or more appropriate abstentions than baseline.

## Technical Domains Covered

Power systems, sensor failures, wiring issues, avionics, communications, airframe structure, propulsion, and safety mechanisms across 25 UAV diagnostic scenarios.

## Dependencies

- Python 3.8+
- NumPy
- Requests (for Ollama API calls)
- Streamlit (for dashboard)

No paid API keys required. Everything runs locally.
