<<<<<<< HEAD
# create_hybrid
=======
# AD-RAG: Augmented Diagnostic Retrieval-Augmented Generation

A hybrid RAG system for terminal diagnostic decision-making in autonomous drone systems. This project demonstrates the difference between baseline and hybrid retrieval approaches for diagnosing hardware and software failures in UAV (Unmanned Aerial Vehicle) systems.

## Overview

AD-RAG evaluates diagnostic queries against a knowledge corpus to determine the root cause of hardware failures. The system compares two retrieval strategies:

- **Baseline**: Uses fixed, generic document retrieval (first 2 documents)
- **Hybrid**: Uses keyword-based semantic retrieval to find the most relevant documents

## Project Structure

```
create_hybrid/
├── main.py                      # Main evaluation script
├── data/
│   ├── corpus.json             # Knowledge base (25 diagnostic documents)
│   ├── queries.json             # 25 diagnostic test queries
│   ├── hypotheses.json          # Candidate hypotheses for each query
│   └── ground_truth.json        # Correct diagnoses for evaluation
├── engine/
│   ├── retriever.py             # Document retrieval (baseline + hybrid)
│   ├── llm_interface.py         # Belief generation from retrieved docs
│   └── safety_gate.py           # Entropy-based safety thresholding
└── evaluation/
    └── metrics.py               # Performance comparison and reporting
```

## Components

### Retriever ([`engine/retriever.py`](engine/retriever.py:3))

The [`DiagnosticRetriever`](engine/retriever.py:3) class handles document retrieval:

- **Baseline mode** (`hybrid=False`): Returns the first `top_k` documents from the corpus
- **Hybrid mode** (`hybrid=True`): Performs keyword matching to find documents containing query terms, returning the most relevant matches

### LLM Interface ([`engine/llm_interface.py`](engine/llm_interface.py:3))

The [`DiagnosticLLM`](engine/llm_interface.py:3) class generates belief probabilities:

- Initializes beliefs with tiny weights
- Boosts belief scores when retrieved documents contain ground truth keywords
- Converts belief weights to probability distributions (0.0 to 1.0)

### Safety Gate ([`engine/safety_gate.py`](engine/safety_gate.py:3))

The [`analyze_safety()`](engine/safety_gate.py:3) function implements an abstention mechanism:

- Calculates Shannon entropy from belief distributions
- If entropy exceeds the threshold, the system **ABSTAINS** from making a decision
- Otherwise, returns the highest-confidence hypothesis as the decision
- A higher threshold (2.0) makes the AI more "confident" and willing to act

### Metrics ([`evaluation/metrics.py`](evaluation/metrics.py:1))

The [`print_comparison_row()`](evaluation/metrics.py:1) function displays:

- Query description and ground truth
- Baseline vs Hybrid predictions with confidence scores
- Decision status (ACT or ABSTAIN)
- Outcome summary (whether hybrid saved the mission or prevented wrong action)

## Running the Evaluation

Execute the main script:

```bash
python main.py
```

The script will:
1. Load the corpus, queries, hypotheses, and ground truth
2. Run both baseline and hybrid retrieval for each query
3. Generate beliefs and apply safety gating
4. Print a detailed comparison for each diagnostic case
5. Display final performance summary

## Configuration

Adjust the safety threshold in [`main.py`](main.py:16):

```python
SAFETY_THRESHOLD = 2.0  # Lower = more cautious (more abstentions)
```

## Data Format

### Corpus ([`data/corpus.json`](data/corpus.json:1))
Each document contains:
- `id`: Unique identifier (D1-D25)
- `text`: Diagnostic knowledge text
- `metadata.domain`: Technical domain (power, sensors, wiring, avionics, etc.)

### Queries ([`data/queries.json`](data/queries.json:1))
Diagnostic scenarios describing symptoms (e.g., "Motor RPM dropped instantly during a full throttle climb.")

### Hypotheses ([`data/hypotheses.json`](data/hypotheses.json:1))
3 candidate diagnoses per query (e.g., ["voltage_sag", "propeller_loose", "esc_thermal"])

### Ground Truth ([`data/ground_truth.json`](data/ground_truth.json:1))
Correct diagnosis for each query

## Example Output

```
======================================================================
ANDURIL AD-RAG: TERMINAL DIAGNOSTIC REPORT
======================================================================

QUERY: Motor RPM dropped instantly during a full throttle climb.
TRUTH: voltage_sag
------------------------------
SYSTEM    | PREDICTION          | CONFIDENCE | DECISION
Baseline  | voltage_sag         | 98.7%      | ACT
Hybrid    | voltage_sag         | 98.7%      | ACT
>>> RESULT: Hybrid saved the mission.

...

======================================================================
FINAL PERFORMANCE SUMMARY
======================================================================
Baseline Correct Actions: 12/25
Hybrid Correct Actions:   20/25
Improvement:             32.0%
======================================================================
```

## Technical Domains Covered

- **Power**: Battery issues, ESC thermal shutdown, BEC failures
- **Sensors**: Pitot tube icing, barometric turbulence, lidar fog interference
- **Wiring**: Connector corrosion, servo potentiometer failures
- **Avionics**: Compass variance, PID oscillations, firmware issues
- **Communications**: VTx interference, telemetry frame loss
- **Airframe**: Structural resonance, motor misalignment
- **Propulsion**: Bearing failures, propeller issues

## Dependencies

- Python 3.x
- NumPy (for numerical operations and entropy calculation)

## License

This project is provided for educational and research purposes.
>>>>>>> 03ede9d (first commit)
