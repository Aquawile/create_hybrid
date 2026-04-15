"""
Entropy-Gated UAV Diagnostic System — Interactive Dashboard
"""
import streamlit as st
import json
import os
import time
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engine.retriever import DiagnosticRetriever
from engine.llm_interface import DiagnosticLLM
from engine.safety_gate import analyze_safety

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="AD-RAG Diagnostic System",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("✈️ Entropy-Gated UAV Diagnostic System")
st.caption("Real LLM • TF-IDF Retrieval • Shannon Entropy Abstention")

# ============================================================
# Sidebar: Configuration
# ============================================================
st.sidebar.header("⚙️ Configuration")
threshold = st.sidebar.slider(
    "Safety Threshold (bits)",
    min_value=0.5, max_value=3.0, value=2.0, step=0.1,
    help="Lower = more cautious. High entropy triggers ABSTAIN."
)
top_k = st.sidebar.slider("Documents to Retrieve", 1, 5, 2)

ollama_url = st.sidebar.text_input("Ollama URL", "http://localhost:11434")
model_name = st.sidebar.text_input("Ollama Model", "llama3")

# ============================================================
# Load Data (cached)
# ============================================================
@st.cache_resource
def load_system(model_name, base_url):
    retriever = DiagnosticRetriever(os.path.join(os.path.dirname(__file__), '..', 'data', 'corpus.json'))
    llm = DiagnosticLLM(model_name=model_name, base_url=base_url)
    return retriever, llm

@st.cache_data
def load_data():
    with open(os.path.join(os.path.dirname(__file__), '..', 'data', 'queries.json')) as f:
        queries = json.load(f)
    with open(os.path.join(os.path.dirname(__file__), '..', 'data', 'hypotheses.json')) as f:
        hypos = json.load(f)
    with open(os.path.join(os.path.dirname(__file__), '..', 'data', 'ground_truth.json')) as f:
        gt = json.load(f)
    return queries, hypos, gt

retriever, llm = load_system(model_name, ollama_url)
queries, hypos, gt = load_data()

# ============================================================
# Session State
# ============================================================
if 'session_log' not in st.session_state:
    st.session_state.session_log = []
if 'stats' not in st.session_state:
    st.session_state.stats = {
        "baseline_correct": 0, "baseline_wrong": 0, "baseline_abstain": 0,
        "hybrid_correct": 0, "hybrid_wrong": 0, "hybrid_abstain": 0,
        "total": 0,
    }

# ============================================================
# Sidebar: Query Selection
# ============================================================
st.sidebar.header("🎯 Query")
query_mode = st.sidebar.radio("Mode", ["Single Query", "Batch Run", "Custom Query"])

# ============================================================
# Main Display
# ============================================================

if query_mode == "Single Query":
    qid = st.sidebar.selectbox("Select Scenario", list(queries.keys()))
    run_btn = st.sidebar.button("▶ Run Diagnosis", type="primary")

    if run_btn:
        query_text = queries[qid]
        hypotheses = hypos[qid]
        truth = gt[qid]

        with st.spinner("Running baseline (TF-IDF)..."):
            b_docs = retriever.retrieve(query_text, hybrid=False, top_k=top_k)
            b_result = llm.generate_beliefs(b_docs, hypotheses, query_text)
            b_choice, b_entropy = analyze_safety(b_result["beliefs"], threshold=threshold)

        with st.spinner("Running hybrid (TF-IDF + keywords)..."):
            h_docs = retriever.retrieve(query_text, hybrid=True, top_k=top_k)
            h_result = llm.generate_beliefs(h_docs, hypotheses, query_text)
            h_choice, h_entropy = analyze_safety(h_result["beliefs"], threshold=threshold)

        # Update stats
        s = st.session_state.stats
        s["total"] += 1
        if b_choice is None: s["baseline_abstain"] += 1
        elif b_choice == truth: s["baseline_correct"] += 1
        else: s["baseline_wrong"] += 1
        if h_choice is None: s["hybrid_abstain"] += 1
        elif h_choice == truth: s["hybrid_correct"] += 1
        else: s["hybrid_wrong"] += 1

        st.session_state.session_log.append({
            "qid": qid, "query": query_text, "truth": truth,
            "baseline": {"choice": b_choice, "entropy": b_entropy, "abstain": b_choice is None},
            "hybrid": {"choice": h_choice, "entropy": h_entropy, "abstain": h_choice is None},
        })

    # ============================================================
    # Display Area
    # ============================================================
    if "b_result" in locals() and "h_result" in locals():
        # Query info
        st.info(f"**Query ({qid}):** {query_text}")
        st.info(f"**Ground Truth:** {truth}")

        # Two columns for comparison
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("📊 Baseline (TF-IDF)")
            b_docs_list = [d['id'] for d in b_docs]
            st.caption(f"Retrieved: {', '.join(b_docs_list)}")

            b_confidence = max(b_result["beliefs"].values()) if not b_result["abstain"] else 0.0
            b_decision = "ABSTAIN" if b_result["abstain"] else "ACT"

            if b_decision == "ABSTAIN":
                st.error(f"**Decision: ABSTAIN**")
            else:
                if b_choice == truth:
                    st.success(f"**Decision: ACT → {b_choice}**")
                else:
                    st.error(f"**Decision: ACT → {b_choice}**")

            st.metric("Confidence", f"{b_confidence:.1%}")
            st.metric("Entropy", f"{b_entropy:.2f} bits")

            # Entropy bar
            entropy_fraction = min(b_entropy / 3.0, 1.0)
            st.progress(entropy_fraction)

            if b_result.get("parsed"):
                st.caption("**Reasoning:**")
                st.caption(b_result["parsed"].get("reasoning", "N/A"))

        with col_b:
            st.subheader("📊 Hybrid (TF-IDF + Keywords)")
            h_docs_list = [d['id'] for d in h_docs]
            st.caption(f"Retrieved: {', '.join(h_docs_list)}")

            h_confidence = max(h_result["beliefs"].values()) if not h_result["abstain"] else 0.0
            h_decision = "ABSTAIN" if h_result["abstain"] else "ACT"

            if h_decision == "ABSTAIN":
                st.error(f"**Decision: ABSTAIN**")
            else:
                if h_choice == truth:
                    st.success(f"**Decision: ACT → {h_choice}**")
                else:
                    st.error(f"**Decision: ACT → {h_choice}**")

            st.metric("Confidence", f"{h_confidence:.1%}")
            st.metric("Entropy", f"{h_entropy:.2f} bits")

            entropy_fraction = min(h_entropy / 3.0, 1.0)
            st.progress(entropy_fraction)

            if h_result.get("parsed"):
                st.caption("**Reasoning:**")
                st.caption(h_result["parsed"].get("reasoning", "N/A"))

        # Outcome banner
        st.divider()
        if h_choice == truth and (b_choice != truth or b_choice is None):
            st.success("✅ Hybrid got it right while Baseline missed.")
        elif h_choice is None and b_choice is not None and b_choice != truth:
            st.warning("⚠️ Hybrid abstained, preventing a wrong action.")
        elif h_choice == truth and b_choice == truth:
            st.info("ℹ️ Both correct — retrieval is adequate.")
        elif h_choice is None and b_choice is None:
            st.info("ℹ️ Both abstained — insufficient evidence for either method.")
        elif h_choice != truth and b_choice != truth:
            st.error("❌ Both wrong — need better retrieval or model.")
        elif h_choice != truth:
            st.error("❌ Hybrid wrong while Baseline was correct.")

        # Retrieved documents display
        with st.expander("📄 Retrieved Documents (Baseline)"):
            for d in b_docs:
                st.markdown(f"**{d['id']}** [{d.get('metadata', {}).get('domain', 'N/A')}]: {d['text']}")

        with st.expander("📄 Retrieved Documents (Hybrid)"):
            for d in h_docs:
                st.markdown(f"**{d['id']}** [{d.get('metadata', {}).get('domain', 'N/A')}]: {d['text']}")

elif query_mode == "Batch Run":
    st.sidebar.info("Runs all 25 scenarios. This will take a few minutes.")
    batch_btn = st.sidebar.button("▶ Run Full Batch", type="primary")

    if batch_btn:
        progress = st.progress(0)
        status_text = st.empty()

        b_correct = b_wrong = b_abstain = 0
        h_correct = h_wrong = h_abstain = 0

        all_results = []
        total = len(queries)

        for i, qid in enumerate(queries):
            status_text.text(f"Running {i+1}/{total}: {qid}")
            progress.progress((i + 1) / total)

            query_text = queries[qid]
            hypotheses = hypos[qid]
            truth = gt[qid]

            b_docs = retriever.retrieve(query_text, hybrid=False, top_k=top_k)
            b_result = llm.generate_beliefs(b_docs, hypotheses, query_text)
            b_choice, b_entropy = analyze_safety(b_result["beliefs"], threshold=threshold)

            h_docs = retriever.retrieve(query_text, hybrid=True, top_k=top_k)
            h_result = llm.generate_beliefs(h_docs, hypotheses, query_text)
            h_choice, h_entropy = analyze_safety(h_result["beliefs"], threshold=threshold)

            if b_choice is None: b_abstain += 1
            elif b_choice == truth: b_correct += 1
            else: b_wrong += 1

            if h_choice is None: h_abstain += 1
            elif h_choice == truth: h_correct += 1
            else: h_wrong += 1

            all_results.append({
                "qid": qid, "truth": truth,
                "baseline": {"choice": b_choice, "entropy": round(b_entropy, 2)},
                "hybrid": {"choice": h_choice, "entropy": round(h_entropy, 2)},
            })

            time.sleep(0.5)  # Be gentle on Ollama

        status_text.text("✅ Batch complete!")

        # Display results
        st.divider()
        st.subheader("Batch Results")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Baseline Accuracy", f"{b_correct}/{total}",
                       f"{b_correct/total:.0%}" if total > 0 else "N/A")
            st.caption(f"Abstain: {b_abstain} | Wrong: {b_wrong}")
        with col2:
            st.metric("Hybrid Accuracy", f"{h_correct}/{total}",
                       f"{h_correct/total:.0%}" if total > 0 else "N/A")
            st.caption(f"Abstain: {h_abstain} | Wrong: {h_wrong}")

        # Results table
        st.divider()
        st.subheader("Detailed Results")

        table_data = []
        for r in all_results:
            table_data.append({
                "Query": r["qid"],
                "Truth": r["truth"],
                "Baseline": r["baseline"]["choice"] or "ABSTAIN",
                "B_Entropy": r["baseline"]["entropy"],
                "Hybrid": r["hybrid"]["choice"] or "ABSTAIN",
                "H_Entropy": r["hybrid"]["entropy"],
                "Hybrid Correct": "✅" if r["hybrid"]["choice"] == r["truth"] else "❌",
            })

        import pandas as pd
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

elif query_mode == "Custom Query":
    custom_query = st.text_area("Describe the UAV symptom:")
    custom_hypos = st.text_input("Candidate diagnoses (comma-separated):", 
                                  "voltage_sag, esc_thermal, propeller_loose")
    custom_run = st.button("▶ Diagnose", type="primary")

    if custom_run and custom_query:
        hypotheses = [h.strip() for h in custom_hypos.split(",")]

        with st.spinner("Retrieving documents..."):
            docs = retriever.retrieve(custom_query, hybrid=True, top_k=top_k)

        with st.spinner("Generating diagnosis..."):
            result = llm.generate_beliefs(docs, hypotheses, custom_query)
            choice, entropy = analyze_safety(result["beliefs"], threshold=threshold)

        st.info(f"**Query:** {custom_query}")

        col1, col2, col3 = st.columns(3)
        with col1:
            if result["abstain"] or choice is None:
                st.error("**ABSTAIN**")
            else:
                st.success(f"**{choice}**")
        with col2:
            st.metric("Confidence", f"{max(result['beliefs'].values()):.1%}")
        with col3:
            st.metric("Entropy", f"{entropy:.2f} bits")

        st.progress(min(entropy / 3.0, 1.0))

        # Evidence
        if result.get("parsed"):
            st.divider()
            p = result["parsed"]
            st.subheader("📋 Diagnosis Details")
            st.caption(f"**Reasoning:** {p.get('reasoning', 'N/A')}")
            st.caption(f"**Evidence:** {p.get('evidence', 'N/A')}")
            st.caption(f"**Documents:** {', '.join(p.get('document_ids', []))}")

            # Alternatives
            alt = p.get("alternatives", {})
            if alt:
                st.caption("**Alternatives considered:**")
                for a, c in alt.items():
                    st.caption(f"  • {a}: {c:.0%}")

# ============================================================
# Session Statistics (always shown at bottom)
# ============================================================
st.divider()
st.subheader("📈 Session Statistics")

stats = st.session_state.stats
total = stats["total"]

if total > 0:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Runs", total)
    with col2:
        b_acc = stats["baseline_correct"] / total if total > 0 else 0
        st.metric("Baseline Accuracy", f"{b_acc:.0%}")
        st.caption(f"✓ {stats['baseline_correct']} | ✗ {stats['baseline_wrong']} | ⊘ {stats['baseline_abstain']}")
    with col3:
        h_acc = stats["hybrid_correct"] / total if total > 0 else 0
        st.metric("Hybrid Accuracy", f"{h_acc:.0%}")
        st.caption(f"✓ {stats['hybrid_correct']} | ✗ {stats['hybrid_wrong']} | ⊘ {stats['hybrid_abstain']}")
else:
    st.caption("No runs yet. Use the sidebar to run a diagnosis.")

# Session log
if st.session_state.session_log:
    with st.expander("📋 Session Log"):
        for entry in st.session_state.session_log:
            b_ch = entry["baseline"]["choice"] or "ABSTAIN"
            h_ch = entry["hybrid"]["choice"] or "ABSTAIN"
            st.markdown(
                f"**{entry['qid']}** — Truth: `{entry['truth']}` | "
                f"Baseline: `{b_ch}` ({entry['baseline']['entropy']:.2f}b) | "
                f"Hybrid: `{h_ch}` ({entry['hybrid']['entropy']:.2f}b)"
            )
