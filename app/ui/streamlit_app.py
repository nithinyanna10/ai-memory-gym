"""Streamlit dashboard: overview, forgetting curve, memory timeline, explainability, ablations."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bench.schemas import BenchmarkConfig
from bench.runner import run_benchmark, save_result, result_to_dataframe


st.set_page_config(page_title="AI Memory Gym", layout="wide")
st.title("AI Memory Test — Cognitive Lab")

with st.sidebar:
    st.header("Configuration")
    scenario_type = st.selectbox("Scenario type", ["personal_assistant", "research", "ops"], index=0)
    policy = st.selectbox(
        "Memory policy",
        ["no_memory", "full_log", "rolling_summary", "vector_rag", "hybrid_brain", "salience_only", "rehearsal"],
        index=1,
    )
    seed = st.number_input("Seed", value=42, min_value=0, step=1)
    number_of_days = st.number_input("Number of days", value=7, min_value=1, max_value=30, step=1)
    wm_size = st.number_input("WM size", value=10, min_value=1, max_value=50, step=1)
    top_k = st.number_input("Top K", value=5, min_value=1, max_value=20, step=1)
    decay_lambda = st.slider("Decay lambda", 0.01, 0.5, 0.1, 0.01)
    salience_threshold = st.slider("Salience threshold", 0.0, 1.0, 0.3, 0.05)
    rehearsal_frequency = st.number_input("Rehearsal frequency (days)", value=3, min_value=1, step=1)
    use_mock_llm = st.checkbox("Use Mock LLM (no API key)", value=True)
    run_clicked = st.button("Run benchmark")

if "results" not in st.session_state:
    st.session_state.results = []
if "current_result" not in st.session_state:
    st.session_state.current_result = None

if run_clicked:
    with st.spinner("Running benchmark..."):
        config = BenchmarkConfig(
            scenario_type=scenario_type,
            policy=policy,
            seed=int(seed),
            number_of_days=int(number_of_days),
            wm_size=int(wm_size),
            top_k=int(top_k),
            decay_lambda=float(decay_lambda),
            salience_threshold=float(salience_threshold),
            rehearsal_frequency=int(rehearsal_frequency),
            use_mock_llm=use_mock_llm,
        )
        result = run_benchmark(config)
        st.session_state.current_result = result
        st.session_state.results.append(result)
    st.success("Benchmark complete.")

result = st.session_state.current_result

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview", "Forgetting Curve", "Memory Timeline", "Retrieval Explainability", "Ablations", "Export",
])

with tab1:
    st.header("Key metrics")
    if result:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{result.accuracy:.2%}")
        c2.metric("Citation Precision", f"{result.citation_precision:.2%}")
        c3.metric("Citation Recall", f"{result.citation_recall:.2%}")
        c4.metric("Hallucination Rate", f"{result.hallucination_rate:.2%}")
        c5.metric("Memory Items", result.memory_items_stored)
        st.metric("Token estimate", result.token_estimate)
        st.metric("Avg retrieval latency (s)", f"{result.retrieval_latency_avg_s:.4f}")
        if len(st.session_state.results) >= 2:
            st.subheader("Policy comparison")
            rows = [{"policy": r.config.policy, "scenario": r.config.scenario_type, "accuracy": r.accuracy,
                     "citation_precision": r.citation_precision, "memory_items": r.memory_items_stored}
                    for r in st.session_state.results]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("Run a benchmark to see metrics.")

with tab2:
    st.header("Forgetting curve")
    if result and result.forgetting_curve:
        fig, ax = plt.subplots()
        days = [x[0] for x in result.forgetting_curve]
        accs = [x[1] for x in result.forgetting_curve]
        ax.plot(days, accs, marker="o", linestyle="-")
        ax.set_xlabel("Day")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy vs day (forgetting curve)")
        ax.set_ylim(0, 1.05)
        st.pyplot(fig)
        plt.close()
    else:
        st.info("Run a benchmark to see the forgetting curve.")

with tab3:
    st.header("Memory timeline")
    if result and result.run_records:
        rows = [{"day": r.day, "turn": r.turn, "type": "episodic", "question": r.question or "(event)",
                 "answer": r.answer[:80] + "..." if len(r.answer) > 80 else r.answer} for r in result.run_records]
        df_timeline = pd.DataFrame(rows)
        day_filter = st.selectbox("Filter by day", ["All"] + list(map(str, sorted(df_timeline["day"].unique()))))
        if day_filter != "All":
            df_timeline = df_timeline[df_timeline["day"] == int(day_filter)]
        type_filter = st.selectbox("Filter by type", ["All", "episodic"])
        if type_filter != "All":
            df_timeline = df_timeline[df_timeline["type"] == type_filter]
        st.dataframe(df_timeline, use_container_width=True)
    else:
        st.info("Run a benchmark to see the memory timeline.")

with tab4:
    st.header("Retrieval explainability")
    if result and result.run_records:
        q_records = [r for r in result.run_records if r.question]
        if q_records:
            sel = st.selectbox("Select question", range(len(q_records)),
                               format_func=lambda i: f"Day {q_records[i].day} — {q_records[i].question[:50]}...")
            r = q_records[sel]
            st.write("**Question:**", r.question)
            st.write("**Gold answer:**", r.gold_answer)
            st.write("**Model answer:**", r.answer)
            st.write("**Correct:**", r.correct)
            st.write("**Retrieved memories (id, reason, score):**")
            for mid, reason, score in r.retrieved:
                st.write(f"  - `{mid}` ({reason}): {score:.3f}")
            st.write("**Citations:**", r.citations)
        else:
            st.info("No question steps in this run.")
    else:
        st.info("Run a benchmark to see retrieval explainability.")

with tab5:
    st.header("Ablations")
    st.write("Compare runs with different knobs.")
    if len(st.session_state.results) >= 1:
        rows = [{"run_id": r.run_id, "policy": r.config.policy, "scenario": r.config.scenario_type,
                 "wm_size": r.config.wm_size, "top_k": r.config.top_k, "decay_lambda": r.config.decay_lambda,
                 "accuracy": r.accuracy, "citation_precision": r.citation_precision, "citation_recall": r.citation_recall,
                 "memory_items": r.memory_items_stored} for r in st.session_state.results]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("Run at least one benchmark to see ablations.")

with tab6:
    st.header("Export")
    if result:
        out_dir = ROOT / "data" / "runs"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = save_result(result, str(out_dir))
        st.success(f"Saved JSON to {path}")
        df = result_to_dataframe(result)
        csv_path = path.replace(".json", ".csv")
        df.to_csv(csv_path, index=False)
        st.success(f"Saved CSV to {csv_path}")
        with open(path) as f:
            json_content = f.read()
        st.download_button("Download JSON", json_content, file_name=f"run_{result.run_id}.json", mime="application/json")
        st.download_button("Download CSV", df.to_csv(index=False), file_name=f"run_{result.run_id}.csv", mime="text/csv")
    else:
        st.info("Run a benchmark to export.")
