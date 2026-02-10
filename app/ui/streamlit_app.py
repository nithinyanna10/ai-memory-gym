"""AI Memory Gym V2 — Cognitive Lab: full-scale evaluation, stress testing, premium UI."""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bench.schemas import BenchmarkConfig, SuiteRunConfig
from bench.runner import run_benchmark, save_result, result_to_dataframe, load_result
from bench.suite_runner import run_suite, load_suite_results
from bench.metrics_v2 import compute_metrics_v2

st.set_page_config(page_title="AI Memory Gym V2", layout="wide", initial_sidebar_state="expanded")
st.markdown("<style> .block-container { padding-top: 1rem; } </style>", unsafe_allow_html=True)
st.title("AI Memory Gym V2 — Cognitive Lab")

with st.sidebar:
    st.header("Configuration")
    scenario_type = st.selectbox(
        "Scenario",
        ["personal_assistant", "research", "ops", "sales_crm", "legal_contract", "meeting_memory", "multi_agent_handoff", "adversarial_injection"],
        index=0,
    )
    policy = st.selectbox(
        "Policy",
        ["no_memory", "full_log", "rolling_summary", "vector_rag", "hybrid_brain", "salience_only", "rehearsal"],
        index=1,
    )
    seed = st.number_input("Seed", value=42, min_value=0, step=1)
    number_of_days = st.number_input("Days", value=7, min_value=1, max_value=30, step=1)
    wm_size = st.number_input("WM size", value=10, min_value=1, max_value=50, step=1)
    top_k = st.number_input("Top K", value=5, min_value=1, max_value=20, step=1)
    llm_mode = st.radio("LLM", ["mock", "rule", "real"], index=0, horizontal=True)
    use_mock_llm = llm_mode != "real"
    stress_mode = st.selectbox("Stress mode", [None, "distraction_flood", "contradiction_injection", "distribution_shift", "memory_corruption"], format_func=lambda x: x or "None")
    if stress_mode == "distraction_flood":
        k_noise = st.number_input("k_noise", value=5, min_value=1, step=1)
        stress_kwargs = {"k_noise": k_noise}
    elif stress_mode == "contradiction_injection":
        p_contradict = st.slider("p_contradict", 0.0, 1.0, 0.5, 0.1)
        stress_kwargs = {"p_contradict": p_contradict}
    elif stress_mode == "distribution_shift":
        style_switch_day = st.number_input("style_switch_day", value=2, min_value=1, step=1)
        stress_kwargs = {"style_switch_day": style_switch_day}
    elif stress_mode == "memory_corruption":
        p_drop = st.slider("p_drop", 0.0, 0.5, 0.1, 0.05)
        p_mutate = st.slider("p_mutate", 0.0, 0.5, 0.1, 0.05)
        stress_kwargs = {"p_drop": p_drop, "p_mutate": p_mutate}
    else:
        stress_kwargs = {}
    st.divider()
    run_clicked = st.button("Run single benchmark")
    st.divider()
    st.subheader("Suite run")
    n_policies = st.number_input("Policies (batch)", value=2, min_value=1, max_value=7)
    n_scenarios = st.number_input("Scenarios (batch)", value=2, min_value=1, max_value=8)
    n_seeds = st.number_input("Seeds (batch)", value=2, min_value=1, max_value=5)
    run_suite_clicked = st.button("Run suite (batch)")

if "results" not in st.session_state:
    st.session_state.results = []
if "current_result" not in st.session_state:
    st.session_state.current_result = None
if "suite_results" not in st.session_state:
    st.session_state.suite_results = []

if run_clicked:
    with st.spinner("Running benchmark..."):
        config = BenchmarkConfig(
            scenario_type=scenario_type,
            policy=policy,
            seed=int(seed),
            number_of_days=int(number_of_days),
            wm_size=int(wm_size),
            top_k=int(top_k),
            decay_lambda=0.1,
            salience_threshold=0.3,
            rehearsal_frequency=3,
            use_mock_llm=use_mock_llm,
            llm_mode=llm_mode,
            stress_mode=stress_mode,
            stress_kwargs=stress_kwargs,
        )
        result = run_benchmark(config)
        st.session_state.current_result = result
        st.session_state.results.append(result)
    st.success("Benchmark complete.")

if run_suite_clicked:
    with st.spinner("Running suite (this may take a while)..."):
        policies = ["full_log", "vector_rag"][:n_policies]
        scenarios = ["personal_assistant", "research", "ops", "sales_crm", "legal_contract", "meeting_memory", "multi_agent_handoff", "adversarial_injection"][:n_scenarios]
        seeds = [42, 43, 44, 45, 46][:n_seeds]
        suite_config = SuiteRunConfig(
            policies=policies,
            scenarios=scenarios,
            seeds=seeds,
            number_of_days=min(5, number_of_days),
            stress_modes=[None],
            llm_mode=llm_mode,
            wm_size=wm_size,
            top_k=top_k,
        )
        suite_result = run_suite(suite_config, out_dir=str(ROOT / "data" / "runs"), use_cache=True)
        st.session_state.suite_results.append(suite_result)
    st.success(f"Suite complete: {len(suite_result.results)} runs. Aggregated: {suite_result.aggregated_csv_path}")

result = st.session_state.current_result
out_dir = ROOT / "data" / "runs"
out_dir.mkdir(parents=True, exist_ok=True)

tabs = st.tabs([
    "Overview", "Forgetting Curve", "Memory Timeline", "Retrieval Explainability", "Ablations", "Export",
    "Leaderboard", "Stress Lab", "Frontier", "Retention", "Interference Map", "Trace Viewer", "Report",
])

with tabs[0]:
    st.header("Key metrics")
    if result:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{result.accuracy:.2%}")
        c2.metric("Citation Precision", f"{result.citation_precision:.2%}")
        c3.metric("Citation Recall", f"{result.citation_recall:.2%}")
        c4.metric("Hallucination Rate", f"{result.hallucination_rate:.2%}")
        c5.metric("Memory Items", result.memory_items_stored)
        st.metric("Token estimate", result.token_estimate)
        metrics_v2 = getattr(result, "metrics_v2", None)
        if metrics_v2:
            st.subheader("V2 metrics")
            m = metrics_v2
            st.write(f"**M-Score:** {m.get('m_score', 0):.3f} | **Retention half-life (day):** {m.get('retention_half_life')} | **Interference rate:** {m.get('interference_rate', 0):.2%} | **Contradiction rate:** {m.get('contradiction_rate', 0):.2%}")
        if len(st.session_state.results) >= 2:
            st.subheader("Policy comparison")
            rows = [{"policy": r.config.policy, "scenario": r.config.scenario_type, "accuracy": r.accuracy, "citation_precision": r.citation_precision, "memory_items": r.memory_items_stored} for r in st.session_state.results]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("Run a benchmark to see metrics.")

with tabs[1]:
    st.header("Forgetting curve")
    if result and result.forgetting_curve:
        fig, ax = plt.subplots()
        days = [x[0] for x in result.forgetting_curve]
        accs = [x[1] for x in result.forgetting_curve]
        ax.plot(days, accs, marker="o", linestyle="-")
        ax.set_xlabel("Day")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy vs day")
        ax.set_ylim(0, 1.05)
        st.pyplot(fig)
        plt.close()
    else:
        st.info("Run a benchmark to see the forgetting curve.")

with tabs[2]:
    st.header("Memory timeline")
    if result and result.run_records:
        rows = [{"day": r.day, "turn": r.turn, "type": "episodic", "question": r.question or "(event)", "answer": (r.answer[:80] + "...") if len(r.answer) > 80 else r.answer} for r in result.run_records]
        df_timeline = pd.DataFrame(rows)
        day_filter = st.selectbox("Filter by day", ["All"] + list(map(str, sorted(df_timeline["day"].unique()))), key="tl_day")
        if day_filter != "All":
            df_timeline = df_timeline[df_timeline["day"] == int(day_filter)]
        st.dataframe(df_timeline, use_container_width=True)
    else:
        st.info("Run a benchmark to see the memory timeline.")

with tabs[3]:
    st.header("Retrieval explainability")
    if result and result.run_records:
        q_records = [r for r in result.run_records if r.question]
        if q_records:
            sel = st.selectbox("Select question", range(len(q_records)), format_func=lambda i: f"Day {q_records[i].day} — {(q_records[i].question or '')[:50]}...", key="ret_sel")
            r = q_records[sel]
            st.write("**Question:**", r.question)
            st.write("**Gold:**", r.gold_answer)
            st.write("**Model:**", r.answer)
            st.write("**Correct:**", r.correct)
            st.write("**Retrieved:**")
            for mid, reason, score in r.retrieved:
                st.write(f"  - `{mid}` ({reason}): {score:.3f}")
            st.write("**Citations:**", r.citations)
        else:
            st.info("No question steps.")
    else:
        st.info("Run a benchmark.")

with tabs[4]:
    st.header("Ablations")
    if len(st.session_state.results) >= 1:
        rows = [{"run_id": r.run_id, "policy": r.config.policy, "scenario": r.config.scenario_type, "stress": getattr(r.config, "stress_mode", None) or "-", "accuracy": r.accuracy, "m_score": (getattr(r, "metrics_v2", None) or {}).get("m_score")} for r in st.session_state.results]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("Run at least one benchmark.")

with tabs[5]:
    st.header("Export")
    if result:
        path = save_result(result, str(out_dir))
        st.success(f"Saved {path}")
        df = result_to_dataframe(result)
        csv_path = path.replace(".json", ".csv")
        df.to_csv(csv_path, index=False)
        with open(path) as f:
            json_content = f.read()
        st.download_button("Download JSON", json_content, file_name=f"run_{result.run_id}.json", mime="application/json")
        st.download_button("Download CSV", df.to_csv(index=False), file_name=f"run_{result.run_id}.csv", mime="text/csv")
    else:
        st.info("Run a benchmark to export.")

with tabs[6]:
    st.header("Leaderboard")
    all_suites = load_suite_results(str(out_dir))
    if all_suites or st.session_state.suite_results:
        rows = []
        for sr in st.session_state.suite_results:
            for r in sr.results:
                m = getattr(r, "metrics_v2", None) or {}
                rows.append({"run_id": r.run_id, "scenario": r.config.scenario_type, "policy": r.config.policy, "stress": r.config.stress_mode or "-", "accuracy": r.accuracy, "m_score": m.get("m_score"), "tokens": r.token_estimate})
        for man in all_suites:
            csv_p = man.get("aggregated_csv")
            if csv_p and Path(csv_p).exists():
                try:
                    dfm = pd.read_csv(csv_p)
                    for _, row in dfm.iterrows():
                        rows.append({"run_id": row.get("run_id"), "scenario": row.get("scenario"), "policy": row.get("policy"), "stress": row.get("stress_mode", "-"), "accuracy": row.get("accuracy"), "m_score": row.get("m_score"), "tokens": row.get("token_estimate")})
                except Exception:
                    pass
        if rows:
            df_lead = pd.DataFrame(rows)
            scenario_filter = st.selectbox("Filter scenario", ["All"] + list(df_lead["scenario"].dropna().unique().tolist()), key="lead_sc")
            if scenario_filter != "All":
                df_lead = df_lead[df_lead["scenario"] == scenario_filter]
            df_lead = df_lead.sort_values("m_score", ascending=False) if "m_score" in df_lead.columns else df_lead.sort_values("accuracy", ascending=False)
            st.dataframe(df_lead, use_container_width=True)
        else:
            st.info("No suite data yet.")
    else:
        st.info("Run a suite (batch) to see the leaderboard.")

with tabs[7]:
    st.header("Stress Lab")
    st.write("Toggle stress modes in the sidebar and run benchmarks to compare.")
    if result and getattr(result.config, "stress_mode", None):
        st.write(f"Current run used stress: **{result.config.stress_mode}** with {result.config.stress_kwargs}")
    if len(st.session_state.results) >= 2:
        with_stress = [r for r in st.session_state.results if getattr(r.config, "stress_mode", None)]
        no_stress = [r for r in st.session_state.results if not getattr(r.config, "stress_mode", None)]
        if with_stress and no_stress:
            fig, ax = plt.subplots()
            ax.bar(["No stress", "With stress"], [no_stress[0].accuracy, with_stress[0].accuracy])
            ax.set_ylabel("Accuracy")
            st.pyplot(fig)
            plt.close()
    else:
        st.info("Run at least 2 benchmarks (one with stress, one without) to compare.")

with tabs[8]:
    st.header("Cost vs Quality Frontier")
    if result or (st.session_state.results or st.session_state.suite_results):
        rows = []
        for r in st.session_state.results:
            rows.append({"accuracy": r.accuracy, "tokens": r.token_estimate, "policy": r.config.policy})
        for sr in st.session_state.suite_results:
            for r in sr.results:
                rows.append({"accuracy": r.accuracy, "tokens": r.token_estimate, "policy": r.config.policy})
        if rows:
            df_f = pd.DataFrame(rows)
            fig, ax = plt.subplots()
            for pol in df_f["policy"].unique():
                sub = df_f[df_f["policy"] == pol]
                ax.scatter(sub["tokens"], sub["accuracy"], label=pol, alpha=0.7)
            ax.set_xlabel("Token estimate")
            ax.set_ylabel("Accuracy")
            ax.legend()
            ax.set_title("Cost vs Quality")
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Run benchmarks to see frontier.")
    else:
        st.info("Run benchmarks to see cost vs quality.")

with tabs[9]:
    st.header("Retention half-life")
    metrics_v2 = getattr(result, "metrics_v2", None) if result else None
    if result and metrics_v2:
        half_life = metrics_v2.get("retention_half_life")
        st.metric("Retention half-life (day)", half_life if half_life is not None else "N/A")
        if result.forgetting_curve:
            fig, ax = plt.subplots()
            days = [x[0] for x in result.forgetting_curve]
            accs = [x[1] for x in result.forgetting_curve]
            ax.plot(days, accs, marker="o")
            ax.axhline(0.5, color="gray", linestyle="--")
            ax.set_xlabel("Day")
            ax.set_ylabel("Accuracy")
            st.pyplot(fig)
            plt.close()
    else:
        st.info("Run a benchmark to see retention.")

with tabs[10]:
    st.header("Interference map (policy × scenario)")
    if st.session_state.suite_results or load_suite_results(str(out_dir)):
        rows = []
        for sr in st.session_state.suite_results:
            for r in sr.results:
                rows.append({"policy": r.config.policy, "scenario": r.config.scenario_type, "interference": (getattr(r, "metrics_v2", None) or {}).get("interference_rate", 0)})
        for man in load_suite_results(str(out_dir)):
            csv_p = man.get("aggregated_csv")
            if csv_p and Path(csv_p).exists():
                try:
                    dfm = pd.read_csv(csv_p)
                    if "retention_half_life" in dfm.columns or "accuracy" in dfm.columns:
                        for _, row in dfm.iterrows():
                            rows.append({"policy": row.get("policy"), "scenario": row.get("scenario"), "interference": 0})
                except Exception:
                    pass
        if rows:
            df_i = pd.DataFrame(rows)
            pivot = df_i.pivot_table(index="policy", columns="scenario", values="interference", aggfunc="mean")
            fig, ax = plt.subplots()
            im = ax.imshow(pivot.fillna(0).values, aspect="auto")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            plt.colorbar(im, ax=ax, label="Interference rate")
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Need suite runs with metrics_v2.")
    else:
        st.info("Run a suite to see interference map.")

with tabs[11]:
    st.header("Trace viewer")
    run_options = [r.run_id for r in st.session_state.results]
    if run_options:
        run_id_sel = st.selectbox("Select run", run_options, key="trace_run")
        res = next((r for r in st.session_state.results if r.run_id == run_id_sel), None)
        if res and res.run_records:
            step_i = st.slider("Step", 0, len(res.run_records) - 1, 0)
            rec = res.run_records[step_i]
            st.write("**Day / Turn:**", rec.day, rec.turn)
            st.write("**Question:**", rec.question or "(event)")
            st.write("**Gold:**", rec.gold_answer)
            st.write("**Answer:**", rec.answer)
            st.write("**Correct:**", rec.correct)
            st.write("**Retrieved:**", rec.retrieved)
            st.write("**Citations:**", rec.citations)
    else:
        st.info("Run a benchmark to view trace.")

with tabs[12]:
    st.header("Report export (PDF)")
    if result:
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib import colors
            pdf_path = str(out_dir / f"report_{result.run_id}.pdf")
            doc = SimpleDocTemplate(pdf_path, pagesize=letter)
            styles = getSampleStyleSheet()
            story = [Paragraph("AI Memory Gym V2 — Run Report", styles["Title"]), Spacer(1, 12)]
            story.append(Paragraph(f"Run ID: {result.run_id}", styles["Normal"]))
            story.append(Paragraph(f"Scenario: {result.config.scenario_type} | Policy: {result.config.policy}", styles["Normal"]))
            story.append(Paragraph(f"Accuracy: {result.accuracy:.2%} | Tokens: {result.token_estimate}", styles["Normal"]))
            mv2 = getattr(result, "metrics_v2", None)
            if mv2:
                story.append(Paragraph(f"M-Score: {mv2.get('m_score', 0):.3f}", styles["Normal"]))
            story.append(Spacer(1, 20))
            data = [["Metric", "Value"], ["Accuracy", f"{result.accuracy:.2%}"], ["Citation Precision", f"{result.citation_precision:.2%}"], ["Memory Items", str(result.memory_items_stored)]]
            t = Table(data)
            t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.grey), ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke)]))
            story.append(t)
            doc.build(story)
            with open(pdf_path, "rb") as f:
                st.download_button("Download PDF report", f.read(), file_name=f"report_{result.run_id}.pdf", mime="application/pdf")
        except Exception as e:
            st.warning(f"PDF export failed: {e}. Install reportlab.")
    else:
        st.info("Run a benchmark to export a report.")
