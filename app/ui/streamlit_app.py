"""AI Memory Gym V2 — Cognitive Lab: production-grade evaluation UI."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bench.schemas import BenchmarkConfig, SuiteRunConfig
from bench.runner import run_benchmark, save_result, result_to_dataframe
from bench.suite_runner import run_suite, load_suite_results

st.set_page_config(page_title="AI Memory Gym V2", layout="wide", initial_sidebar_state="expanded")

# Load custom CSS
_css_path = Path(__file__).parent / "styles.css"
if _css_path.exists():
    st.markdown(f"<style>{_css_path.read_text()}</style>", unsafe_allow_html=True)
st.markdown("<style>.block-container { padding-top: 0.5rem; }</style>", unsafe_allow_html=True)

# Session state
if "results" not in st.session_state:
    st.session_state.results = []
if "current_result" not in st.session_state:
    st.session_state.current_result = None
if "suite_results" not in st.session_state:
    st.session_state.suite_results = []
if "last_run_cached" not in st.session_state:
    st.session_state.last_run_cached = False
if "presets" not in st.session_state:
    st.session_state.presets = {}
if "suite_progress" not in st.session_state:
    st.session_state.suite_progress = {"current": 0, "total": 0, "log": []}

DATA_DIR = ROOT / "data" / "runs"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SCENARIO_OPTIONS = [
    "personal_assistant", "personal_prefs", "research", "ops", "sales_crm",
    "legal_contract", "meeting_memory", "multi_agent_handoff", "adversarial_injection",
    "research_long", "ops_noisy_slack", "safety_pii", "tool_use_procedure",
]
POLICY_OPTIONS = ["no_memory", "full_log", "rolling_summary", "vector_rag", "hybrid_brain", "salience_only", "rehearsal"]


# ---------- Sidebar: Experiment Builder ----------
with st.sidebar:
    st.header("Experiment Builder")

    st.subheader("Scenario Pack")
    scenario_type = st.selectbox("Scenario", SCENARIO_OPTIONS, index=0, key="sb_scenario")

    st.subheader("Policy")
    policy = st.selectbox("Policy", POLICY_OPTIONS, index=1, key="sb_policy")

    st.subheader("Stress")
    stress_mode = st.selectbox(
        "Stress mode",
        [None, "distraction_flood", "contradiction_injection", "distribution_shift", "memory_corruption"],
        format_func=lambda x: x or "None",
        key="sb_stress",
    )
    stress_kwargs = {}
    if stress_mode == "distraction_flood":
        stress_kwargs["k_noise"] = st.number_input("k_noise", value=5, min_value=1, step=1, key="sb_k_noise")
        stress_kwargs["similarity_to_target"] = st.slider("similarity_to_target", 0.0, 1.0, 0.0, 0.1, key="sb_sim")
    elif stress_mode == "contradiction_injection":
        stress_kwargs["p_contradict"] = st.slider("p_contradict", 0.0, 1.0, 0.5, 0.1, key="sb_p_contradict")
        stress_kwargs["targeted"] = st.checkbox("targeted", value=True, key="sb_targeted")
    elif stress_mode == "distribution_shift":
        stress_kwargs["style_switch_day"] = st.number_input("style_switch_day", value=2, min_value=1, step=1, key="sb_style_day")
        stress_kwargs["shift_style"] = ("slack", "formal", "noisy")
    elif stress_mode == "memory_corruption":
        stress_kwargs["p_drop"] = st.slider("p_drop", 0.0, 0.5, 0.1, 0.05, key="sb_p_drop")
        stress_kwargs["p_mutate"] = st.slider("p_mutate", 0.0, 0.5, 0.1, 0.05, key="sb_p_mutate")
        stress_kwargs["mutate_strength"] = st.slider("mutate_strength", 0.0, 1.0, 1.0, 0.1, key="sb_mutate_strength")

    st.subheader("Advanced")
    seed = st.number_input("Seed", value=42, min_value=0, step=1, key="sb_seed")
    number_of_days = st.number_input("Days", value=7, min_value=1, max_value=30, step=1, key="sb_days")
    wm_size = st.number_input("WM size", value=10, min_value=1, max_value=50, step=1, key="sb_wm")
    top_k = st.number_input("Top K", value=5, min_value=1, max_value=20, step=1, key="sb_topk")
    llm_mode = st.radio("LLM", ["mock", "rule", "real"], index=0, horizontal=True, key="sb_llm")

    st.subheader("Save / Load preset")
    preset_name = st.text_input("Preset name", value="", key="preset_name")
    col_save, col_load = st.columns(2)
    with col_save:
        if st.button("Save preset"):
            st.session_state.presets[preset_name or "default"] = {
                "scenario_type": scenario_type, "policy": policy, "stress_mode": stress_mode,
                "stress_kwargs": stress_kwargs, "seed": seed, "number_of_days": number_of_days,
                "wm_size": wm_size, "top_k": top_k, "llm_mode": llm_mode,
            }
            st.success("Saved.")
    with col_load:
        preset_choice = st.selectbox("Load", ["(none)"] + list(st.session_state.presets.keys()), key="preset_load")
        if st.button("Load preset") and preset_choice != "(none)":
            p = st.session_state.presets[preset_choice]
            st.session_state.sb_scenario = p.get("scenario_type", scenario_type)
            st.session_state.sb_policy = p.get("policy", policy)
            st.session_state.sb_stress = p.get("stress_mode")
            st.session_state.sb_seed = p.get("seed", 42)
            st.session_state.sb_days = p.get("number_of_days", 7)
            st.session_state.sb_wm = p.get("wm_size", 10)
            st.session_state.sb_topk = p.get("top_k", 5)
            st.session_state.sb_llm = p.get("llm_mode", "mock")
            st.rerun()


def build_config():
    return BenchmarkConfig(
        scenario_type=scenario_type,
        policy=policy,
        seed=int(seed),
        number_of_days=int(number_of_days),
        wm_size=int(wm_size),
        top_k=int(top_k),
        decay_lambda=0.1,
        salience_threshold=0.3,
        rehearsal_frequency=3,
        use_mock_llm=(llm_mode != "real"),
        llm_mode=llm_mode,
        stress_mode=stress_mode,
        stress_kwargs=stress_kwargs,
    )


# ---------- Top header ----------
header_col1, header_col2, header_col3 = st.columns([2, 1, 1])
with header_col1:
    st.title("AI Memory Gym V2 — Cognitive Lab")
with header_col2:
    run_single_clicked = st.button("Run single")
    run_suite_clicked = st.button("Run suite")
with header_col3:
    result = st.session_state.current_result
    status = "cached" if st.session_state.last_run_cached else ("ready" if result else "ready")
    st.markdown(f'<span class="status-pill {status}">{status}</span>', unsafe_allow_html=True)
    if result and result.run_id:
        st.code(result.run_id, language=None)
    if result and st.session_state.last_run_cached:
        st.caption("(cached)")

# Run handlers
if run_single_clicked:
    st.session_state.last_run_cached = False
    with st.spinner("Running benchmark..."):
        config = build_config()
        result = run_benchmark(config)
        save_result(result, str(DATA_DIR))
        st.session_state.current_result = result
        st.session_state.results.append(result)
    st.success("Run complete.")
    st.rerun()

if run_suite_clicked:
    st.session_state.last_run_cached = False
    policies = ["full_log", "vector_rag", "rolling_summary"][:2]
    scenarios = SCENARIO_OPTIONS[:4]
    seeds = [42, 43][:2]
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
    with st.spinner("Running suite (progress in terminal if running from CLI)..."):
        suite_result = run_suite(suite_config, out_dir=str(DATA_DIR), use_cache=True)
    st.session_state.suite_results.append(suite_result)
    if suite_result.results:
        st.session_state.current_result = suite_result.results[0]
    st.success(f"Suite complete: {len(suite_result.results)} runs. Aggregated: {suite_result.aggregated_csv_path}")

# ---------- Main: view selector ----------
view = st.radio(
    "View",
    ["Dashboard", "Leaderboard", "Stress Lab", "Traces", "Memory MRI", "Reports", "Legacy"],
    horizontal=True,
    key="main_view",
)

result = st.session_state.current_result

# ---------- Dashboard ----------
if view == "Dashboard":
    if result:
        mv2 = getattr(result, "metrics_v2", None) or {}
        st.subheader("Metric cards")
        q1, q2, q3, q4 = st.columns(4)
        with q1:
            st.metric("Accuracy (Quality)", f"{result.accuracy:.2%}", delta=None)
        with q2:
            half_life = mv2.get("retention_half_life")
            st.metric("Retention half-life (Reliability)", str(half_life) if half_life is not None else "N/A", delta=None)
        with q3:
            st.metric("Token estimate (Cost)", result.token_estimate, delta=None)
        with q4:
            pii = mv2.get("pii_leakage_rate", 0)
            st.metric("PII leakage (Safety)", f"{pii:.2%}", delta=None)

        st.subheader("Pareto: Accuracy vs Token Cost")
        all_rows = []
        for r in st.session_state.results:
            all_rows.append({"accuracy": r.accuracy, "tokens": r.token_estimate, "policy": r.config.policy, "run_id": r.run_id})
        for sr in st.session_state.suite_results:
            for r in sr.results:
                all_rows.append({"accuracy": r.accuracy, "tokens": r.token_estimate, "policy": r.config.policy, "run_id": r.run_id})
        if all_rows:
            df_p = pd.DataFrame(all_rows)
            fig, ax = plt.subplots()
            for pol in df_p["policy"].unique():
                sub = df_p[df_p["policy"] == pol]
                ax.scatter(sub["tokens"], sub["accuracy"], label=pol, alpha=0.7)
            if result:
                ax.scatter([result.token_estimate], [result.accuracy], s=200, marker="*", color="red", zorder=5, label="Current run")
            ax.set_xlabel("Token estimate")
            ax.set_ylabel("Accuracy")
            ax.legend()
            ax.set_title("Accuracy vs Token Cost")
            st.pyplot(fig)
            plt.close()

        st.subheader("Forgetting curve")
        if result.forgetting_curve:
            fig, ax = plt.subplots()
            days = [x[0] for x in result.forgetting_curve]
            accs = [x[1] for x in result.forgetting_curve]
            ax.plot(days, accs, marker="o", linestyle="-", label="Current run")
            if len(st.session_state.results) > 1 and any(r.forgetting_curve for r in st.session_state.results):
                accs_list = [[y for _, y in r.forgetting_curve] for r in st.session_state.results if r.forgetting_curve]
                if accs_list and len(accs_list[0]) == len(days):
                    mean_acc = np.mean(accs_list, axis=0)
                    std_acc = np.std(accs_list, axis=0)
                    ax.fill_between(days, mean_acc - std_acc, mean_acc + std_acc, alpha=0.3)
                    ax.plot(days, mean_acc, linestyle="--", label="Mean ± 1 std")
            ax.set_xlabel("Day")
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0, 1.05)
            ax.legend()
            st.pyplot(fig)
            plt.close()

        st.subheader("Error taxonomy & top failures")
        wrong = [r for r in result.run_records if not r.correct and r.question]
        if wrong:
            fail_df = pd.DataFrame([{"day": r.day, "question": (r.question or "")[:60], "gold": (r.gold_answer or "")[:40], "answer": (r.answer or "")[:40]} for r in wrong[:10]])
            st.dataframe(fail_df, use_container_width=True)
            fig, ax = plt.subplots()
            ax.pie([len(wrong), max(0, len([r for r in result.run_records if r.question]) - len(wrong))], labels=["Wrong", "Correct"], autopct="%1.0f%%")
            st.pyplot(fig)
            plt.close()
        else:
            st.info("No incorrect answers in this run.")
    else:
        st.markdown('<div class="empty-state"><strong>Run experiment</strong><br/>Use the sidebar to configure and click "Run single" or "Run suite".</div>', unsafe_allow_html=True)
        st.caption("Example: Scenario = personal_assistant, Policy = full_log, Seed = 42.")

# ---------- Leaderboard ----------
if view == "Leaderboard":
    all_suites = load_suite_results(str(DATA_DIR))
    rows = []
    for sr in st.session_state.suite_results:
        for r in sr.results:
            m = getattr(r, "metrics_v2", None) or {}
            rows.append({"run_id": r.run_id, "scenario": r.config.scenario_type, "policy": r.config.policy, "stress": getattr(r.config, "stress_mode", None) or "-", "seed": r.config.seed, "accuracy": r.accuracy, "m_score": m.get("m_score"), "tokens": r.token_estimate})
    for man in all_suites:
        csv_p = man.get("aggregated_csv")
        if csv_p and Path(csv_p).exists():
            try:
                dfm = pd.read_csv(csv_p)
                for _, row in dfm.iterrows():
                    rows.append({"run_id": row.get("run_id"), "scenario": row.get("scenario"), "policy": row.get("policy"), "stress": row.get("stress_mode", "-"), "seed": row.get("seed"), "accuracy": row.get("accuracy"), "m_score": row.get("m_score"), "tokens": row.get("token_estimate")})
            except Exception:
                pass
    if rows:
        df_lead = pd.DataFrame(rows)
        sc_filter = st.selectbox("Filter scenario", ["All"] + list(df_lead["scenario"].dropna().unique().tolist()), key="lead_sc")
        if sc_filter != "All":
            df_lead = df_lead[df_lead["scenario"] == sc_filter]
        stress_filter = st.selectbox("Filter stress", ["All"] + list(df_lead["stress"].dropna().unique().tolist()), key="lead_stress")
        if stress_filter != "All":
            df_lead = df_lead[df_lead["stress"] == stress_filter]
        sort_col = "m_score" if "m_score" in df_lead.columns and df_lead["m_score"].notna().any() else "accuracy"
        df_lead = df_lead.sort_values(sort_col, ascending=False)
        st.dataframe(df_lead, use_container_width=True)
        if "m_score" in df_lead.columns and df_lead["m_score"].notna().any() and len(df_lead) > 1:
            fig, ax = plt.subplots()
            df_lead.dropna(subset=["m_score"]).boxplot(column="m_score", by="policy", ax=ax)
            ax.set_xlabel("Policy")
            ax.set_ylabel("M-Score")
            st.pyplot(fig)
            plt.close()
    else:
        st.markdown('<div class="empty-state"><strong>No leaderboard data</strong><br/>Run a suite to see rankings.</div>', unsafe_allow_html=True)

# ---------- Stress Lab ----------
if view == "Stress Lab":
    st.write("Toggle stress modes in the sidebar and run benchmarks to compare.")
    if result and getattr(result.config, "stress_mode", None):
        st.write(f"Current run stress: **{result.config.stress_mode}** with `{result.config.stress_kwargs}`")
    with_stress = [r for r in st.session_state.results if getattr(r.config, "stress_mode", None)]
    no_stress = [r for r in st.session_state.results if not getattr(r.config, "stress_mode", None)]
    if with_stress and no_stress:
        fig, ax = plt.subplots()
        ax.bar(["No stress", "With stress"], [no_stress[0].accuracy, with_stress[0].accuracy])
        ax.set_ylabel("Accuracy")
        st.pyplot(fig)
        plt.close()
    rows_heat = []
    for sr in st.session_state.suite_results:
        for r in sr.results:
            rows_heat.append({"policy": r.config.policy, "scenario": r.config.scenario_type, "accuracy": r.accuracy, "stress": getattr(r.config, "stress_mode", None) or "none"})
    if rows_heat:
        df_h = pd.DataFrame(rows_heat)
        pivot = df_h.pivot_table(index="policy", columns="scenario", values="accuracy", aggfunc="mean")
        fig, ax = plt.subplots()
        im = ax.imshow(pivot.fillna(0).values, aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        plt.colorbar(im, ax=ax, label="Accuracy")
        st.pyplot(fig)
        plt.close()
    else:
        st.info("Run benchmarks with and without stress to see comparison.")

# ---------- Traces ----------
if view == "Traces":
    run_options = [r.run_id for r in st.session_state.results if r.run_id]
    if run_options:
        run_id_sel = st.selectbox("Run", run_options, key="trace_run")
        res = next((r for r in st.session_state.results if r.run_id == run_id_sel), None)
        if res and res.run_records:
            step_i = st.slider("Step", 0, len(res.run_records) - 1, 0, key="trace_step")
            rec = res.run_records[step_i]
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Prompt**")
                prompt_text = getattr(rec, "prompt_text", None)
                st.text(prompt_text[:3000] if prompt_text else "(not recorded)")
            with c2:
                st.write("**Retrieved memories** (scores & reasons)")
                for mid, reason, score in rec.retrieved:
                    st.caption(f"`{mid}` — {reason}: {score:.3f}")
            st.write("**Memory updates**")
            updates = getattr(rec, "memory_updates", None) or []
            for u in updates:
                st.json(u)
            st.write("**Answer**")
            st.write(rec.answer)
            st.write("**Gold**")
            st.write(rec.gold_answer or "-")
            st.write("**Correct**")
            st.write(rec.correct)
            if rec.retrieved:
                best = max(rec.retrieved, key=lambda x: x[2])
                st.caption(f"Highest influence (heuristic): {best[0]} — {best[1]} ({best[2]:.3f})")
    else:
        st.markdown('<div class="empty-state"><strong>No traces</strong><br/>Run a benchmark to view step-by-step traces.</div>', unsafe_allow_html=True)

# ---------- Memory MRI ----------
if view == "Memory MRI":
    run_options = [r.run_id for r in st.session_state.results if r.run_id]
    if run_options:
        run_id_mri = st.selectbox("Run for MRI", run_options, key="mri_run")
        res = next((r for r in st.session_state.results if r.run_id == run_id_mri), None)
        if res and res.run_records:
            texts, days, types_list = [], [], []
            for r in res.run_records:
                for mid, reason, score in r.retrieved:
                    texts.append(f"{mid} {reason}")
                    types_list.append("retrieved")
                    days.append(r.day)
                for u in (getattr(r, "memory_updates", None) or []):
                    snip = u.get("text_snippet", u.get("id", ""))
                    texts.append(snip[:200])
                    types_list.append("update")
                    days.append(r.day)
            if len(texts) >= 2:
                from numpy.linalg import svd
                vocab = {}
                for t in texts:
                    for w in t.lower().split():
                        vocab.setdefault(w, len(vocab))
                X = np.zeros((len(texts), len(vocab)))
                for i, t in enumerate(texts):
                    for w in t.lower().split():
                        if w in vocab:
                            X[i, vocab[w]] += 1
                X = X - X.mean(axis=0)
                U, S, Vt = svd(X, full_matrices=False)
                coords = U[:, :2] * S[:2]
                fig, ax = plt.subplots()
                scatter = ax.scatter(coords[:, 0], coords[:, 1], c=days, cmap="viridis", s=50, alpha=0.8)
                for i, (x, y) in enumerate(coords):
                    ax.annotate(str(days[i]), (x, y), fontsize=6)
                plt.colorbar(scatter, ax=ax, label="Day")
                ax.set_title("Memory projection (PCA-style)")
                st.pyplot(fig)
                plt.close()
                sel_idx = st.selectbox("Select point (by index)", range(len(texts)), format_func=lambda i: f"{i}: day={days[i]} {texts[i][:50]}...", key="mri_sel")
                st.write("**Memory text:**", texts[sel_idx])
                st.write("**Day:**", days[sel_idx], "**Type:**", types_list[sel_idx])
            else:
                st.info("Need at least 2 memory items for projection.")
        else:
            st.info("No run records.")
    else:
        st.markdown('<div class="empty-state"><strong>No runs</strong><br/>Run a benchmark to see Memory MRI.</div>', unsafe_allow_html=True)

# ---------- Reports ----------
if view == "Reports":
    if result:
        if st.button("Generate PDF report"):
            try:
                from reportlab.lib.pagesizes import letter
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
                from reportlab.lib.styles import getSampleStyleSheet
                from reportlab.lib import colors
                pdf_path = str(DATA_DIR / f"report_{result.run_id}.pdf")
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
        st.info("Run a benchmark to generate a report.")

# ---------- Legacy ----------
if view == "Legacy":
    tabs_legacy = st.tabs(["Overview", "Forgetting Curve", "Export"])
    with tabs_legacy[0]:
        if result:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{result.accuracy:.2%}")
            c2.metric("Citation Precision", f"{result.citation_precision:.2%}")
            c3.metric("Citation Recall", f"{result.citation_recall:.2%}")
            c4.metric("Memory Items", result.memory_items_stored)
            mv2 = getattr(result, "metrics_v2", None)
            if mv2:
                st.write("**M-Score:**", mv2.get("m_score"), "| **Retention half-life:**", mv2.get("retention_half_life"))
        else:
            st.info("Run a benchmark.")
    with tabs_legacy[1]:
        if result and result.forgetting_curve:
            fig, ax = plt.subplots()
            ax.plot([x[0] for x in result.forgetting_curve], [x[1] for x in result.forgetting_curve], marker="o")
            st.pyplot(fig)
            plt.close()
        else:
            st.info("No forgetting curve.")
    with tabs_legacy[2]:
        if result:
            path = save_result(result, str(DATA_DIR))
            st.success(f"Saved {path}")
            df = result_to_dataframe(result)
            st.download_button("Download CSV", df.to_csv(index=False), file_name=f"run_{result.run_id}.csv", mime="text/csv")
        else:
            st.info("Run a benchmark to export.")
