# AI Memory Gym V2 — UI & Feature Enhancement Suggestions

## UI improvements

### 1. **Header & run feedback**
- **Copy run_id to clipboard**: Add a "Copy" button next to run_id that uses `st.code` + JS or a small snippet so judges can paste into reports.
- **Last run summary in header**: Show one-line summary under the pill, e.g. "personal_assistant · full_log · 85% · 1.2k tokens".
- **Cached indicator**: When a run was loaded from cache, show a small "Cached" badge so users know they didn’t re-execute.

### 2. **Experiment Builder (sidebar)**
- **Scenario/policy descriptions**: Use `st.selectbox(..., help="...")` or expanders with 1-line descriptions (e.g. "personal_assistant: preferences + distractors; query later").
- **Preset export/import**: Save presets to a JSON file and load from file (not only session_state) so they survive refresh and can be shared.
- **Quick presets**: Buttons like "Default", "Stress test", "Full suite (minimal)" that apply a known config in one click.
- **Collapsible Advanced**: Put seed, days, wm_size, top_k, LLM in an `st.expander("Advanced")` to reduce clutter.

### 3. **Dashboard**
- **Delta vs baseline**: For metric cards, compute a "baseline" run (e.g. first run or a chosen run) and show delta (e.g. "+2.3%" or "-100 tokens") in `st.metric(..., delta=...)`.
- **Tiny sparklines**: If you have a short time series (e.g. accuracy by day), add a small sparkline in the metric card (e.g. last 7 days) using a minimal inline chart or unicode.
- **Error taxonomy**: Beyond Wrong/Correct pie, add categories (e.g. "Wrong retrieval", "Hallucination", "Contradiction") if you can infer them from run_records/tags.
- **Export chart as PNG**: Add "Download chart" for Pareto and forgetting curve (matplotlib `fig.savefig` to buffer, then `st.download_button`).

### 4. **Leaderboard**
- **Persist leaderboard source**: Load runs from `data/runs/runs/*/manifest.json` (and aggregated suite CSVs) so the leaderboard reflects all saved runs, not only this session.
- **Violin plot option**: In addition to box plot, offer violin (e.g. via matplotlib or plotly) for M-score distribution by policy.
- **Sortable columns**: Use `st.dataframe(..., column_config=...)` or a library that allows click-to-sort on column headers.
- **Highlight current run**: If current result is in the leaderboard, highlight that row.

### 5. **Stress Lab**
- **Stress selector**: Dropdown to pick which stress mode to show in the heatmap (when multiple stress modes exist in data).
- **Before/after with multiple runs**: Compare no-stress vs each stress type (e.g. bar group: none, distraction, contradiction) when you have multiple runs.
- **Tooltips on heatmap cells**: Show exact accuracy (e.g. 0.85) and run count on hover (e.g. with plotly or custom HTML).

### 6. **Traces**
- **Load run from disk**: Run selector that includes runs from `data/runs/runs/*` (read manifest or traces.jsonl) so users can inspect past runs after refresh.
- **Step timeline**: A small horizontal timeline (day/turn) with a draggable cursor for step selection instead of only a slider.
- **Expandable panels**: Put Prompt, Retrieved, Memory updates, Answer in expanders so users can open only what they need.
- **Syntax highlight**: Use `st.code(..., language="text")` for prompt and answer for readability.
- **Diff view**: Optional "Diff vs gold" for the answer (e.g. word-level or sentence-level diff).

### 7. **Memory MRI**
- **Load run from disk**: Same as Traces — allow selecting runs from `data/runs/runs/*`.
- **Shape by type**: Use different markers (e.g. circle vs square) for "retrieved" vs "update" in the 2D plot.
- **Optional UMAP**: If `umap-learn` is installed, offer "Projection: PCA | UMAP" for better separation.
- **Hover tooltip**: On hover (e.g. with plotly), show memory text snippet and day/type without opening the selector.
- **"Which questions used this memory?"**: For a selected point, list question steps that had this memory in `retrieved`.

### 8. **Reports**
- **Report scope**: Option to include "current run only" vs "leaderboard slice" vs "full suite" in the PDF.
- **Recommendations section**: Auto-generate 2–3 bullet recommendations from metrics (e.g. "PII leakage > 0 → consider filtering" or "Low half-life → try rehearsal policy").

### 9. **Global polish**
- **Dark mode**: Use Streamlit’s theme or a toggle and ensure CSS (status pill, cards) has dark variants.
- **Keyboard shortcuts**: Optional short note in sidebar, e.g. "R = Run single" if you add `st.session_state` hooks (Streamlit support is limited but possible with custom components).
- **Empty states**: Every view has a clear CTA (e.g. "Run single" or "Run suite") and a one-line example.
- **Loading state for suite**: During suite run, show a progress bar (e.g. "Run 3/12") and optionally tail of `run.log` in an expander.

---

## New features

### 10. **Suite config in UI**
- **Suite builder**: In sidebar or a modal, let users pick policies (multi-select), scenarios (multi-select), seeds (e.g. "42,43,44" or range), stress modes, then "Run suite" uses that instead of hardcoded lists.
- **Save suite config**: Save/load a named "suite preset" (policies × scenarios × seeds × stress) and run it from the UI.

### 11. **Compare runs**
- **Compare 2–3 runs**: Select 2–3 run_ids and show side-by-side: config, accuracy, M-score, forgetting curve overlay, and a small table of per-step correct/incorrect.
- **Comparison PDF**: "Export comparison report" for the selected runs.

### 12. **Load runs from disk**
- **Run picker**: A single source of truth for "all runs": scan `data/runs/runs/*/manifest.json` (and optionally `data/runs/run_*.json`) and populate a run list. Use this in Traces, Memory MRI, Leaderboard, and Compare.
- **Refresh button**: "Reload runs from disk" to pick up new runs without restarting the app.

### 13. **Export & share**
- **Export dashboard as HTML**: One-click export of current view (e.g. metrics + charts) as a single HTML file for sharing.
- **Share preset link**: Encode preset (or suite config) in query params so "Share" gives a URL that pre-fills the Experiment Builder (e.g. `?scenario=ops&policy=vector_rag&stress=distraction_flood`).

### 14. **Reproducibility & config**
- **Show config hash**: In sidebar or header, display "Config hash: abc123" so users can tie a run to a precise config.
- **Reproduce run**: Button "Re-run with same config" that sets sidebar from selected run’s manifest and triggers Run single.

### 15. **Metrics & alerts**
- **Threshold alerts**: Optional thresholds (e.g. accuracy < 0.7, PII > 0) and show a warning banner on Dashboard when current run violates them.
- **Trend**: If multiple runs with same scenario+policy exist (different seeds), show "Trend: improving / stable / degrading" (e.g. linear regression on accuracy).

### 16. **Live suite progress**
- **Streaming progress**: Run suite in a thread or subprocess and stream progress (e.g. "Running policy=full_log scenario=ops seed=42...") and append to a log area; update progress bar and run list as each run completes.
- **Cancel suite**: Optional "Cancel" button that stops after the current run.

### 17. **Scenario preview**
- **Preview scenario**: Button "Preview scenario" that generates the scenario (steps + ground truth) for current scenario_type/days/seed and shows a read-only table of steps and evaluation questions (no run).

### 18. **API status (when LLM = real)**
- **API status**: When LLM mode is "real", show a small indicator (e.g. "API: OK" or "API: Unreachable") by calling a minimal health endpoint or a single token request.

---

## Quick wins (low effort)

1. Add `help="..."` to key sidebar inputs (scenario, policy, stress).
2. Put Advanced (seed, days, wm, top_k, LLM) in `st.expander("Advanced")`.
3. Leaderboard: load runs from `data/runs` (manifests + suite CSVs) so it’s not session-only.
4. Traces / Memory MRI: run selector that includes runs from `data/runs/runs/*`.
5. Dashboard: add `delta` to metric cards using first run or a selected baseline.
6. Export chart as PNG for Pareto and forgetting curve.
7. Preset export/import to JSON file.

---

## Medium effort

1. Suite builder in UI (multi-select policies, scenarios, seeds, stress).
2. Compare 2–3 runs view (side-by-side + overlay curves).
3. Stress Lab: heatmap with stress-mode filter and tooltips.
4. Memory MRI: shape by type, "which questions used this memory", optional UMAP.
5. Report: recommendations section and scope option.
6. Live suite progress (streaming log + progress bar).

---

## Higher effort

1. Dark mode and theme toggle.
2. Share preset/suite via URL query params.
3. Scenario preview (generate and show steps without running).
4. Error taxonomy (categories beyond correct/wrong).
5. API status indicator for real LLM.

Use this as a backlog; pick by impact and effort for your team.
