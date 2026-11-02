"""Interactive Gradio dashboard.

Local:   OLYMPICS_ROOT=workspace python -m apps.dashboard.app
Colab:   from apps.dashboard.app import build_app; build_app(ROOT).launch(share=True)
"""
from __future__ import annotations

import os

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from olympics.dashboard_logic import DashboardData
from olympics.feedback.store import FeedbackStore

THEME = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")
PLOT_KW = dict(template="plotly_white", margin=dict(l=40, r=20, t=50, b=40))


def _empty(msg: str = "No data for this selection") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(size=16))
    fig.update_layout(**PLOT_KW, xaxis_visible=False, yaxis_visible=False)
    return fig


def _years(a, b) -> tuple[int, int]:
    lo, hi = int(a), int(b)
    return (lo, hi) if lo <= hi else (hi, lo)


def _kpi_md(k: dict) -> str:
    cells = "".join(f"<div style='flex:1;min-width:120px;padding:10px 14px;border-radius:10px;background:#f1f5f9'>"
                    f"<div style='font-size:12px;color:#64748b'>{name}</div><div style='font-size:24px;font-weight:700'>{val:,}</div></div>"
                    for name, val in k.items())
    return f"<div style='display:flex;flex-wrap:wrap;gap:10px'>{cells}</div>"


def build_app(root: str | None = None) -> gr.Blocks:
    root = root or os.environ.get("OLYMPICS_ROOT", "workspace")
    D = DashboardData(root)
    y0, y1 = D.year_range
    predictor = store = None
    try:
        from olympics.serving import Predictor
        predictor = Predictor(D.paths.models)
        store = FeedbackStore(D.paths.feedback)
    except Exception as exc:  # noqa: BLE001 - dashboard still works without a trained model
        print(f"[dashboard] predictor unavailable: {exc}")

    default_top = list(D.medal_table(top=5)["Country"])

    with gr.Blocks(title="Olympics Analytics Platform", theme=THEME) as demo:
        gr.Markdown("# 🏅 Olympics Analytics Platform\nExplore 120+ years of Olympic data, compare countries, forecast medals and score athlete entries.")

        # ---------------------------------------------------------------- Overview
        with gr.Tab("Overview"):
            with gr.Row():
                season = gr.Radio(["Both", *D.seasons], value="Summer" if "Summer" in D.seasons else "Both", label="Season")
                y_from = gr.Slider(y0, y1, value=y0, step=1, label="From year")
                y_to = gr.Slider(y0, y1, value=y1, step=1, label="To year")
                sports = gr.Dropdown(D.sports, multiselect=True, label="Sports (optional)")
            kpi_html = gr.HTML()
            with gr.Row():
                tbl = gr.Dataframe(label="Medal table", interactive=False, wrap=True)
                with gr.Column():
                    part_plot = gr.Plot(label="Participation")
            bar_plot = gr.Plot(label="Top countries")

            def overview(season, y_from, y_to, sports):
                yr = _years(y_from, y_to)
                k = D.kpis(season, yr, sports or None)
                t = D.medal_table(season, yr, sports or None, 25)
                p = D.participation(season, yr, sports or None)
                if len(p):
                    pm = p.melt(id_vars=["Year", "Season"], value_vars=["Athletes", "Nations", "Events"], var_name="Measure", value_name="Count")
                    pf = px.line(pm, x="Year", y="Count", color="Measure", line_dash="Season", markers=True, title="Athletes, nations and events per edition")
                    pf.update_layout(**PLOT_KW)
                else:
                    pf = _empty()
                if len(t):
                    bf = px.bar(t.head(15), x="Country", y=["Gold", "Silver", "Bronze"], title="Top 15 countries by medals",
                                color_discrete_map={"Gold": "#d4af37", "Silver": "#9ca3af", "Bronze": "#b45309"})
                    bf.update_layout(**PLOT_KW, barmode="stack")
                else:
                    bf = _empty()
                return _kpi_md(k), t, pf, bf
            for c in (season, y_from, y_to, sports):
                c.change(overview, [season, y_from, y_to, sports], [kpi_html, tbl, part_plot, bar_plot])
            demo.load(overview, [season, y_from, y_to, sports], [kpi_html, tbl, part_plot, bar_plot])

        # ---------------------------------------------------------------- Compare countries
        with gr.Tab("Compare countries"):
            with gr.Row():
                cs = gr.Dropdown(D.countries, value=default_top[:4], multiselect=True, label="Countries (pick 2-8)")
                cseason = gr.Radio(["Both", *D.seasons], value="Summer" if "Summer" in D.seasons else "Both", label="Season")
                c_from = gr.Slider(y0, y1, value=y0, step=1, label="From year")
                c_to = gr.Slider(y0, y1, value=y1, step=1, label="To year")
            cmp_plot, cmp_total = gr.Plot(), gr.Plot()
            cmp_tbl = gr.Dataframe(interactive=False)

            def compare(cs, cseason, c_from, c_to):
                if not cs:
                    return _empty("Pick at least one country"), _empty("Pick at least one country"), pd.DataFrame()
                d = D.compare_countries(cs, cseason, _years(c_from, c_to))
                if d.empty:
                    return _empty(), _empty(), d
                f = px.line(d, x="Year", y="Total", color="Country", markers=True, title="Medals per edition")
                f.update_layout(**PLOT_KW)
                tot = d.groupby("Country")[["Gold", "Silver", "Bronze"]].sum().reset_index()
                f2 = px.bar(tot, x="Country", y=["Gold", "Silver", "Bronze"], title="Total medals in the selected period",
                            color_discrete_map={"Gold": "#d4af37", "Silver": "#9ca3af", "Bronze": "#b45309"})
                f2.update_layout(**PLOT_KW, barmode="group")
                return f, f2, tot.assign(Total=tot[["Gold", "Silver", "Bronze"]].sum(axis=1)).sort_values("Total", ascending=False)
            for c in (cs, cseason, c_from, c_to):
                c.change(compare, [cs, cseason, c_from, c_to], [cmp_plot, cmp_total, cmp_tbl])
            demo.load(compare, [cs, cseason, c_from, c_to], [cmp_plot, cmp_total, cmp_tbl])

        # ---------------------------------------------------------------- Country drill-down
        with gr.Tab("Country drill-down"):
            country = gr.Dropdown(D.countries, value=default_top[0] if default_top else None, label="Country")
            cd_kpi = gr.HTML()
            with gr.Row():
                cd_trend, cd_sport = gr.Plot(), gr.Plot()
            cd_ath = gr.Dataframe(label="Most decorated athletes", interactive=False)

            def drill(country):
                if not country:
                    return "", _empty(), _empty(), pd.DataFrame()
                pr = D.country_profile(country)
                f1 = px.area(pr["trend"], x="Year", y="Total", title=f"{country}: medals per edition")
                f1.update_layout(**PLOT_KW)
                f2 = px.bar(pr["by_sport"].head(12), x="Total", y="Sport", orientation="h", title="Best sports")
                f2.update_layout(**PLOT_KW, yaxis=dict(autorange="reversed"))
                return _kpi_md(pr["kpis"]), f1, f2, pr["top_athletes"]
            country.change(drill, country, [cd_kpi, cd_trend, cd_sport, cd_ath])
            demo.load(drill, country, [cd_kpi, cd_trend, cd_sport, cd_ath])

        # ---------------------------------------------------------------- Sport drill-down
        with gr.Tab("Sport drill-down"):
            sport = gr.Dropdown(D.sports, value="Athletics" if "Athletics" in D.sports else D.sports[0], label="Sport")
            sp_kpi = gr.HTML()
            with gr.Row():
                sp_age, sp_cty = gr.Plot(), gr.Plot()
            sp_events = gr.Dataframe(label="Events", interactive=False)

            def drill_sport(sport):
                if not sport:
                    return "", _empty(), _empty(), pd.DataFrame()
                s = D.sport_drilldown(sport)
                f1 = px.histogram(s["ages"], x="Age", color="Sex", barmode="overlay", opacity=0.6, nbins=30, title=f"{sport}: age distribution by sex")
                f1.update_layout(**PLOT_KW)
                f2 = px.bar(s["top_countries"], x="Total", y="Country", orientation="h", title="Most successful countries")
                f2.update_layout(**PLOT_KW, yaxis=dict(autorange="reversed"))
                return _kpi_md(s["kpis"]), f1, f2, s["events"]
            sport.change(drill_sport, sport, [sp_kpi, sp_age, sp_cty, sp_events])
            demo.load(drill_sport, sport, [sp_kpi, sp_age, sp_cty, sp_events])

        # ---------------------------------------------------------------- Athletes
        with gr.Tab("Athlete explorer"):
            q = gr.Textbox(label="Search athlete name", placeholder="e.g. Phelps")
            gr.Dataframe(D.top_athletes(top=20), label="Most decorated athletes of all time", interactive=False)
            res = gr.Dataframe(label="Results", interactive=False)
            q.submit(D.athlete_career, q, res)

        # ---------------------------------------------------------------- Forecast
        with gr.Tab("Forecast"):
            fc_tbl = D.table("forecast_next_edition.csv")
            gr.Markdown("Next-edition medal forecast from the best rolling-backtest model, with a 90% prediction interval. "
                        "Validated with a rolling-origin backtest (no future leakage).")
            if "forecast" in fc_tbl.columns:
                top = fc_tbl.head(20)
                f = go.Figure()
                f.add_trace(go.Bar(x=top["country"], y=top["forecast"], name="Forecast",
                                   error_y=dict(type="data", symmetric=False, array=top["upper"] - top["forecast"], arrayminus=top["forecast"] - top["lower"])))
                f.add_trace(go.Scatter(x=top["country"], y=top["last_actual"], mode="markers", name="Last edition", marker=dict(size=9, color="#E4572E")))
                f.update_layout(**PLOT_KW, title="Forecast vs last edition")
                gr.Plot(f)
            gr.Dataframe(fc_tbl, interactive=False, label="Forecast table")
            gr.Dataframe(D.table("forecast_model_comparison.csv"), interactive=False, label="Backtest: which method wins?")

        # ---------------------------------------------------------------- Predictor
        with gr.Tab("Medal predictor"):
            if predictor is None:
                gr.Markdown("⚠️ No trained model found. Run the pipeline (`final` stage) first.")
            else:
                gr.Markdown(f"Model **{predictor.version}** ({predictor.meta.get('algorithm')}). Probability that this entry wins a medal, with the reasons.")
                with gr.Row():
                    p_season = gr.Radio(["Summer", "Winter"], value="Summer", label="Season")
                    p_year = gr.Number(value=2016, precision=0, label="Year")
                    p_sex = gr.Radio(["M", "F"], value="M", label="Sex")
                with gr.Row():
                    p_noc = gr.Dropdown(predictor.nocs, value="USA" if "USA" in predictor.nocs else predictor.nocs[0], label="Country (NOC)")
                    p_sport = gr.Dropdown(predictor.sports(), value="Swimming", label="Sport")
                with gr.Row():
                    p_age, p_h, p_w = gr.Number(24, label="Age"), gr.Number(180, label="Height (cm)"), gr.Number(75, label="Weight (kg)")
                with gr.Row():
                    p_ent = gr.Number(40, precision=0, label="Athletes in event"); p_team = gr.Number(1, precision=0, label="Athletes from this country in event")
                    p_pg = gr.Number(0, precision=0, label="Previous Games"); p_pm = gr.Number(0, precision=0, label="Previous medals")
                go_btn = gr.Button("Predict", variant="primary")
                out_md, out_plot, pid = gr.Markdown(), gr.Plot(), gr.State("")
                with gr.Row():
                    actual = gr.Radio(["Won a medal", "No medal"], label="Human feedback: what actually happened?")
                    fb_btn = gr.Button("Submit feedback")
                fb_msg = gr.Markdown()

                def run_predict(season, year, sex, noc, sport, age, h, w, ent, team, pg, pm):
                    req = {"Season": season, "Year": int(year), "Sex": sex, "NOC": noc, "Sport": sport, "Age": age, "Height": h, "Weight": w,
                           "event_entrants": int(ent), "noc_event_entrants": int(team), "athlete_prev_games": int(pg), "athlete_prev_medals": int(pm)}
                    r = predictor.predict([req], explain=True)[0]
                    new_id = store.log_prediction(r["features_used"], r["probability"], int(r["predicted_medal"]), r["model_version"], r["threshold"])
                    ex = pd.DataFrame(r["explanation"])
                    f = px.bar(ex.iloc[::-1], x="effect_on_probability", y="feature", orientation="h", title="What pushed the probability up or down (vs a typical entry)",
                               color=ex.iloc[::-1]["effect_on_probability"] > 0, color_discrete_map={True: "#2E5EAA", False: "#E4572E"})
                    f.update_layout(**PLOT_KW, showlegend=False)
                    verdict = "✅ **Likely medal**" if r["predicted_medal"] else "❌ **Unlikely medal**"
                    return (f"### {verdict}\nProbability **{100 * r['probability']:.1f}%** (decision threshold {100 * r['threshold']:.1f}%).  \nPrediction id `{new_id}`", f, new_id)
                go_btn.click(run_predict, [p_season, p_year, p_sex, p_noc, p_sport, p_age, p_h, p_w, p_ent, p_team, p_pg, p_pm], [out_md, out_plot, pid])

                def send_fb(pred_id, choice):
                    if not pred_id or not choice:
                        return "Make a prediction and pick the actual outcome first."
                    store.log_feedback(pred_id, choice == "Won a medal", "dashboard")
                    return f"Thanks — recorded. {store.summary()}"
                fb_btn.click(send_fb, [pid, actual], fb_msg)

        # ---------------------------------------------------------------- Model report
        with gr.Tab("Model report"):
            fm = D.metrics("final_model_metrics.json")
            if fm:
                t = fm["test_metrics"]
                gr.Markdown(f"**Champion:** {fm['algorithm']} · version `{fm['version']}` · calibration `{fm['calibration']}` · threshold {fm['threshold']:.3f}")
                gr.Dataframe(pd.DataFrame([{k: round(v, 4) for k, v in t.items() if isinstance(v, float)}]).T.reset_index().rename(columns={"index": "metric", 0: "test value"}), interactive=False, label="All test metrics")
            gr.Dataframe(D.table("model_comparison_validation.csv"), interactive=False, label="All models compared (validation)")
            gr.Dataframe(D.table("cross_validation_top_models.csv"), interactive=False, label="Grouped cross-validation of the top models")
            gr.Dataframe(D.table("slice_metrics.csv"), interactive=False, label="Performance by slice")
            figs = D.figures()
            if figs:
                gr.Gallery(figs, label="Figures (what each shows and why it matters)", columns=2, height=700)

        # ---------------------------------------------------------------- Statistics
        with gr.Tab("Statistics"):
            gr.Markdown("Hypothesis tests, effect sizes and confidence intervals behind the conclusions.")
            gr.Dataframe(D.table("medalist_vs_rest_tests.csv"), interactive=False, label="Medallists vs non-medallists (Welch t, Cohen's d, Holm-adjusted p)")
            gr.Dataframe(D.table("age_by_sport_summary.csv"), interactive=False, label="Age by sport")
            gr.Dataframe(D.table("age_by_sport_pairwise_mannwhitney.csv").head(50), interactive=False, label="Pairwise Mann-Whitney (Holm)")
            gr.Dataframe(D.table("medal_rate_ci_by_noc.csv").head(40), interactive=False, label="Medal rate with Wilson CI")
            gr.JSON(D.metrics("age_across_sports_tests.json"), label="ANOVA / Kruskal-Wallis")
            gr.JSON(D.metrics("host_advantage.json"), label="Host advantage")

        # ---------------------------------------------------------------- Monitoring
        with gr.Tab("Monitoring & data quality"):
            gr.Dataframe(D.table("data_quality_checks.csv"), interactive=False, label="Data quality checks")
            gr.Dataframe(D.table("drift_report.csv"), interactive=False, label="Feature drift (PSI / KS)")
            gr.JSON(D.metrics("monitoring_summary.json"), label="Drift + performance summary")
            if store is not None:
                gr.JSON(store.summary(), label="Human feedback so far")

        # ---------------------------------------------------------------- SQL
        with gr.Tab("SQL console"):
            gr.Markdown("Read-only SQL on tables `athlete_events`, `medals`, `regions`.")
            sql_in = gr.Textbox(lines=6, label="SQL", value="SELECT region AS country, SUM(Gold) AS gold, COUNT(*) AS total\nFROM medals\nGROUP BY region\nORDER BY gold DESC\nLIMIT 10")
            sql_btn, sql_out, sql_err = gr.Button("Run"), gr.Dataframe(interactive=False), gr.Markdown()

            def run_sql(query):
                try:
                    return D.sql(query), ""
                except Exception as exc:  # noqa: BLE001
                    return pd.DataFrame(), f"⚠️ {exc}"
            sql_btn.click(run_sql, sql_in, [sql_out, sql_err])
    return demo


if __name__ == "__main__":
    build_app().launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
