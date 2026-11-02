"""End-to-end pipeline. Every stage is checkpointed: re-running after a Colab disconnect resumes where it stopped."""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import descriptive as desc
from .analysis import inference as inf
from .analysis import trends, viz
from .checkpoint import Checkpointer, fingerprint
from .config import Paths, config_fingerprint, to_jsonable
from .data.clean import build_medal_table, clean_events, country_tally
from .data.ingest import load_source
from .data.validate import enforce, validate_events, write_quality_report
from .forecasting import medals as fc
from .io_utils import frame_path, get_logger, load_frame, read_json, save_frame, save_table, write_json, write_text
from .ml import train
from .ml.features import build_ml_frame, to_model_frame
from .monitoring import drift
from .ml.registry import latest_version, load_model
from .ml.split import make_splits, split_report
from .sql.layer import build_database, list_queries, local_copy, run_query
from .tracking import Tracker

warnings.filterwarnings("ignore", category=FutureWarning)
log = get_logger()

STAGES = ["ingest", "clean", "sql", "stats", "forecast", "features", "compare", "final", "monitor", "report"]


class Pipeline:
    def __init__(self, cfg: dict, root: str | Path | None = None, demo: bool = False):
        self.cfg, self.demo = cfg, demo
        self.paths = Paths.from_env(root).ensure()
        self.ck = Checkpointer(self.paths.checkpoints)
        self.log = get_logger("olympics", self.paths.logs / "pipeline.log")
        self.tracker = Tracker(self.paths.mlruns)
        self._clean = self._medals = self._regions = self._ml = None

    # ------------------------------------------------------------------ cached loaders
    def _fp(self, *sections: str) -> str:
        return fingerprint(config_fingerprint(self.cfg, list(sections)), self.demo)

    def clean_df(self) -> pd.DataFrame:
        if self._clean is None:
            self._clean = load_frame(self.paths.processed / "clean_events")
        return self._clean

    def medals(self) -> pd.DataFrame:
        if self._medals is None:
            self._medals = load_frame(self.paths.processed / "medals")
        return self._medals

    def regions(self) -> pd.DataFrame:
        return load_frame(self.paths.interim / "regions")

    def ml_frame(self) -> pd.DataFrame:
        if self._ml is None:
            self._ml = load_frame(self.paths.processed / "ml_frame")
        return self._ml

    def _out(self, stem: Path) -> list[Path]:
        p = frame_path(stem)
        return [p] if p else [stem.with_suffix(".parquet")]

    # ------------------------------------------------------------------ stages
    def ingest(self, force=False, force_download=False) -> dict:
        def work():
            ev, reg, info = load_source(self.cfg, self.paths.raw, demo=self.demo, force_download=force_download)
            save_frame(ev, self.paths.interim / "events_raw")
            save_frame(reg, self.paths.interim / "regions")
            write_json(self.paths.metrics / "source_info.json", info)
            return {"rows": len(ev), "source": info["source"]}
        return self.ck.run("ingest", work, self._fp("data"), self._out(self.paths.interim / "events_raw"), force)

    def clean(self, force=False) -> dict:
        def work():
            ev, reg = load_frame(self.paths.interim / "events_raw"), self.regions()
            report = validate_events(ev, reg, self.cfg)
            write_quality_report(report, self.paths.tables, self.paths.metrics)
            enforce(report, self.cfg)
            df, clog = clean_events(ev, reg, self.cfg)
            med = build_medal_table(df)
            save_frame(df, self.paths.processed / "clean_events")
            save_frame(med, self.paths.processed / "medals")
            write_json(self.paths.metrics / "cleaning_log.json", clog)
            save_table(country_tally(med), self.paths.tables / "medal_tally_by_country.csv")
            self._clean = self._medals = None
            return {"rows_out": len(df), "quality_score": report["summary"]["quality_score"], **{k: v for k, v in clog.items() if k.endswith("removed")}}
        return self.ck.run("clean", work, self._fp("data", "validation"), self._out(self.paths.processed / "clean_events"), force)

    def sql(self, force=False) -> dict:
        def work():
            build_database(self.clean_df(), self.medals(), self.regions(), self.paths.db_file)
            local_db = local_copy(self.paths.db_file)
            P = {"season": None, "year_min": 1896, "year_max": 2100}
            specs = {"country_medal_summary": P, "medal_rank_by_games": {"season": None, "top_n": 3},
                     "athlete_career": {"min_games": 2, "limit": 200}, "participation_trend": {"season": None},
                     "sport_profile": {"season": None, "min_entries": 500}, "host_effect": {"season": None}}
            for name, params in specs.items():
                save_table(run_query(local_db, name, params), self.paths.tables / f"sql_{name}.csv")
            return {"queries_exported": len(specs), "available_queries": list_queries()}
        return self.ck.run("sql", work, self._fp("data"), [self.paths.db_file], force)

    def stats(self, force=False) -> dict:
        def work():
            s, df, med = self.cfg["stats"], self.clean_df(), self.medals()
            book, T = viz.FigureBook(self.paths.figures), self.paths.tables
            alpha, summary = s["alpha"], {}
            qa = read_json(self.paths.metrics / "data_quality.json", {})
            if qa.get("missingness"):
                viz.plot_missingness(book, qa["missingness"])
            save_table(desc.describe_numeric(df, ["Age", "Height", "Weight", "BMI"]), T / "descriptive_numeric.csv")
            save_table(desc.describe_numeric(df, ["Age", "Height", "Weight", "BMI"], by="Sex"), T / "descriptive_numeric_by_sex.csv")
            save_table(desc.describe_categorical(df, ["Sex", "Season", "Sport", "region"]), T / "descriptive_categorical.csv")
            save_table(desc.sport_physique_summary(df), T / "sport_physique_summary.csv")

            for season in (["Summer", "Winter"] if "Winter" in set(df["Season"]) else ["Summer"]):
                tr = trends.participation_trends(df, season, alpha)
                save_table(tr["editions"], T / f"participation_by_edition_{season.lower()}.csv")
                write_json(self.paths.metrics / f"trend_tests_{season.lower()}.json", {"tests": tr["tests"], "adf_athletes_p": tr["adf_athletes_p"]})
                if season == "Summer":
                    viz.plot_participation(book, tr["editions"], season)
                    viz.plot_women_share(book, tr["editions"], season)
                    summary["trend_athletes"] = tr["tests"]["athletes"]["trend"]
            viz.plot_country_trends(book, med, "Summer")

            rates = inf.medal_rate_ci(df, "NOC", 500, alpha)
            save_table(rates, T / "medal_rate_ci_by_noc.csv")
            viz.plot_medal_rate_ci(book, rates)

            age = inf.compare_groups(df, "Age", "Sport", s["top_sports"], s["min_group_size"], alpha, seed=s["seed"])
            save_table(age["group_summary"], T / "age_by_sport_summary.csv")
            save_table(age["pairwise"], T / "age_by_sport_pairwise_mannwhitney.csv")
            write_json(self.paths.metrics / "age_across_sports_tests.json", {k: v for k, v in age.items() if not isinstance(v, pd.DataFrame)})
            viz.plot_age_by_sport(book, df, s["top_sports"])
            summary["age_kruskal_p"] = age["kruskal_p"]

            chi = inf.chi_square_independence(df, "Sex", "Sport", top_a=None)
            top = df["Sport"].value_counts().head(20).index
            chi20 = inf.chi_square_independence(df[df["Sport"].isin(top)], "Sex", "Sport")
            save_table(chi20["_std_residuals"].T.round(2).reset_index(), T / "chi_square_sex_sport_residuals.csv")
            viz.plot_chi_residuals(book, chi20["_std_residuals"].T, "Sex x Sport: standardised residuals (top 20 sports)")
            chi_medal = inf.chi_square_independence(df.assign(medal_type=df["Medal"].fillna("None")), "medal_type", "Season")
            write_json(self.paths.metrics / "chi_square_tests.json", [{k: v for k, v in c.items() if not k.startswith("_")} for c in (chi, chi20, chi_medal)])

            mv = inf.medalist_vs_rest_tests(df, alpha=alpha)
            save_table(mv, T / "medalist_vs_rest_tests.csv")
            viz.plot_physique(book, df)
            pv = inf.participation_vs_medals(df, med, alpha)
            save_table(pv["panel"], T / "participation_medals_panel.csv")
            write_json(self.paths.metrics / "participation_vs_medals.json", {k: v for k, v in pv.items() if k != "panel"})
            viz.plot_participation_vs_medals(book, pv["panel"])
            viz.plot_corr_heatmap(book, df, ["Age", "Height", "Weight", "BMI", "Year", "won_medal", "is_host"])
            summary["participation_spearman"] = pv["spearman"]["r"]

            ha = trends.host_advantage(df, med, "Summer", alpha)
            if "table" in ha and len(ha["table"]):
                save_table(ha["table"], T / "host_advantage.csv")
                if "mean_uplift_pp" in ha:
                    viz.plot_host_advantage(book, ha["table"])
                    summary["host_uplift_pp"] = ha["mean_uplift_pp"]
            write_json(self.paths.metrics / "host_advantage.json", {k: v for k, v in ha.items() if k != "table"})
            return to_jsonable(summary)
        return self.ck.run("stats", work, self._fp("data", "stats", "validation"), [self.paths.metrics / "host_advantage.json"], force)

    def forecast(self, force=False) -> dict:
        def work():
            res = fc.run_forecast(self.medals(), self.cfg)
            save_table(res["summary"], self.paths.tables / "forecast_model_comparison.csv")
            save_table(res["backtest"], self.paths.tables / "forecast_backtest_by_edition.csv")
            save_table(res["forecast"], self.paths.tables / "forecast_next_edition.csv")
            write_json(self.paths.metrics / "forecast_summary.json", {"best_model": res["best_model"], "holdout_editions": res["holdout_editions"],
                       "backtest_years": res["backtest_years"], "interval_coverage_in_backtest": res["interval_coverage_in_backtest"],
                       "naive_mae": float(res["summary"].query("model == 'naive_last'")["mae"].iloc[0]), "best_mae": float(res["summary"].iloc[0]["mae"])})
            book = viz.FigureBook(self.paths.figures)
            fc.plot_forecast(book, res)
            fc.plot_model_comparison(book, res["summary"])
            return {"best_model": res["best_model"], "best_mae": float(res["summary"].iloc[0]["mae"])}
        return self.ck.run("forecast", work, self._fp("data", "forecast"), [self.paths.tables / "forecast_next_edition.csv"], force)

    def features(self, force=False) -> dict:
        def work():
            ml = build_ml_frame(self.clean_df())
            save_frame(ml, self.paths.processed / "ml_frame")
            splits = make_splits(ml, self.cfg)
            train.save_splits(self.paths, splits)
            rep = split_report(ml, splits, self.cfg["ml"]["target"], self.cfg["ml"]["split"]["group_cols"])
            save_table(rep, self.paths.tables / "split_report.csv")
            write_json(self.paths.metrics / "split_group_overlap.json", rep.attrs["group_overlap"])
            self._ml = None
            return {"rows": len(ml), "positive_rate": float(ml[self.cfg["ml"]["target"]].mean()), "group_overlap": rep.attrs["group_overlap"]}
        return self.ck.run("features", work, self._fp("data", "ml"), self._out(self.paths.processed / "ml_frame") + [self.paths.processed / "splits.json"], force)

    def compare(self, force=False) -> dict:
        def work():
            ranked = train.compare_models(self.cfg, self.paths, self.ml_frame(), train.load_splits(self.paths), self.tracker, force=force)
            return {"models_ok": int(len(ranked)), "best": ranked.iloc[0]["model"] if len(ranked) else None,
                    "best_pr_auc": float(ranked.iloc[0]["pr_auc"]) if len(ranked) else None}
        return self.ck.run("compare", work, self._fp("data", "ml"), [self.paths.tables / "model_comparison_validation.csv"], force)

    def final(self, force=False) -> dict:
        def work():
            ranked = pd.read_csv(self.paths.tables / "model_comparison_validation.csv")
            splits, ml = train.load_splits(self.paths), self.ml_frame()
            cv = train.cross_validate_top(self.cfg, self.paths, ml, splits, ranked, force=force)
            data_fp = fingerprint(len(ml), float(ml[self.cfg["ml"]["target"]].sum()), self.cfg["data"]["source"], self.demo)
            return train.finalize_model(self.cfg, self.paths, ml, splits, ranked, cv, self.clean_df(), self.tracker, data_fp)
        return self.ck.run("final", work, self._fp("data", "ml"), [self.paths.metrics / "final_model_metrics.json"], force)

    def monitor(self, force=False) -> dict:
        """Demonstrates monitoring: reference = all editions before the last two, current = the last two.

        Drift uses every row (features only, no labels needed). Performance uses ONLY held-out rows
        (validation + test splits): the model was fitted on the training split, so scoring training rows
        would report inflated, meaningless numbers for both windows.
        """
        def work():
            ml, model = self.ml_frame(), load_model(self.paths.models)[0]
            splits = train.load_splits(self.paths)
            eds = ml[["Year", "Season"]].drop_duplicates().sort_values(["Year", "Season"])
            cur_keys = set(eds.tail(2).apply(lambda r: f"{r.Year}|{r.Season}", axis=1))
            key = ml["Year"].astype(str) + "|" + ml["Season"]
            cur_mask = key.isin(cur_keys).to_numpy()
            X = to_model_frame(ml)
            ref, cur = X[~cur_mask].sample(min(50000, int((~cur_mask).sum())), random_state=1), X[cur_mask]
            table = drift.detect_drift(ref, cur, self.cfg)
            save_table(table, self.paths.tables / "drift_report.csv")

            held = np.zeros(len(ml), dtype=bool)
            held[np.concatenate([splits["val"], splits["test"]])] = True
            ref_idx, cur_idx = np.flatnonzero(held & ~cur_mask), np.flatnonzero(held & cur_mask)
            y = ml[self.cfg["ml"]["target"]].to_numpy()
            perf = drift.performance_monitor(y[ref_idx], model.predict_proba(X.iloc[ref_idx]), y[cur_idx], model.predict_proba(X.iloc[cur_idx]),
                                             model.threshold, self.cfg)
            result = {"drift": drift.drift_summary(table),
                      "prediction_drift": drift.prediction_drift(model.predict_proba(ref), model.predict_proba(cur)), "performance": perf,
                      "reference": "all editions before the last two", "current": sorted(cur_keys),
                      "note": "Performance is measured on held-out (validation+test) rows only. 'Year' is excluded from drift checks."}
            write_json(self.paths.metrics / "monitoring_summary.json", result)
            drift.evidently_report(ref, cur, self.paths.metrics / "evidently_drift_report.html")
            return {"overall_drift": result["drift"]["overall"], "performance_status": perf["status"]}
        return self.ck.run("monitor", work, self._fp("data", "ml", "monitoring"), [self.paths.metrics / "monitoring_summary.json"], force)

    def report(self, force=False) -> dict:
        from .reporting import build_report
        return self.ck.run("report", lambda: {"report": str(build_report(self.paths, self.cfg))}, str(time.time()), [], True)

    # ------------------------------------------------------------------ orchestration
    def run(self, stages: list[str] | None = None, force: list[str] | None = None, force_download: bool = False) -> dict:
        stages = stages or STAGES
        force = set(force or [])
        out = {}
        for st in stages:
            fn = getattr(self, st)
            kwargs = {"force": st in force or "all" in force}
            if st == "ingest":
                kwargs["force_download"] = force_download
            out[st] = fn(**kwargs)
        write_json(self.paths.metrics / "run_summary.json", {"finished": time.strftime("%Y-%m-%d %H:%M:%S"), "stages": out, "demo": self.demo})
        return out
