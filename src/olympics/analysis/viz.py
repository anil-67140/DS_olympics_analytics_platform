"""Figures with built-in storytelling: each PNG is registered with a title, 'what it shows' and 'why it matters'."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..io_utils import read_json, write_json, write_text  # noqa: E402

PALETTE = ["#2E5EAA", "#E4572E", "#17A398", "#F3A712", "#7D4E95", "#6C757D", "#A44A3F", "#5C9EAD"]
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "axes.prop_cycle": plt.cycler(color=PALETTE),
    "axes.titlesize": 12, "axes.titleweight": "bold", "font.size": 10,
})


class FigureBook:
    """Saves figures and keeps ``figure_index.json`` + ``FIGURES.md`` (caption catalogue) in sync."""

    def __init__(self, out_dir: Path):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "figure_index.json"
        self.index: dict = read_json(self.index_path, {}) or {}

    def save(self, fig, name: str, title: str, what: str, why: str) -> Path:
        path = self.dir / f"{name}.png"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        self.index[name] = {"file": path.name, "title": title, "what_it_shows": what, "why_it_matters": why}
        write_json(self.index_path, self.index)
        self._write_markdown()
        return path

    def _write_markdown(self) -> None:
        lines = ["# Figure catalogue", ""]
        for name, meta in sorted(self.index.items()):
            lines += [f"## {meta['title']}", f"![{meta['title']}]({meta['file']})", "",
                      f"**What it shows:** {meta['what_it_shows']}", "", f"**Why it matters:** {meta['why_it_matters']}", ""]
        write_text(self.dir / "FIGURES.md", "\n".join(lines))


# ---------------------------------------------------------------------------------------------- EDA
def plot_missingness(book: FigureBook, missingness: dict) -> Path:
    m = pd.DataFrame(missingness).T.sort_values("pct", ascending=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(m.index, m["pct"], color=PALETTE[0])
    ax.set_xlabel("% of rows missing")
    ax.set_title("Where the data has gaps")
    return book.save(fig, "01_missingness", "Missing values by column",
                     "Share of rows with no value for each column; Height and Weight are missing most often.",
                     "Models and averages that ignore this gap would be biased toward well-documented athletes/eras, so imputation and missing-indicators are part of the pipeline.")


def plot_participation(book: FigureBook, editions: pd.DataFrame, season: str) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, col, label in zip(axes, ["athletes", "nations", "events"], ["Athletes", "Nations", "Events"]):
        ax.plot(editions["Year"], editions[col], marker="o", ms=3)
        ax.set_title(f"{label} per {season} Games")
    fig.suptitle(f"The {season} Games have grown steadily", fontweight="bold")
    return book.save(fig, "02_participation_growth", f"{season} Games: athletes, nations, events",
                     "Number of athletes, competing nations and medal events at each edition.",
                     "Growth in participation is the denominator behind every medal count: more events means more medals to win, so raw medal totals across eras are not directly comparable.")


def plot_women_share(book: FigureBook, editions: pd.DataFrame, season: str) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(editions["Year"], editions["women_pct"], marker="o", color=PALETTE[1])
    ax.axhline(50, ls="--", color="grey", lw=1)
    ax.set_ylabel("% women among athletes")
    ax.set_title(f"Women's participation, {season} Games")
    return book.save(fig, "03_women_share", "Share of women athletes over time",
                     "Percentage of athletes who are women at each edition; dashed line is parity.",
                     "Tracks progress toward gender parity and flags that pre-1990 data is heavily male, which affects any comparison of physique or age by sex.")


def plot_country_trends(book: FigureBook, medals: pd.DataFrame, season: str, top: int = 8) -> Path:
    m = medals[medals["Season"] == season]
    top_c = m["region"].value_counts().head(top).index
    pivot = m[m["region"].isin(top_c)].groupby(["Year", "region"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for c in pivot.columns:
        ax.plot(pivot.index, pivot[c], marker="o", ms=3, label=c)
    ax.legend(ncol=4, fontsize=8, frameon=False)
    ax.set_ylabel("Medals per edition")
    ax.set_title(f"Medals per edition for the {top} most successful countries ({season})")
    return book.save(fig, "04_country_medal_trends", "Medal trajectories of leading countries",
                     "Medals won per edition by the all-time top countries; team events are counted once.",
                     "Shows dynasties, boycotts (1980/1984), state dissolution and host bumps - context any forecast or ranking must respect.")


def plot_age_by_sport(book: FigureBook, df: pd.DataFrame, top: int = 12) -> Path:
    sports = df["Sport"].value_counts().head(top).index
    data = [df.loc[df["Sport"] == s, "Age"].dropna() for s in sports]
    order = np.argsort([d.median() for d in data])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot([data[i] for i in order], vert=False, tick_labels=[sports[i] for i in order], showfliers=False)
    ax.set_xlabel("Age (years)")
    ax.set_title("Age distribution by sport (top sports by entries)")
    return book.save(fig, "05_age_by_sport", "Age by sport",
                     "Median, quartiles and range of athlete age in the most-contested sports.",
                     "Motivates the ANOVA / Kruskal-Wallis test: sports clearly differ in typical age, so 'age' is only meaningful relative to the sport.")


def plot_physique(book: FigureBook, df: pd.DataFrame) -> Path:
    s = df.dropna(subset=["Height", "Weight"])
    s = s.sample(min(len(s), 40000), random_state=1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True, sharey=True)
    for ax, flag, lab in zip(axes, [0, 1], ["No medal", "Medal"]):
        sub = s[s["won_medal"] == flag]
        hb = ax.hexbin(sub["Height"], sub["Weight"], gridsize=40, cmap="viridis", mincnt=1, bins="log")
        ax.set_title(lab)
        ax.set_xlabel("Height (cm)")
    axes[0].set_ylabel("Weight (kg)")
    fig.colorbar(hb, ax=axes, label="log10(count)")
    return book.save(fig, "06_height_weight_medal", "Physique of medallists vs the rest",
                     "Density of height and weight for athletes who did and did not win a medal.",
                     "Medallists are slightly taller/heavier on average, but the clouds overlap heavily: physique alone cannot predict medals.")


def plot_medal_rate_ci(book: FigureBook, rates: pd.DataFrame, label_col: str = "NOC", top: int = 25) -> Path:
    r = rates.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.errorbar(r["rate"] * 100, r[label_col], xerr=[(r["rate"] - r["ci_low"]) * 100, (r["ci_high"] - r["rate"]) * 100],
                fmt="o", color=PALETTE[0], ecolor="grey", capsize=2)
    ax.set_xlabel("Medals per 100 entries (95% Wilson CI)")
    ax.set_title("Medal rate by country with uncertainty")
    return book.save(fig, "07_medal_rate_ci", "Medal rate with confidence intervals",
                     "Medals per 100 athlete-event entries for the best countries, with 95% Wilson intervals.",
                     "Countries with few entries have wide intervals; overlapping intervals mean the ranking between those countries is not statistically supported.")


def plot_host_advantage(book: FigureBook, table: pd.DataFrame) -> Path:
    t = table.sort_values("home_share_pct")
    fig, ax = plt.subplots(figsize=(7.5, 6))
    y = np.arange(len(t))
    ax.hlines(y, t["other_share_pct"], t["home_share_pct"], color="grey", lw=1)
    ax.scatter(t["other_share_pct"], y, color=PALETTE[5], label="Other Games", zorder=3)
    ax.scatter(t["home_share_pct"], y, color=PALETTE[1], label="Home Games", zorder=3)
    ax.set_yticks(y, t["NOC"])
    ax.set_xlabel("Share of all medals (%)")
    ax.legend(frameon=False)
    ax.set_title("Home advantage: host nations at home vs away")
    return book.save(fig, "08_host_advantage", "Host-nation advantage",
                     "For every host country, share of medals at its home Games (red) vs its other Games (grey).",
                     "A consistent shift right supports a real home advantage, which is why 'is_host' is a model and forecast feature.")


def plot_corr_heatmap(book: FigureBook, df: pd.DataFrame, cols: list[str]) -> Path:
    c = df[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{c.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Spearman correlations between numeric variables")
    return book.save(fig, "09_correlation_heatmap", "Correlation heatmap",
                     "Rank correlation between age, body measures, year and medal outcome.",
                     "Height, weight and BMI are strongly inter-correlated (redundant), while correlation with winning a medal is weak - a signal the problem needs context features such as country history.")


def plot_chi_residuals(book: FigureBook, resid: pd.DataFrame, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(resid.values, cmap="RdBu_r", vmin=-8, vmax=8, aspect="auto")
    ax.set_xticks(range(resid.shape[1]), resid.columns, rotation=45, ha="right")
    ax.set_yticks(range(resid.shape[0]), resid.index)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Standardised residual")
    ax.set_title(title)
    return book.save(fig, "10_chi_square_residuals", "Which combinations drive the association",
                     "Standardised residuals of the chi-square test: red = more entries than independence predicts, blue = fewer.",
                     "A significant chi-square only says 'associated'; the residuals show where (e.g. which sports are male- or female-dominated).")


def plot_participation_vs_medals(book: FigureBook, panel: pd.DataFrame) -> Path:
    p = panel[(panel["athletes"] > 0)]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(p["athletes"], p["medals_won"] + 0.5, s=8, alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Athletes sent (log scale)")
    ax.set_ylabel("Medals won + 0.5 (log scale)")
    ax.set_title("Delegation size vs medals won (country x edition)")
    return book.save(fig, "11_participation_vs_medals", "Participation vs medals",
                     "Each dot is one country at one Games.",
                     "Medals rise with delegation size, but with large scatter: the many dots at the bottom are big delegations that won nothing.")
