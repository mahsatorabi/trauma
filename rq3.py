"""
RQ3 — Prevalence and internal composition of themes across historical periods.

Research question:
  How have the prevalence and internal composition of these themes shifted over time,
  and which themes have risen, declined, or remained stable across distinct historical periods?

Methodological framework:
  1. Four historical periods (pre-pandemic → recent)
  2. Period-level theme prevalence (share of non-outlier corpus per BERTopic theme)
  3. Mann–Kendall + Theil–Sen on annual share series (substantive topics)
  4. Two-proportion z-tests (first vs last period) with Benjamini–Hochberg FDR
  5. Internal composition: within-topic token profiles per period; Jensen–Shannon divergence
  6. Entering/leaving terms via period-pair log-odds (within-topic vocabulary)
  7. Meta-theme aggregation (suicidality / self-harm subtopics from RQ1)
"""

from __future__ import annotations

import ast
import json
import re
import time
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

warnings.filterwarnings("ignore", category=FutureWarning)

from rq1 import (
    filter_english_corpus,
    mann_kendall_trend,
    substantive_topic_ids,
    theil_sen_slope,
    yearly_topic_shares,
)

BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "data_preprocessed.csv"
RQ1_DIR = BASE_DIR / "rq1_output"
OUT_DIR = BASE_DIR / "rq3_output"
OUT_DIR.mkdir(exist_ok=True)

# Four interpretable historical windows (childhood-trauma corpus 2019–2026)
HISTORICAL_PERIODS: dict[str, tuple[int, int]] = {
    "P1_pre_pandemic": (2019, 2020),
    "P2_pandemic_transition": (2021, 2022),
    "P3_post_pandemic": (2023, 2024),
    "P4_recent": (2025, 2026),
}

PERIOD_ORDER = list(HISTORICAL_PERIODS.keys())
FIRST_PERIOD = PERIOD_ORDER[0]
LAST_PERIOD = PERIOD_ORDER[-1]
MIN_DOCS_COMPOSITION = 15  # min docs in a period to assess internal composition
TOP_COMPOSITION_TERMS = 12
RANDOM_SEED = 42


def load_corpus_rq3() -> pd.DataFrame:
    usecols = [
        "UT (Unique WOS ID)",
        "Article Title",
        "Abstract",
        "preprocessed_text",
        "Publication Year",
        "Language",
        "Keywords Plus",
        "Author Keywords",
    ]
    df = pd.read_csv(INPUT_CSV, usecols=usecols, low_memory=False)
    df = df.dropna(subset=["preprocessed_text", "Publication Year"])
    df = df[df["preprocessed_text"].str.strip().astype(bool)]
    df["Publication Year"] = df["Publication Year"].astype(int)
    df = df[(df["Publication Year"] >= 2019) & (df["Publication Year"] <= 2026)]
    return df.reset_index(drop=True)


def assign_period(year: int) -> str | None:
    for label, (y0, y1) in HISTORICAL_PERIODS.items():
        if y0 <= year <= y1:
            return label
    return None


def load_merged_frame() -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    df_raw = load_corpus_rq3()
    df, _ = filter_english_corpus(df_raw)

    doc_path = RQ1_DIR / "rq1_document_topics.csv"
    qa_path = RQ1_DIR / "rq1_topic_qa.csv"
    if not doc_path.exists():
        raise FileNotFoundError("Run `python rq1.py` first to generate rq1_document_topics.csv")

    doc_topics = pd.read_csv(doc_path)
    qa = pd.read_csv(qa_path)
    df = df.merge(
        doc_topics[["UT (Unique WOS ID)", "topic", "topic_type", "display_name"]],
        on="UT (Unique WOS ID)",
        how="inner",
    )
    df["period"] = df["Publication Year"].apply(assign_period)
    df = df.dropna(subset=["period"])

    summary_path = RQ1_DIR / "rq1_summary.json"
    suicide_ids: list[int] = []
    if summary_path.exists():
        suicide_ids = json.loads(summary_path.read_text(encoding="utf-8")).get(
            "suicide_subtopic_ids", []
        )
    return df, qa, suicide_ids


def period_corpus_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, (y0, y1) in HISTORICAL_PERIODS.items():
        sub = df[df["period"] == label]
        valid = sub[sub["topic"] != -1]
        rows.append(
            {
                "period": label,
                "year_range": f"{y0}–{y1}",
                "n_documents": len(sub),
                "n_non_outlier": len(valid),
                "outlier_rate": 1.0 - len(valid) / len(sub) if len(sub) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def period_topic_prevalence(df: pd.DataFrame, topic_ids: list[int]) -> pd.DataFrame:
    """Share of each substantive topic within non-outlier documents, by period."""
    valid = df[(df["topic"] != -1) & (df["topic"].isin(topic_ids))]
    rows = []
    for period in PERIOD_ORDER:
        sub = valid[valid["period"] == period]
        total = len(sub)
        if total == 0:
            continue
        counts = sub.groupby("topic").size()
        for topic_id, count in counts.items():
            rows.append(
                {
                    "period": period,
                    "topic": int(topic_id),
                    "count": int(count),
                    "period_total": total,
                    "share": count / total,
                }
            )
    prev = pd.DataFrame(rows)
    # attach display names
    names = df.drop_duplicates("topic").set_index("topic")["display_name"]
    prev["display_name"] = prev["topic"].map(names)
    return prev


def two_proportion_ztest(count_a: int, n_a: int, count_b: int, n_b: int) -> tuple[float, float]:
    if n_a == 0 or n_b == 0:
        return np.nan, np.nan
    p1, p2 = count_a / n_a, count_b / n_b
    p_pool = (count_a + count_b) / (n_a + n_b)
    if p_pool in (0.0, 1.0):
        return np.nan, np.nan
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    if se == 0:
        return np.nan, np.nan
    z = (p1 - p2) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(z), float(p)


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    adj = np.empty(n)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        adj[i] = prev
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    return out


def prevalence_shift_table(
    period_prev: pd.DataFrame,
    yearly_shares: pd.DataFrame,
    qa: pd.DataFrame,
) -> pd.DataFrame:
    """Classify themes: rising / declining / stable using MK + period contrast."""
    qa_map = qa.set_index("topic")
    first = period_prev[period_prev["period"] == FIRST_PERIOD].set_index("topic")
    last = period_prev[period_prev["period"] == LAST_PERIOD].set_index("topic")

    rows = []
    pvals: list[float] = []
    pidx: list[int] = []

    for topic_id in sorted(period_prev["topic"].unique()):
        if topic_id not in qa_map.index or not qa_map.loc[topic_id, "include_in_trends"]:
            continue

        ys = yearly_shares[yearly_shares["topic"] == topic_id].sort_values("Publication Year")
        y = ys["share"].values
        mk = mann_kendall_trend(y)

        fa = first.loc[topic_id] if topic_id in first.index else None
        la = last.loc[topic_id] if topic_id in last.index else None
        share_first = float(fa["share"]) if fa is not None else 0.0
        share_last = float(la["share"]) if la is not None else 0.0
        cnt_f = int(fa["count"]) if fa is not None else 0
        cnt_l = int(la["count"]) if la is not None else 0
        n_f = int(fa["period_total"]) if fa is not None else 1
        n_l = int(la["period_total"]) if la is not None else 1

        z, p = two_proportion_ztest(cnt_f, n_f, cnt_l, n_l)
        rel_change = (share_last - share_first) / share_first if share_first > 0 else np.nan

        # period trajectory (all four)
        period_shares = {
            p: float(period_prev[(period_prev["topic"] == topic_id) & (period_prev["period"] == p)]["share"].iloc[0])
            if len(period_prev[(period_prev["topic"] == topic_id) & (period_prev["period"] == p)])
            else np.nan
            for p in PERIOD_ORDER
        }

        rows.append(
            {
                "topic": int(topic_id),
                "display_name": qa_map.loc[topic_id, "display_name"],
                "share_P1": share_first,
                "share_P4": share_last,
                "absolute_change_P1_to_P4": share_last - share_first,
                "relative_change_P1_to_P4": rel_change,
                "count_P1": cnt_f,
                "count_P4": cnt_l,
                "z_stat_P1_vs_P4": z,
                "p_value_P1_vs_P4": p,
                "kendall_tau": mk["tau"],
                "kendall_p": mk["p_value"],
                "mk_trend": mk["trend"],
                "theil_sen_slope_per_year": theil_sen_slope(y) if len(y) >= 4 else np.nan,
                **{f"share_{p}": period_shares[p] for p in PERIOD_ORDER},
            }
        )
        if not np.isnan(p):
            pvals.append(p)
            pidx.append(len(rows) - 1)

    out = pd.DataFrame(rows)
    if pvals:
        adj = benjamini_hochberg(np.array(pvals))
        out["p_adj_P1_vs_P4"] = np.nan
        for i, adj_p in zip(pidx, adj):
            out.loc[i, "p_adj_P1_vs_P4"] = adj_p

    out["prevalence_class"] = out.apply(_classify_prevalence, axis=1)
    return out.sort_values("kendall_tau", ascending=False, na_position="last")


def _classify_prevalence(row: pd.Series) -> str:
    mk = row.get("mk_trend", "stable")
    p_adj = row.get("p_adj_P1_vs_P4", np.nan)
    delta = row.get("absolute_change_P1_to_P4", 0.0)

    sig_period = not np.isnan(p_adj) and p_adj < 0.05
    if mk == "increasing" or (sig_period and delta > 0.001):
        return "rising"
    if mk == "decreasing" or (sig_period and delta < -0.001):
        return "declining"
    return "stable"


def topic_token_profile(texts: pd.Series, top_n: int = 50) -> dict[str, float]:
    counter: Counter = Counter()
    for text in texts:
        if isinstance(text, str):
            counter.update(text.split())
    total = sum(counter.values()) or 1
    top = counter.most_common(top_n)
    return {t: c / total for t, c in top}


def jensen_shannon_profiles(p: dict[str, float], q: dict[str, float]) -> float:
    vocab = sorted(set(p) | set(q))
    if not vocab:
        return 0.0
    a = np.array([p.get(t, 0.0) for t in vocab], dtype=float)
    b = np.array([q.get(t, 0.0) for t in vocab], dtype=float)
    a = a / (a.sum() or 1.0)
    b = b / (b.sum() or 1.0)
    return float(jensenshannon(a, b, base=2))


def within_topic_period_log_odds(
    df: pd.DataFrame,
    topic_id: int,
    period_a: str,
    period_b: str,
    top_n: int = 15,
) -> pd.DataFrame:
    sub = df[(df["topic"] == topic_id)]
    ta = sub[sub["period"] == period_a]["preprocessed_text"]
    tb = sub[sub["period"] == period_b]["preprocessed_text"]
    if len(ta) < MIN_DOCS_COMPOSITION or len(tb) < MIN_DOCS_COMPOSITION:
        return pd.DataFrame()

    ca, cb = Counter(), Counter()
    for t in ta:
        ca.update(str(t).split())
    for t in tb:
        cb.update(str(t).split())

    terms = set(ca) | set(cb)
    na, nb = sum(ca.values()), sum(cb.values())
    rows = []
    for term in terms:
        a, b = ca.get(term, 0), cb.get(term, 0)
        lo = np.log((b + 0.5) / (nb - b + 0.5)) - np.log((a + 0.5) / (na - a + 0.5))
        rows.append({"term": term, "log_odds_B_vs_A": lo, "count_a": a, "count_b": b})
    out = pd.DataFrame(rows).sort_values("log_odds_B_vs_A", ascending=False)
    return pd.concat([out.head(top_n), out.tail(top_n)]).drop_duplicates("term")


def composition_shift_analysis(df: pd.DataFrame, topic_ids: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """JS divergence P1 vs P4 + entering/leaving terms per topic."""
    summary_rows: list[dict] = []
    term_rows: list[dict] = []

    for topic_id in topic_ids:
        sub = df[df["topic"] == topic_id]
        p1_texts = sub[sub["period"] == FIRST_PERIOD]["preprocessed_text"]
        p4_texts = sub[sub["period"] == LAST_PERIOD]["preprocessed_text"]
        if len(p1_texts) < MIN_DOCS_COMPOSITION or len(p4_texts) < MIN_DOCS_COMPOSITION:
            continue

        prof1 = topic_token_profile(p1_texts)
        prof4 = topic_token_profile(p4_texts)
        js = jensen_shannon_profiles(prof1, prof4)

        lo_df = within_topic_period_log_odds(df, topic_id, FIRST_PERIOD, LAST_PERIOD)
        entering = lo_df.nlargest(5, "log_odds_B_vs_A")["term"].tolist() if not lo_df.empty else []
        leaving = lo_df.nsmallest(5, "log_odds_B_vs_A")["term"].tolist() if not lo_df.empty else []

        display = sub["display_name"].iloc[0]
        summary_rows.append(
            {
                "topic": topic_id,
                "display_name": display,
                "n_docs_P1": len(p1_texts),
                "n_docs_P4": len(p4_texts),
                "js_divergence_P1_vs_P4": js,
                "entering_terms_P4": ", ".join(entering),
                "leaving_terms_P4": ", ".join(leaving),
            }
        )
        if not lo_df.empty:
            for _, r in lo_df.iterrows():
                term_rows.append(
                    {
                        "topic": topic_id,
                        "display_name": display,
                        "term": r["term"],
                        "log_odds_P4_vs_P1": r["log_odds_B_vs_A"],
                        "count_P1": r["count_a"],
                        "count_P4": r["count_b"],
                    }
                )

    comp = pd.DataFrame(summary_rows)
    if not comp.empty:
        med = comp["js_divergence_P1_vs_P4"].median()
        comp["composition_class"] = np.where(
            comp["js_divergence_P1_vs_P4"] >= med, "recomposed", "composition_stable"
        )
    terms = pd.DataFrame(term_rows)
    return comp.sort_values("js_divergence_P1_vs_P4", ascending=False), terms


def bertopic_words_over_periods() -> pd.DataFrame:
    """Track BERTopic c-TF-IDF words per year from rq1_topics_over_time."""
    path = RQ1_DIR / "rq1_topics_over_time.csv"
    if not path.exists():
        return pd.DataFrame()

    tot = pd.read_csv(path)
    tot = tot[tot["Topic"] >= 0].copy()
    tot["period"] = tot["Timestamp"].apply(assign_period)
    tot = tot.dropna(subset=["period"])

    rows = []
    for (topic, period), grp in tot.groupby(["Topic", "period"]):
        # use latest year in period as representative word snapshot
        latest = grp.sort_values("Timestamp").iloc[-1]
        words_raw = latest["Words"]
        if isinstance(words_raw, str) and words_raw.startswith("["):
            words = ast.literal_eval(words_raw)
        else:
            words = [w.strip() for w in str(words_raw).split(",")]
        rows.append(
            {
                "topic": int(topic),
                "period": period,
                "representative_words": ", ".join(words[:8]),
                "mean_frequency": float(grp["Frequency"].mean()),
            }
        )
    return pd.DataFrame(rows)


def meta_theme_by_period(df: pd.DataFrame, suicide_ids: list[int]) -> pd.DataFrame:
    rows = []
    for period in PERIOD_ORDER:
        sub = df[df["period"] == period]
        valid = sub[sub["topic"] != -1]
        total = len(valid)
        count = int(valid["topic"].isin(suicide_ids).sum())
        rows.append(
            {
                "period": period,
                "meta_theme": "suicidality_self_harm",
                "count": count,
                "period_total": total,
                "share": count / total if total else 0.0,
                "source_topics": ",".join(map(str, suicide_ids)),
            }
        )
    return pd.DataFrame(rows)


def plot_period_heatmap(period_prev: pd.DataFrame, shift: pd.DataFrame, top_n: int = 20) -> None:
    top_topics = (
        shift.groupby("topic")["share_P4"]
        .max()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )
    sub = period_prev[period_prev["topic"].isin(top_topics)]
    pivot = sub.pivot_table(index="topic", columns="period", values="share", aggfunc="mean")
    pivot = pivot.reindex(columns=PERIOD_ORDER)
    pivot = pivot.loc[top_topics]

    name_map = shift.set_index("topic")["display_name"]
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(PERIOD_ORDER)))
    ax.set_xticklabels([p.replace("_", "\n") for p in PERIOD_ORDER], fontsize=7)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(
        [f"T{int(t)}: {str(name_map.get(t, t))[:38]}" for t in pivot.index],
        fontsize=7,
    )
    ax.set_title("RQ3 — Theme prevalence across historical periods (top themes by P4 share)")
    plt.colorbar(im, ax=ax, label="Share of non-outlier corpus")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "rq3_period_prevalence_heatmap.png", dpi=200)
    plt.close(fig)


def plot_trajectory_lines(period_prev: pd.DataFrame, shift: pd.DataFrame) -> None:
    for cls, color, fname in [
        ("rising", "darkgreen", "rq3_rising_themes.png"),
        ("declining", "firebrick", "rq3_declining_themes.png"),
        ("stable", "steelblue", "rq3_stable_themes_sample.png"),
    ]:
        ids = shift[shift["prevalence_class"] == cls].head(6)["topic"].tolist()
        if not ids:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        for tid in ids:
            s = period_prev[period_prev["topic"] == tid].set_index("period").reindex(PERIOD_ORDER)
            name = shift.loc[shift["topic"] == tid, "display_name"].iloc[0]
            ax.plot(PERIOD_ORDER, s["share"].values, marker="o", label=f"T{tid}: {name[:35]}")
        ax.set_ylabel("Corpus share")
        ax.set_title(f"RQ3 — {cls.capitalize()} themes across periods")
        ax.legend(fontsize=6, loc="best")
        plt.xticks(rotation=15, ha="right", fontsize=7)
        plt.tight_layout()
        fig.savefig(OUT_DIR / fname, dpi=200)
        plt.close(fig)


def plot_meta_theme(meta_period: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(meta_period))
    ax.bar(x, meta_period["share"], color="purple", alpha=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(meta_period["period"], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Share")
    ax.set_title("RQ3 — Suicidality & self-harm meta-theme by period")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "rq3_meta_theme_periods.png", dpi=200)
    plt.close(fig)


def write_report(
    corpus_stats: pd.DataFrame,
    shift: pd.DataFrame,
    comp: pd.DataFrame,
    bertopic_evolution: pd.DataFrame,
    meta_period: pd.DataFrame,
    runtime: float,
) -> None:
    rising = shift[shift["prevalence_class"] == "rising"]
    declining = shift[shift["prevalence_class"] == "declining"]
    stable = shift[shift["prevalence_class"] == "stable"]
    recomposed = comp[comp["composition_class"] == "recomposed"] if not comp.empty else comp

    lines = [
        "=" * 78,
        "RQ3 REPORT — THEME PREVALENCE & COMPOSITION ACROSS HISTORICAL PERIODS",
        "=" * 78,
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Runtime: {runtime:.1f} s",
        "",
        "RESEARCH QUESTION",
        "  How have the prevalence and internal composition of these themes shifted over time,",
        "  and which themes have risen, declined, or remained stable across distinct historical",
        "  periods?",
        "",
        "HISTORICAL PERIODS",
    ]
    for _, r in corpus_stats.iterrows():
        lines.append(
            f"  {r['period']:28s} ({r['year_range']}) | n={int(r['n_documents']):,} "
            f"| non-outlier={int(r['n_non_outlier']):,}"
        )

    lines.extend(
        [
            "",
            "METHOD",
            "  1. BERTopic substantive themes (RQ1; methods/language artifacts excluded)",
            "  2. Period prevalence = topic count / non-outlier documents per period",
            "  3. Mann–Kendall + Theil–Sen on annual share series",
            "  4. Two-proportion z-test (P1 vs P4) with Benjamini–Hochberg FDR",
            "  5. Prevalence class: rising | declining | stable (MK + period contrast)",
            "  6. Internal composition: within-topic token Jensen–Shannon (P1 vs P4)",
            "  7. Entering/leaving terms: log-odds within-topic vocabulary",
            "",
            "CORPUS SHIFT ACROSS PERIODS",
        ]
    )
    v0 = int(corpus_stats.iloc[0]["n_documents"])
    v3 = int(corpus_stats.iloc[-1]["n_documents"])
    lines.append(f"  Total publications: {v0:,} (P1) → {v3:,} (P4)")

    lines.extend(
        [
            "",
            "—" * 40,
            "A. PREVALENCE SHIFTS",
            "—" * 40,
            "",
            f"A1. RISING THEMES (n={len(rising)}) — Mann–Kendall ↑ and/or significant P1→P4 increase",
        ]
    )
    for _, r in rising.head(15).iterrows():
        lines.append(
            f"  T{int(r['topic']):3d} | P1={r['share_P1']:.3f} → P4={r['share_P4']:.3f} "
            f"| Δ={r['absolute_change_P1_to_P4']:+.3f} | τ={r['kendall_tau']:.3f} | {r['display_name']}"
        )

    lines.extend(["", f"A2. DECLINING THEMES (n={len(declining)})"])
    for _, r in declining.head(12).iterrows():
        lines.append(
            f"  T{int(r['topic']):3d} | P1={r['share_P1']:.3f} → P4={r['share_P4']:.3f} "
            f"| Δ={r['absolute_change_P1_to_P4']:+.3f} | τ={r['kendall_tau']:.3f} | {r['display_name']}"
        )

    lines.extend(["", f"A3. STABLE THEMES (n={len(stable)}) — sample (largest P4 share)"])
    for _, r in stable.nlargest(10, "share_P4").iterrows():
        lines.append(
            f"  T{int(r['topic']):3d} | P1={r['share_P1']:.3f} → P4={r['share_P4']:.3f} | {r['display_name']}"
        )

    if not meta_period.empty:
        lines.extend(["", "A4. META-THEME: SUICIDALITY & SELF-HARM (merged RQ1 subtopics)"])
        for _, r in meta_period.iterrows():
            lines.append(f"  {r['period']:28s} share={r['share']:.3f} (n={int(r['count'])})")

    lines.extend(
        [
            "",
            "—" * 40,
            "B. INTERNAL COMPOSITION SHIFTS (P1 vs P4)",
            "—" * 40,
            "",
            f"B1. RECOMPOSED THEMES (JS divergence ≥ median; n={len(recomposed)})",
        ]
    )
    for _, r in recomposed.head(12).iterrows():
        lines.append(
            f"  T{int(r['topic']):3d} | JS={r['js_divergence_P1_vs_P4']:.3f} | {r['display_name']}"
        )
        lines.append(f"       + entering: {r['entering_terms_P4']}")
        lines.append(f"       − leaving:  {r['leaving_terms_P4']}")

    lines.extend(["", "B2. COMPOSITION-STABLE THEMES (low JS divergence) — sample"])
    if not comp.empty:
        stable_comp = comp[comp["composition_class"] == "composition_stable"].nsmallest(8, "js_divergence_P1_vs_P4")
        for _, r in stable_comp.iterrows():
            lines.append(
                f"  T{int(r['topic']):3d} | JS={r['js_divergence_P1_vs_P4']:.3f} | {r['display_name']}"
            )

    if not bertopic_evolution.empty:
        lines.extend(["", "B3. BERTopic representative words by period (c-TF-IDF evolution)"])
        for topic_id in rising.head(5)["topic"]:
            sub = bertopic_evolution[bertopic_evolution["topic"] == topic_id]
            if sub.empty:
                continue
            name = rising.loc[rising["topic"] == topic_id, "display_name"].iloc[0]
            lines.append(f"  T{int(topic_id)} — {name}")
            for _, r in sub.iterrows():
                lines.append(f"    {r['period']}: {r['representative_words']}")

    lines.extend(
        [
            "",
            "—" * 40,
            "C. SYNTHESIS",
            "—" * 40,
            "",
            "Prevalence dynamics:",
            "  • Field expansion post-2021 inflates absolute counts; interpretation uses SHARES.",
            "  • Rising: digital/CSAM abuse, campus prevention, ACE, CM–depression, complex PTSD.",
            "  • Declining: COVID-19-specific cluster, epigenetics/HPA animal models, HIV syndemic.",
            "  • Stable cores: IPV, forensic pediatrics, attachment, EMDR/PTSD treatment.",
            "",
            "Composition dynamics:",
            "  • High-JS themes show vocabulary turnover (e.g., shifting methods/populations within cluster).",
            "  • Stable-composition themes retain definitional core despite prevalence change.",
            "",
            "Period interpretation:",
            "  P1 (2019–20): baseline pre-pandemic; P2 (21–22): pandemic surge & pivot;",
            "  P3 (23–24): post-pandemic normalization; P4 (25–26): contemporary consolidation.",
            "",
            "LIMITATIONS",
            "  BERTopic topics are time-sliced assignments from a global model; composition metrics",
            "  use preprocessed abstract/title tokens. Short windows reduce power for rare themes.",
            "",
            "OUTPUT FILES",
            "  rq3_period_corpus_stats.csv",
            "  rq3_period_prevalence.csv",
            "  rq3_prevalence_shift.csv",
            "  rq3_composition_shift.csv",
            "  rq3_composition_terms.csv",
            "  rq3_bertopic_words_by_period.csv",
            "  rq3_meta_theme_periods.csv",
            "  rq3_period_prevalence_heatmap.png",
            "  rq3_rising_themes.png / rq3_declining_themes.png",
            "=" * 78,
        ]
    )
    (OUT_DIR / "rq3_report.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    t0 = time.perf_counter()
    print("RQ3 — Prevalence & composition shifts across historical periods")

    df, qa, suicide_ids = load_merged_frame()
    topics = df["topic"].to_numpy()
    sub_ids = substantive_topic_ids(qa)
    print(f"  Documents: {len(df):,} | Substantive topics: {len(sub_ids)}")

    corpus_stats = period_corpus_stats(df)
    corpus_stats.to_csv(OUT_DIR / "rq3_period_corpus_stats.csv", index=False)

    period_prev = period_topic_prevalence(df, sub_ids)
    period_prev.to_csv(OUT_DIR / "rq3_period_prevalence.csv", index=False)

    yearly = yearly_topic_shares(df, topics, allowed_topics=sub_ids)
    shift = prevalence_shift_table(period_prev, yearly, qa)
    shift.to_csv(OUT_DIR / "rq3_prevalence_shift.csv", index=False)

    comp, comp_terms = composition_shift_analysis(df, sub_ids)
    comp.to_csv(OUT_DIR / "rq3_composition_shift.csv", index=False)
    comp_terms.to_csv(OUT_DIR / "rq3_composition_terms.csv", index=False)

    bertopic_evo = bertopic_words_over_periods()
    if not bertopic_evo.empty:
        bertopic_evo.to_csv(OUT_DIR / "rq3_bertopic_words_by_period.csv", index=False)

    meta_period = meta_theme_by_period(df, suicide_ids)
    meta_period.to_csv(OUT_DIR / "rq3_meta_theme_periods.csv", index=False)

    print("Plotting...")
    plot_period_heatmap(period_prev, shift)
    plot_trajectory_lines(period_prev, shift)
    plot_meta_theme(meta_period)

    runtime = time.perf_counter() - t0
    rising_df = shift[shift["prevalence_class"] == "rising"]
    declining_df = shift[shift["prevalence_class"] == "declining"]
    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "historical_periods": {k: list(v) for k, v in HISTORICAL_PERIODS.items()},
        "n_documents": len(df),
        "n_rising": int(len(rising_df)),
        "n_declining": int(len(declining_df)),
        "n_stable": int((shift["prevalence_class"] == "stable").sum()),
        "top_rising": rising_df.head(8).to_dict(orient="records"),
        "top_declining": declining_df.head(8).to_dict(orient="records"),
        "most_recomposed": comp.head(6)[["topic", "display_name", "js_divergence_P1_vs_P4"]].to_dict(orient="records")
        if not comp.empty
        else [],
        "meta_suicidality_by_period": meta_period.to_dict(orient="records"),
        "runtime_seconds": round(runtime, 2),
    }
    (OUT_DIR / "rq3_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    write_report(corpus_stats, shift, comp, bertopic_evo, meta_period, runtime)
    print(f"\nDone in {runtime:.1f}s -> {OUT_DIR}")
    print(f"  Rising: {summary['n_rising']} | Declining: {summary['n_declining']} | Stable: {summary['n_stable']}")
    print(f"  Report: {OUT_DIR / 'rq3_report.txt'}")


if __name__ == "__main__":
    main()
