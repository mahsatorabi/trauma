"""
RQ1 — Temporal evolution of trauma-research themes and shift drivers.

Pipeline (quality-controlled):
  1. English-only filter (langdetect + Spanish-token heuristic on abstracts)
  2. BERTopic + topics_over_time
  3. Exclude language-artifact & flag methods topics
  4. Merge suicidality/self-harm subtopics into meta-theme
  5. Mann–Kendall trends, drivers, plots (substantive topics only)
"""

from __future__ import annotations

import json
import re
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bertopic import BERTopic
from scipy import stats
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "data_preprocessed.csv"
OUT_DIR = BASE_DIR / "rq1_output"
OUT_DIR.mkdir(exist_ok=True)

EARLY_PERIOD = (2019, 2021)
LATE_PERIOD = (2023, 2025)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PREFER_OFFLINE_EMBEDDINGS = True  # TF-IDF+SVD (HF often blocked by local proxy)
MIN_TOPIC_SIZE = 60
NR_TOPICS = None  # do not force-merge topics (avoids one dominant mega-topic)
TOP_N_WORDS = 12
RANDOM_SEED = 42

# Spanish / HTML artifact tokens (topic QA)
SPANISH_MARKER_TOKENS = {
    "de", "la", "el", "los", "las", "en", "un", "una", "que", "con", "por",
    "del", "al", "se", "su", "le", "es", "y", "o", "a", "e", "u",
    "eacute", "oacute", "iacute", "uacute", "ntilde", "agrave", "ocirc",
}
HTML_ARTIFACT_TOKENS = {"eacute", "oacute", "iacute", "uacute", "ntilde", "agrave", "ocirc"}
MIN_ENGLISH_CONFIDENCE = 0.85
MAX_SPANISH_TOKEN_RATIO = 0.12  # on preprocessed tokens

# Methods / statistics / psychometrics (excluded from substantive RQ1 trends)
METHODS_PHRASE_MARKERS = {
    "network structure", "social network", "network approach", "network analysis",
    "psychometric network", "gaussian graphical", "ebicglasso", "centrality",
    "latent profile", "latent class", "class class", "latent class analysis",
    "structural equation", "confirmatory factor", "internal consistency",
    "psychometric property", "item response", "logistic regression",
    "multivariable", "adjusted odds", "aor ci", "machine learning",
    "random forest", "neural network", "predictive model", "moderated mediation",
    "mediation analysis", "path analysis", "roc auc", "factor analysis",
    "exploratory factor", "convergent validity", "discriminant validity",
}
METHODS_NETWORK_WORDS = {
    "node", "bridge", "central", "centrality", "ela", "edge", "network",
    "strength", "ebicglasso",
}
METHODS_STATS_WORDS = {
    "psychometric", "psychometrics", "validity", "reliability", "cronbach",
    "confirmatory", "cfa", "sem", "irt", "lpa", "lca", "latent",
    "regression", "multivariable", "aor", "mediation", "moderator",
}

# Suicide / self-harm subtopics → merged meta-theme (filled after clustering)
SUICIDE_LABEL_MARKERS = (
    "suicidal", "ideation", "suicide", "selfharm", "self-injury", "nssi",
    "nonsuicidal", "suicidality",
)

# Optional display-name overrides (filled after fit from keywords if empty)
TOPIC_DISPLAY_NAMES: dict[int, str] = {}


# ---------------------------------------------------------------------------
def load_corpus() -> pd.DataFrame:
    usecols = [
        "UT (Unique WOS ID)",
        "Article Title",
        "Abstract",
        "preprocessed_text",
        "token_count",
        "Publication Year",
        "Document Type",
        "Keywords Plus",
        "Times Cited, WoS Core",
        "Research Areas",
        "Language",
    ]
    df = pd.read_csv(INPUT_CSV, usecols=usecols, low_memory=False)
    df = df.dropna(subset=["preprocessed_text", "Publication Year"])
    df = df[df["preprocessed_text"].str.strip().astype(bool)]
    df["Publication Year"] = df["Publication Year"].astype(int)
    df = df[(df["Publication Year"] >= 2019) & (df["Publication Year"] <= 2026)]
    df = df.reset_index(drop=True)
    return df


def _detect_language_langdetect(text: str) -> tuple[str, float]:
    try:
        from langdetect import DetectorFactory, detect_langs

        DetectorFactory.seed = RANDOM_SEED
        if not text or len(text.strip()) < 40:
            return "unknown", 0.0
        langs = detect_langs(text[:5000])
        if not langs:
            return "unknown", 0.0
        top = langs[0]
        return top.lang, float(top.prob)
    except Exception:
        return "unknown", 0.0


def spanish_token_ratio(preprocessed: str) -> float:
    tokens = preprocessed.split()
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t in SPANISH_MARKER_TOKENS)
    return hits / len(tokens)


def filter_english_corpus(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (english_df, removed_log)."""
    records: list[dict] = []
    keep_mask: list[bool] = []

    for _, row in df.iterrows():
        abstract = str(row.get("Abstract", "") or "")
        prep = str(row["preprocessed_text"])
        lang, conf = _detect_language_langdetect(abstract)
        sp_ratio = spanish_token_ratio(prep)
        scopus_lang = str(row.get("Language", "")).lower()

        reasons: list[str] = []
        if scopus_lang and scopus_lang != "english":
            reasons.append(f"scopus_language={scopus_lang}")
        if lang not in ("en", "unknown") and conf >= 0.9:
            reasons.append(f"langdetect={lang}({conf:.2f})")
        if lang == "en" and conf < MIN_ENGLISH_CONFIDENCE and conf > 0:
            reasons.append(f"low_english_confidence={conf:.2f}")
        if sp_ratio >= MAX_SPANISH_TOKEN_RATIO:
            reasons.append(f"spanish_token_ratio={sp_ratio:.3f}")

        remove = bool(reasons)
        keep_mask.append(not remove)
        if remove:
            records.append(
                {
                    "UT (Unique WOS ID)": row["UT (Unique WOS ID)"],
                    "Article Title": row.get("Article Title", ""),
                    "detected_lang": lang,
                    "lang_confidence": conf,
                    "spanish_token_ratio": sp_ratio,
                    "reason": "; ".join(reasons),
                }
            )

    english_df = df.loc[keep_mask].reset_index(drop=True)
    removed = pd.DataFrame(records)
    return english_df, removed


def _clear_proxy_env() -> None:
    import os

    for key in list(os.environ):
        if "proxy" in key.lower():
            os.environ.pop(key, None)


def compute_document_embeddings(docs: list[str]) -> tuple[np.ndarray, str]:
    _clear_proxy_env()
    if not PREFER_OFFLINE_EMBEDDINGS:
        try:
            model = SentenceTransformer(EMBEDDING_MODEL)
            emb = model.encode(docs, show_progress_bar=True, batch_size=64)
            return emb, EMBEDDING_MODEL
        except Exception as exc:
            print(f"  [fallback] Transformer embeddings unavailable: {exc}")

    from sklearn.decomposition import TruncatedSVD

    vec = TfidfVectorizer(
        max_features=25_000,
        ngram_range=(1, 2),
        min_df=10,
        max_df=0.92,
    )
    matrix = vec.fit_transform(docs)
    svd = TruncatedSVD(n_components=128, random_state=RANDOM_SEED)
    emb = svd.fit_transform(matrix)
    label = "TF-IDF + TruncatedSVD (128d) [offline fallback]"
    print(f"  Using embeddings: {label}")
    return emb, label


def build_topic_model(docs: list[str]) -> tuple[BERTopic, np.ndarray, str]:
    embeddings, emb_label = compute_document_embeddings(docs)
    vectorizer = CountVectorizer(
        ngram_range=(1, 2),
        min_df=10,
        max_df=0.90,
    )
    topic_model = BERTopic(
        vectorizer_model=vectorizer,
        min_topic_size=MIN_TOPIC_SIZE,
        nr_topics=NR_TOPICS,
        top_n_words=TOP_N_WORDS,
        calculate_probabilities=False,
        verbose=True,
    )
    topics, _ = topic_model.fit_transform(docs, embeddings)
    return topic_model, np.array(topics), emb_label


def topic_keywords(topic_model: BERTopic, topic_id: int) -> list[str]:
    tw = topic_model.get_topic(int(topic_id)) or []
    return [w for w, _ in tw]


def load_topic_words_from_info(path: Path | None = None) -> dict[int, list[str]]:
    import ast

    info_path = path or (OUT_DIR / "rq1_topic_info.csv")
    info = pd.read_csv(info_path)
    out: dict[int, list[str]] = {}
    for _, row in info.iterrows():
        tid = int(row["Topic"])
        if tid < 0:
            continue
        rep = row["Representation"]
        words = ast.literal_eval(rep) if isinstance(rep, str) else list(rep)
        out[tid] = [str(w) for w in words]
    return out


def resolve_topic_words(
    topic_id: int,
    topic_model: BERTopic | None = None,
    topic_words: dict[int, list[str]] | None = None,
) -> list[str]:
    if topic_words is not None and topic_id in topic_words:
        return topic_words[topic_id]
    if topic_model is not None:
        return topic_keywords(topic_model, topic_id)
    return []


def is_methods_topic(words: list[str]) -> bool:
    if not words:
        return False
    joined = " ".join(words)
    top5 = words[:5]
    top8 = words[:8]
    top5_set = set(top5)

    if any(p in joined for p in METHODS_PHRASE_MARKERS):
        return True

    network_hits = len(top5_set & METHODS_NETWORK_WORDS)
    if network_hits >= 2:
        return True
    if "node" in top5_set and any(
        w in joined for w in ("network", "centrality", "bridge", "social", "structure")
    ):
        return True

    if "aor" in top5_set or ("multivariable" in joined and "adjust" in joined):
        return True

    stats_hits = sum(1 for w in top8 if w in METHODS_STATS_WORDS)
    if stats_hits >= 3:
        return True
    if "psychometric" in top5_set and any(
        w in joined for w in ("internal", "consistency", "confirmatory", "validity", "property")
    ):
        return True
    if top5_set & {"latent", "lca", "lpa"} and any(
        w in joined for w in ("class", "profile", "membership", "identify")
    ):
        return True
    return False


def classify_topic_words(words: list[str]) -> str:
    if not words:
        return "unknown"
    joined = " ".join(words)
    top8 = words[:8]
    top5 = words[:5]
    spanish_hits = sum(1 for w in top5 if w in SPANISH_MARKER_TOKENS)
    html_hits = sum(1 for w in top5 if w in HTML_ARTIFACT_TOKENS)
    if spanish_hits >= 3 or html_hits >= 2:
        return "language_artifact"
    if is_methods_topic(words):
        return "methods"
    suicide_hits = sum(1 for w in top8 if any(m in w for m in SUICIDE_LABEL_MARKERS))
    if suicide_hits >= 2:
        return "suicidality"
    return "substantive"


def classify_topic(
    topic_model: BERTopic | None,
    topic_id: int,
    topic_words: dict[int, list[str]] | None = None,
) -> str:
    if topic_id < 0:
        return "outlier"
    words = resolve_topic_words(topic_id, topic_model, topic_words)
    if not words:
        return "unknown"
    return classify_topic_words(words)


def auto_display_name(words: list[str]) -> str:
    """Readable theme label from top keywords."""
    if not words:
        return "Unknown theme"
    joined = " ".join(words[:10])
    rules: list[tuple[str, str]] = [
        ("ace", "Adverse childhood experiences (ACE)"),
        ("ipv", "Intimate partner violence"),
        ("suicidal", "Suicidality & suicidal ideation"),
        ("selfharm", "Self-harm"),
        ("nssi", "Non-suicidal self-injury (NSSI)"),
        ("emdr", "PTSD treatment (EMDR / trauma-focused CBT)"),
        ("psychosis", "Psychosis & psychotic-like experiences"),
        ("methylation", "Epigenetics & DNA methylation"),
        ("maternal", "Maternal & infant mental health"),
        ("covid", "COVID-19 pandemic impact"),
        ("college", "College students & campus prevention"),
        ("imagebased", "Image-based sexual abuse"),
        ("borderline", "Borderline personality disorder"),
    ]
    for key, label in rules:
        if key in joined:
            return label
    return ", ".join(words[:4]).title()


def display_name(
    topic_model: BERTopic | None,
    topic_id: int,
    topic_words: dict[int, list[str]] | None = None,
) -> str:
    if topic_id in TOPIC_DISPLAY_NAMES:
        return TOPIC_DISPLAY_NAMES[topic_id]
    return auto_display_name(resolve_topic_words(topic_id, topic_model, topic_words))


def identify_suicide_topics(
    topic_model: BERTopic | None = None,
    topic_words: dict[int, list[str]] | None = None,
    topic_ids: list[int] | None = None,
) -> list[int]:
    if topic_ids is None:
        if topic_model is not None:
            topic_ids = [int(t) for t in topic_model.get_topics() if t >= 0]
        elif topic_words is not None:
            topic_ids = sorted(topic_words)
        else:
            return []
    ids: list[int] = []
    for tid in topic_ids:
        if tid < 0:
            continue
        words = resolve_topic_words(tid, topic_model, topic_words)[:8]
        hits = sum(1 for w in words if any(m in w for m in SUICIDE_LABEL_MARKERS))
        if hits >= 2:
            ids.append(int(tid))
    return ids


def build_topic_qa_table(
    topics: np.ndarray,
    topic_model: BERTopic | None = None,
    topic_words: dict[int, list[str]] | None = None,
) -> pd.DataFrame:
    rows = []
    for tid in sorted(set(topics)):
        if tid < 0:
            continue
        ttype = classify_topic(topic_model, int(tid), topic_words)
        kw = resolve_topic_words(int(tid), topic_model, topic_words)
        rows.append(
            {
                "topic": int(tid),
                "topic_type": ttype,
                "include_in_trends": ttype == "substantive",
                "display_name": display_name(topic_model, int(tid), topic_words),
                "label_keywords": ", ".join(kw[:8]),
                "document_count": int((topics == tid).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("document_count", ascending=False)


def substantive_topic_ids(qa: pd.DataFrame) -> list[int]:
    return qa.loc[qa["include_in_trends"], "topic"].astype(int).tolist()


def topics_over_time_analysis(
    topic_model: BERTopic,
    docs: list[str],
    timestamps: list[int],
) -> pd.DataFrame:
    return topic_model.topics_over_time(
        docs,
        timestamps,
        global_tuning=True,
        evolution_tuning=True,
    )


def yearly_topic_shares(
    df: pd.DataFrame,
    topics: np.ndarray,
    allowed_topics: list[int] | None = None,
) -> pd.DataFrame:
    tmp = df[["Publication Year"]].copy()
    tmp["topic"] = topics
    tmp = tmp[tmp["topic"] != -1]
    if allowed_topics is not None:
        tmp = tmp[tmp["topic"].isin(allowed_topics)]
    counts = tmp.groupby(["Publication Year", "topic"]).size().reset_index(name="count")
    year_totals = tmp.groupby("Publication Year").size().rename("year_total")
    counts = counts.merge(year_totals, on="Publication Year")
    counts["share"] = counts["count"] / counts["year_total"]
    return counts


def build_meta_theme_shares(
    df: pd.DataFrame,
    topics: np.ndarray,
    suicide_topic_ids: list[int],
    meta_name: str = "suicidality_self_harm",
) -> pd.DataFrame:
    rows = []
    for year in sorted(df["Publication Year"].unique()):
        year_topics = topics[df["Publication Year"] == year]
        valid = year_topics[year_topics != -1]
        total = len(valid)
        count = int(np.isin(year_topics, suicide_topic_ids).sum())
        rows.append(
            {
                "Publication Year": int(year),
                "meta_theme": meta_name,
                "count": count,
                "year_total": total,
                "share": count / total if total else 0.0,
                "source_topics": ",".join(map(str, suicide_topic_ids)),
            }
        )
    return pd.DataFrame(rows)


def mann_kendall_trend(y: np.ndarray) -> dict:
    x = np.arange(len(y))
    if len(y) < 4 or np.allclose(y, y[0]):
        return {"tau": np.nan, "p_value": np.nan, "trend": "insufficient"}
    tau, p = stats.kendalltau(x, y, nan_policy="omit")
    if np.isnan(p):
        return {"tau": float(tau), "p_value": np.nan, "trend": "insufficient"}
    trend = "increasing" if p < 0.05 and tau > 0 else "decreasing" if p < 0.05 and tau < 0 else "stable"
    return {"tau": float(tau), "p_value": float(p), "trend": trend}


def theil_sen_slope(y: np.ndarray) -> float:
    if len(y) < 2:
        return np.nan
    res = stats.theilslopes(y, np.arange(len(y)))
    return float(res[0])


def detect_changepoints(y: np.ndarray) -> list[int]:
    try:
        import ruptures as rpt

        if len(y) < 6:
            return []
        algo = rpt.Pelt(model="rbf", min_size=2, jump=1).fit(y.reshape(-1, 1))
        return [b for b in algo.predict(pen=0.5) if b < len(y)]
    except Exception:
        return []


def topic_trend_table(
    shares: pd.DataFrame,
    qa: pd.DataFrame,
) -> pd.DataFrame:
    qa_map = qa.set_index("topic")
    rows: list[dict] = []
    for topic_id in sorted(shares["topic"].unique()):
        sub = shares[shares["topic"] == topic_id].sort_values("Publication Year")
        y = sub["share"].values
        mk = mann_kendall_trend(y)
        ttype = qa_map.loc[topic_id, "topic_type"] if topic_id in qa_map.index else "unknown"
        rows.append(
            {
                "topic": int(topic_id),
                "topic_type": ttype,
                "display_name": qa_map.loc[topic_id, "display_name"] if topic_id in qa_map.index else "",
                "label_keywords": qa_map.loc[topic_id, "label_keywords"] if topic_id in qa_map.index else "",
                "years": len(sub),
                "mean_share": float(np.mean(y)),
                "share_2019_2021": float(sub[sub["Publication Year"].between(*EARLY_PERIOD)]["share"].mean())
                if not sub[sub["Publication Year"].between(*EARLY_PERIOD)].empty
                else np.nan,
                "share_2023_2025": float(sub[sub["Publication Year"].between(*LATE_PERIOD)]["share"].mean())
                if not sub[sub["Publication Year"].between(*LATE_PERIOD)].empty
                else np.nan,
                "kendall_tau": mk["tau"],
                "kendall_p": mk["p_value"],
                "trend": mk["trend"],
                "theil_sen_slope_per_year": theil_sen_slope(y) if len(y) >= 4 else np.nan,
                "changepoint_indices": detect_changepoints(y),
            }
        )
    return pd.DataFrame(rows).sort_values("kendall_tau", ascending=False, na_position="last")


def meta_theme_trend(meta_shares: pd.DataFrame) -> dict:
    sub = meta_shares.sort_values("Publication Year")
    y = sub["share"].values
    mk = mann_kendall_trend(y)
    return {
        "meta_theme": sub["meta_theme"].iloc[0],
        "source_topics": sub["source_topics"].iloc[0],
        **mk,
        "theil_sen_slope_per_year": theil_sen_slope(y) if len(y) >= 4 else np.nan,
        "mean_share": float(np.mean(y)),
    }


def emerging_keywords_log_odds(df: pd.DataFrame, top_k: int = 40) -> pd.DataFrame:
    def tokenize_kw(s: str) -> list[str]:
        if not isinstance(s, str):
            return []
        return [t.strip().lower() for t in s.replace(";", ",").split(",") if t.strip()]

    early = df[df["Publication Year"].between(*EARLY_PERIOD)]
    late = df[df["Publication Year"].between(*LATE_PERIOD)]

    def kw_counts(frame: pd.DataFrame) -> Counter:
        c: Counter = Counter()
        for val in frame["Keywords Plus"].dropna():
            c.update(tokenize_kw(val))
        return c

    c_early, c_late = kw_counts(early), kw_counts(late)
    n_early, n_late = max(sum(c_early.values()), 1), max(sum(c_late.values()), 1)
    rows = []
    for term in set(c_early) | set(c_late):
        a, b = c_early.get(term, 0), c_late.get(term, 0)
        lo = np.log((b + 0.5) / (n_late - b + 0.5)) - np.log((a + 0.5) / (n_early - a + 0.5))
        rows.append({"term": term, "count_early": a, "count_late": b, "log_odds": lo})
    out = pd.DataFrame(rows).sort_values("log_odds", ascending=False)
    return pd.concat([out.head(top_k), out.sort_values("log_odds").head(top_k)]).drop_duplicates("term")


def driver_correlations(df: pd.DataFrame, shares: pd.DataFrame) -> pd.DataFrame:
    annual = df.groupby("Publication Year").agg(
        n_papers=("UT (Unique WOS ID)", "count"),
        mean_citations=("Times Cited, WoS Core", "mean"),
        pct_review=("Document Type", lambda s: s.astype(str).str.contains("Review", case=False).mean()),
    )
    merged = shares.merge(annual, on="Publication Year", how="left")
    rows = []
    for topic_id in sorted(shares["topic"].unique()):
        sub = merged[merged["topic"] == topic_id].sort_values("Publication Year")
        if len(sub) < 4:
            continue
        for cov in ["n_papers", "mean_citations", "pct_review"]:
            if sub[cov].nunique() < 2 or sub["share"].nunique() < 2:
                continue
            r, p = stats.spearmanr(sub["share"], sub[cov], nan_policy="omit")
            if np.isnan(r):
                continue
            rows.append(
                {
                    "topic": int(topic_id),
                    "covariate": cov,
                    "spearman_r": float(r),
                    "p_value": float(p),
                }
            )
    return pd.DataFrame(rows)


def plot_topics_over_time(
    tot: pd.DataFrame,
    qa: pd.DataFrame,
    trends: pd.DataFrame,
    top_n: int = 8,
) -> None:
    substantive = set(substantive_topic_ids(qa))
    freq = (
        tot[tot["Topic"].isin(substantive)]
        .groupby("Topic")["Frequency"]
        .mean()
        .sort_values(ascending=False)
    )
    top_topics = [int(t) for t in freq.head(top_n).index]
    sub = tot[tot["Topic"].isin(top_topics)]
    pivot = sub.pivot_table(index="Topic", columns="Timestamp", values="Frequency", aggfunc="mean")
    pivot = pivot.reindex(top_topics)

    qa_idx = qa.set_index("topic")
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(len(pivot)))
    labels = [f"T{int(t)}: {qa_idx.loc[t, 'display_name'][:50]}" for t in pivot.index]
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([int(c) for c in pivot.columns], rotation=45)
    ax.set_xlabel("Publication year")
    ax.set_title("RQ1 — Substantive themes over time (artifacts excluded)")
    plt.colorbar(im, ax=ax, label="Frequency")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "rq1_topics_heatmap.png", dpi=200)
    plt.close(fig)

    sub_trends = trends[
        (trends["topic_type"] == "substantive") & (trends["trend"].isin(["increasing", "decreasing"]))
    ]
    inc = sub_trends[sub_trends["trend"] == "increasing"].head(5)["topic"].tolist()
    dec = sub_trends[sub_trends["trend"] == "decreasing"].head(3)["topic"].tolist()
    sel = [int(t) for t in inc + dec if int(t) in tot["Topic"].values]

    fig, ax = plt.subplots(figsize=(11, 6))
    for t in sel:
        s = tot[tot["Topic"] == t].sort_values("Timestamp")
        name = qa_idx.loc[t, "display_name"] if t in qa_idx.index else str(t)
        ax.plot(s["Timestamp"], s["Frequency"], marker="o", label=f"T{t}: {name[:40]}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Topic frequency")
    ax.set_title("RQ1 — Rising & declining substantive themes")
    ax.legend(fontsize=7, loc="best")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "rq1_topics_lines.png", dpi=200)
    plt.close(fig)


def plot_yearly_volume(df: pd.DataFrame) -> None:
    vol = df.groupby("Publication Year").size()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(vol.index, vol.values, color="steelblue")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of papers")
    ax.set_title("RQ1 — Annual publication volume (English-filtered corpus)")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "rq1_annual_volume.png", dpi=200)
    plt.close(fig)


def write_report(
    df: pd.DataFrame,
    removed: pd.DataFrame,
    qa: pd.DataFrame,
    trends: pd.DataFrame,
    meta_tr: dict,
    emerging: pd.DataFrame,
    drivers: pd.DataFrame,
    n_topics: int,
    n_outliers: int,
    runtime_sec: float,
    emb_label: str,
    suicide_ids: list[int],
) -> None:
    n_removed = len(removed)
    n_artifact = int((qa["topic_type"] == "language_artifact").sum())
    n_methods = int((qa["topic_type"] == "methods").sum())
    sub_trends = trends[trends["topic_type"] == "substantive"]

    lines = [
        "=" * 78,
        "RQ1 REPORT — THEMATIC EVOLUTION & SHIFT DRIVERS (QUALITY-CONTROLLED)",
        "=" * 78,
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Runtime: {runtime_sec:.1f} s",
        "",
        "RESEARCH QUESTION",
        "  How have research themes and priorities in trauma studies evolved over time,",
        "  and what factors have driven these shifts?",
        "",
        "DATA & QUALITY CONTROL",
        f"  Source: {INPUT_CSV.name}",
        f"  After English filter: {len(df):,} documents",
        f"  Removed (non-English / Spanish-heavy): {n_removed:,}",
        f"  Years: {df['Publication Year'].min()}–{df['Publication Year'].max()}",
        "  Filters: langdetect on abstract; Spanish-token ratio on preprocessed text;",
        f"            max Spanish-token ratio = {MAX_SPANISH_TOKEN_RATIO}",
        "",
        "METHOD",
        "  1. BERTopic (UMAP + HDBSCAN + c-TF-IDF)",
        f"     Embeddings: {emb_label}",
        f"     min_cluster_size = {MIN_TOPIC_SIZE}",
        "  2. topics_over_time (yearly bins)",
        "  3. Topic QA: exclude language_artifact & methods from trend tables",
        "  4. Meta-theme merge: suicidality_self_harm = topics "
        + ", ".join(map(str, suicide_ids)),
        "  5. Mann–Kendall on yearly topic share (substantive only)",
        "  6. Driver analysis (volume, citations, reviews, Keywords Plus log-odds)",
        "",
        "MODEL SUMMARY",
        f"  Substantive topics: {n_topics}",
        f"  Language-artifact topics (excluded): {n_artifact}",
        f"  Methods topics (reported separately): {n_methods}",
        f"  Outlier documents (topic -1): {n_outliers:,}",
        "",
        "META-THEME: SUICIDALITY & SELF-HARM (merged)",
        f"  Source BERTopic IDs: {suicide_ids}",
        f"  Trend: {meta_tr.get('trend', 'n/a')} | tau={meta_tr.get('tau', float('nan')):.3f} "
        f"| p={meta_tr.get('p_value', float('nan')):.4f}",
        "",
        "SUBSTANTIVE TOPIC TRENDS (Mann–Kendall on yearly share)",
    ]
    for _, row in sub_trends.head(20).iterrows():
        lines.append(
            f"  T{int(row['topic']):2d} | {row['trend']:11s} | tau={row['kendall_tau']:.3f} "
            f"p={row['kendall_p']:.4f} | {row['display_name']}"
        )

    lines.extend(["", "EXCLUDED TOPICS (do not interpret as research themes)"])
    excl = qa[qa["topic_type"].isin(["language_artifact", "methods"])]
    for _, row in excl.iterrows():
        lines.append(
            f"  T{int(row['topic']):2d} [{row['topic_type']}] {row['label_keywords']}"
        )

    lines.extend(["", "EMERGING KEYWORDS PLUS (2023–2025 vs 2019–2021, top log-odds)"])
    for _, row in emerging.nlargest(15, "log_odds").iterrows():
        lines.append(
            f"  + {row['term']:30s} log-odds={row['log_odds']:+.3f} "
            f"(early={int(row['count_early'])}, late={int(row['count_late'])})"
        )

    lines.extend(["", "DRIVER CORRELATIONS (p < 0.05, substantive topics)"])
    sig = drivers[drivers["p_value"] < 0.05].sort_values("spearman_r", key=lambda s: s.abs(), ascending=False)
    for _, row in sig.head(15).iterrows():
        lines.append(
            f"  T{int(row['topic']):2d} <-> {row['covariate']:15s} r={row['spearman_r']:+.3f} p={row['p_value']:.4f}"
        )

    lines.extend(
        [
            "",
            "OUTPUT FILES",
            "  rq1_document_topics.csv",
            "  rq1_removed_non_english.csv",
            "  rq1_topic_qa.csv",
            "  rq1_topic_trends.csv (all classified topics)",
            "  rq1_substantive_topic_trends.csv",
            "  rq1_meta_theme_shares.csv",
            "  rq1_topics_over_time.csv",
            "  rq1_topic_info.csv",
            "=" * 78,
        ]
    )
    (OUT_DIR / "rq1_report.txt").write_text("\n".join(lines), encoding="utf-8")


def reclassify_outputs() -> None:
    """Re-apply topic QA and regenerate trends/plots without refitting BERTopic."""
    import time

    t0 = time.perf_counter()
    print("Reclassifying topics from saved BERTopic outputs...")

    df_raw = load_corpus()
    df, removed = filter_english_corpus(df_raw)
    doc_topics = pd.read_csv(OUT_DIR / "rq1_document_topics.csv")
    topics = doc_topics["topic"].to_numpy()
    topic_words = load_topic_words_from_info()

    summary_path = OUT_DIR / "rq1_summary.json"
    emb_label = "saved model (reclassify)"
    if summary_path.exists():
        prev = json.loads(summary_path.read_text(encoding="utf-8"))
        emb_label = prev.get("embedding_model", emb_label)

    qa = build_topic_qa_table(topics, topic_words=topic_words)
    qa.to_csv(OUT_DIR / "rq1_topic_qa.csv", index=False)

    suicide_ids = identify_suicide_topics(topic_words=topic_words)
    substantive_ids = substantive_topic_ids(qa)
    n_outliers = int((topics == -1).sum())

    doc_topics["topic_type"] = [classify_topic(None, int(t), topic_words) for t in topics]
    doc_topics["display_name"] = [
        display_name(None, int(t), topic_words) if t >= 0 else "outlier" for t in topics
    ]
    doc_topics.to_csv(OUT_DIR / "rq1_document_topics.csv", index=False)

    tot = pd.read_csv(OUT_DIR / "rq1_topics_over_time.csv")

    shares_all = yearly_topic_shares(df, topics)
    shares_all.to_csv(OUT_DIR / "rq1_yearly_topic_shares.csv", index=False)
    shares_sub = yearly_topic_shares(df, topics, allowed_topics=substantive_ids)
    shares_sub.to_csv(OUT_DIR / "rq1_yearly_topic_shares_substantive.csv", index=False)

    meta_shares = build_meta_theme_shares(df, topics, suicide_ids)
    meta_shares.to_csv(OUT_DIR / "rq1_meta_theme_shares.csv", index=False)
    meta_tr = meta_theme_trend(meta_shares)

    trends = topic_trend_table(shares_all, qa)
    trends.to_csv(OUT_DIR / "rq1_topic_trends.csv", index=False)
    sub_trends = trends[trends["topic_type"] == "substantive"].copy()
    sub_trends.to_csv(OUT_DIR / "rq1_substantive_topic_trends.csv", index=False)

    emerging = emerging_keywords_log_odds(df)
    emerging.to_csv(OUT_DIR / "rq1_emerging_keywords.csv", index=False)
    drivers = driver_correlations(df, shares_sub)
    drivers.to_csv(OUT_DIR / "rq1_driver_correlations.csv", index=False)

    plot_yearly_volume(df)
    plot_topics_over_time(tot, qa, trends)

    runtime = time.perf_counter() - t0
    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "mode": "reclassify",
        "n_documents_raw": len(df_raw),
        "n_documents_english": len(df),
        "n_removed_non_english": len(removed),
        "n_outliers": n_outliers,
        "n_substantive_topics": len(substantive_ids),
        "n_language_artifact_topics": int((qa["topic_type"] == "language_artifact").sum()),
        "n_methods_topics": int((qa["topic_type"] == "methods").sum()),
        "methods_topic_ids": qa.loc[qa["topic_type"] == "methods", "topic"].astype(int).tolist(),
        "suicide_subtopic_ids": suicide_ids,
        "meta_theme_suicidality": meta_tr,
        "embedding_model": emb_label,
        "runtime_seconds": round(runtime, 2),
        "top_increasing_substantive": sub_trends[sub_trends["trend"] == "increasing"]
        .head(10)[["topic", "display_name", "kendall_tau"]]
        .to_dict(orient="records"),
        "top_decreasing_substantive": sub_trends[sub_trends["trend"] == "decreasing"]
        .head(10)[["topic", "display_name", "kendall_tau"]]
        .to_dict(orient="records"),
    }
    (OUT_DIR / "rq1_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    write_report(
        df, removed, qa, trends, meta_tr, emerging, drivers,
        len(substantive_ids), n_outliers, runtime, emb_label, suicide_ids,
    )
    print(f"\nReclassify done in {runtime:.1f}s -> {OUT_DIR}")
    print(f"  Methods topics excluded: {summary['methods_topic_ids']}")


def main() -> None:
    import time

    t0 = time.perf_counter()

    print("Loading corpus...")
    df_raw = load_corpus()
    print(f"  Raw records: {len(df_raw):,}")

    print("Filtering to English-only documents...")
    df, removed = filter_english_corpus(df_raw)
    print(f"  Kept: {len(df):,} | Removed: {len(removed):,}")
    removed.to_csv(OUT_DIR / "rq1_removed_non_english.csv", index=False)

    docs = df["preprocessed_text"].tolist()
    timestamps = df["Publication Year"].tolist()

    print(f"Fitting BERTopic on {len(docs):,} documents...")
    topic_model, topics, emb_label = build_topic_model(docs)

    # Save per-document assignments
    doc_topics = df[["UT (Unique WOS ID)", "Article Title", "Publication Year"]].copy()
    doc_topics["topic"] = topics
    doc_topics["topic_type"] = [classify_topic(topic_model, int(t)) for t in topics]
    doc_topics["display_name"] = [
        display_name(topic_model, int(t)) if t >= 0 else "outlier" for t in topics
    ]
    doc_topics.to_csv(OUT_DIR / "rq1_document_topics.csv", index=False)

    qa = build_topic_qa_table(topics, topic_model=topic_model)
    qa.to_csv(OUT_DIR / "rq1_topic_qa.csv", index=False)

    suicide_ids = identify_suicide_topics(topic_model=topic_model)
    print(f"  Suicide/self-harm subtopics merged: {suicide_ids}")

    substantive_ids = substantive_topic_ids(qa)
    n_outliers = int((topics == -1).sum())

    print("Computing topics over time...")
    tot = topics_over_time_analysis(topic_model, docs, timestamps)
    tot.to_csv(OUT_DIR / "rq1_topics_over_time.csv", index=False)
    topic_model.get_topic_info().to_csv(OUT_DIR / "rq1_topic_info.csv", index=False)

    shares_all = yearly_topic_shares(df, topics)
    shares_all.to_csv(OUT_DIR / "rq1_yearly_topic_shares.csv", index=False)

    shares_sub = yearly_topic_shares(df, topics, allowed_topics=substantive_ids)
    shares_sub.to_csv(OUT_DIR / "rq1_yearly_topic_shares_substantive.csv", index=False)

    meta_shares = build_meta_theme_shares(df, topics, suicide_ids)
    meta_shares.to_csv(OUT_DIR / "rq1_meta_theme_shares.csv", index=False)
    meta_tr = meta_theme_trend(meta_shares)

    print("Trend & driver analysis...")
    trends = topic_trend_table(shares_all, qa)
    trends.to_csv(OUT_DIR / "rq1_topic_trends.csv", index=False)

    sub_trends = trends[trends["topic_type"] == "substantive"].copy()
    sub_trends.to_csv(OUT_DIR / "rq1_substantive_topic_trends.csv", index=False)

    emerging = emerging_keywords_log_odds(df)
    emerging.to_csv(OUT_DIR / "rq1_emerging_keywords.csv", index=False)

    drivers = driver_correlations(df, shares_sub)
    drivers.to_csv(OUT_DIR / "rq1_driver_correlations.csv", index=False)

    print("Plotting...")
    plot_yearly_volume(df)
    plot_topics_over_time(tot, qa, trends)

    runtime = time.perf_counter() - t0
    n_substantive = len(substantive_ids)

    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "n_documents_raw": len(df_raw),
        "n_documents_english": len(df),
        "n_removed_non_english": len(removed),
        "n_outliers": n_outliers,
        "n_substantive_topics": n_substantive,
        "n_language_artifact_topics": int((qa["topic_type"] == "language_artifact").sum()),
        "n_methods_topics": int((qa["topic_type"] == "methods").sum()),
        "suicide_subtopic_ids": suicide_ids,
        "meta_theme_suicidality": meta_tr,
        "embedding_model": emb_label,
        "runtime_seconds": round(runtime, 2),
        "top_increasing_substantive": sub_trends[sub_trends["trend"] == "increasing"]
        .head(10)[["topic", "display_name", "kendall_tau"]]
        .to_dict(orient="records"),
        "top_decreasing_substantive": sub_trends[sub_trends["trend"] == "decreasing"]
        .head(10)[["topic", "display_name", "kendall_tau"]]
        .to_dict(orient="records"),
    }
    (OUT_DIR / "rq1_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    write_report(
        df, removed, qa, trends, meta_tr, emerging, drivers,
        n_substantive, n_outliers, runtime, emb_label, suicide_ids,
    )

    print(f"\nDone in {runtime:.1f}s -> {OUT_DIR}")
    print(f"  Report: {OUT_DIR / 'rq1_report.txt'}")


if __name__ == "__main__":
    import sys

    if "--reclassify" in sys.argv:
        reclassify_outputs()
    else:
        main()
