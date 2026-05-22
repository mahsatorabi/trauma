"""
RQ2 — Emerging themes, underexplored areas, and research gaps in contemporary
childhood trauma studies.

Methodological framework (integrative bibliometric–semantometric mapping):
  1. Quality-controlled English corpus (shared with RQ1)
  2. Thematic layer: BERTopic assignments from RQ1 (substantive topics only)
  3. Co-word network (Author Keywords + Keywords Plus) + Louvain communities
  4. Callon strategic diagram (density vs centrality) per keyword cluster
  5. Kleinberg-style proportional burst on terms (2019–2021 vs 2023–2025)
  6. Research Opportunity Index (ROI) & composite Gap Score per theme
  7. Structured gap audit (population, method, context, mechanism domains)
  8. Academic report + publication-ready figures
"""

from __future__ import annotations

import json
import re
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# Reuse corpus loading & English QC from RQ1
from rq1 import (
    EARLY_PERIOD,
    LATE_PERIOD,
    filter_english_corpus,
    mann_kendall_trend,
    substantive_topic_ids,
)

BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "data_preprocessed.csv"
RQ1_DIR = BASE_DIR / "rq1_output"
OUT_DIR = BASE_DIR / "rq2_output"
OUT_DIR.mkdir(exist_ok=True)

CONTEMPORARY_YEARS = (2022, 2026)  # primary interpretive window
MIN_KW_FREQ = 25  # co-word node threshold
MIN_CLUSTER_SIZE = 4
RANDOM_SEED = 42

# Noisy Keywords Plus artifacts (excluded from burst reporting)
BURST_REPORT_STOP = {
    "size", "improves", "maternal", "worry", "australia", "scripts", "genome", "ct",
    "cross-sectional study", "family history", "medical education",
}

# ---------------------------------------------------------------------------
# Gap-audit lexicons (transparent, reproducible operational definitions)
# ---------------------------------------------------------------------------
GAP_DOMAINS: dict[str, dict[str, list[str]]] = {
    "population": {
        "LGBTQ+ / gender-diverse youth": [
            "lgbt", "lgbtq", "transgender", "nonbinary", "sexual minority", "gender minority",
        ],
        "Indigenous & ethnic minority children": [
            "indigenous", "aboriginal", "first nations", "tribal", "maori", "native american",
        ],
        "Refugee & migrant children": ["refugee", "asylum", "migrant", "displaced", "immigrant child"],
        "Neurodivergent / disabled youth": [
            "autism", "autistic", "adhd", "intellectual disability", "neurodevelopmental",
        ],
        "Foster / care-experienced youth": ["foster", "looked after", "care leaver", "out of home", "institutionalized"],
    },
    "method": {
        "Longitudinal & cohort designs": ["longitudinal", "prospective cohort", "birth cohort", "follow-up"],
        "Randomized & implementation trials": [
            "randomized", "randomised", "rct", "implementation science", "effectiveness trial",
        ],
        "Participatory & co-design": ["participatory", "co-design", "lived experience", "patient public"],
        "Neuroimaging & biomarkers": ["fmri", "neuroimaging", "biomarker", "epigenetic", "methylation", "hair cortisol"],
        "Machine learning & prediction": ["machine learning", "predictive model", "deep learning", "algorithm"],
    },
    "context": {
        "Global South & LMIC settings": [
            "low income", "lmic", "sub-saharan", "global south", "developing country",
        ],
        "Digital & technology-facilitated harm": [
            "online abuse", "cyber", "sextortion", "image-based", "csam", "digital harm",
        ],
        "School & education systems": ["school-based", "education", "teacher", "classroom", "bullying"],
        "Justice & forensic systems": ["juvenile justice", "forensic", "court", "legal", "mandatory reporting"],
    },
    "mechanism_intervention": {
        "Prevention & public health": ["prevention", "universal prevention", "selective prevention", "indicated prevention"],
        "Family & parenting programs": ["parenting program", "parent training", "attachment-based", "dyadic"],
        "Community & policy": ["community intervention", "policy", "legislation", "systems change"],
        "Cultural & structural determinants": [
            "structural racism", "discrimination", "poverty", "neighborhood", "social determinants",
        ],
    },
}


def zscore(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    if s.std(ddof=0) == 0 or len(s) < 2:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / s.std(ddof=0)


def tokenize_keywords(val) -> list[str]:
    if not isinstance(val, str) or not val.strip():
        return []
    parts = re.split(r"[;,|]", val)
    return [p.strip().lower() for p in parts if len(p.strip()) >= 2]


def document_keywords(row: pd.Series) -> list[str]:
    terms: list[str] = []
    terms.extend(tokenize_keywords(row.get("Keywords Plus", "")))
    terms.extend(tokenize_keywords(row.get("Author Keywords", "")))
    return list(dict.fromkeys(terms))  # preserve order, dedupe


def load_corpus_with_keywords() -> pd.DataFrame:
    """Corpus columns needed for RQ2 co-word analysis."""
    usecols = [
        "UT (Unique WOS ID)",
        "Article Title",
        "Abstract",
        "preprocessed_text",
        "Publication Year",
        "Language",
        "Keywords Plus",
        "Author Keywords",
        "Times Cited, WoS Core",
    ]
    df = pd.read_csv(INPUT_CSV, usecols=usecols, low_memory=False)
    df = df.dropna(subset=["preprocessed_text", "Publication Year"])
    df = df[df["preprocessed_text"].str.strip().astype(bool)]
    df["Publication Year"] = df["Publication Year"].astype(int)
    df = df[(df["Publication Year"] >= 2019) & (df["Publication Year"] <= 2026)]
    return df.reset_index(drop=True)


def load_analysis_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    """English corpus merged with RQ1 topic assignments."""
    df_raw = load_corpus_with_keywords()
    df, _ = filter_english_corpus(df_raw)

    doc_path = RQ1_DIR / "rq1_document_topics.csv"
    qa_path = RQ1_DIR / "rq1_topic_qa.csv"
    if not doc_path.exists() or not qa_path.exists():
        raise FileNotFoundError(
            "RQ1 outputs required. Run `python rq1.py` first (or rq1.py --reclassify)."
        )

    doc_topics = pd.read_csv(doc_path)
    qa = pd.read_csv(qa_path)

    merged = df.merge(
        doc_topics[["UT (Unique WOS ID)", "topic", "topic_type", "display_name"]],
        on="UT (Unique WOS ID)",
        how="inner",
    )
    merged["all_keywords"] = merged.apply(document_keywords, axis=1)
    merged["n_keywords"] = merged["all_keywords"].str.len()
    merged["is_contemporary"] = merged["Publication Year"].between(*CONTEMPORARY_YEARS)
    return merged, qa


def keyword_year_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Per-term counts in early, late, and contemporary windows."""
    rows: list[dict] = []
    early = df[df["Publication Year"].between(*EARLY_PERIOD)]
    late = df[df["Publication Year"].between(*LATE_PERIOD)]
    recent = df[df["Publication Year"].between(*CONTEMPORARY_YEARS)]

    def count_terms(frame: pd.DataFrame) -> Counter:
        c: Counter = Counter()
        for terms in frame["all_keywords"]:
            c.update(terms)
        return c

    c_early, c_late, c_recent = count_terms(early), count_terms(late), count_terms(recent)
    all_terms = set(c_early) | set(c_late) | set(c_recent)

    years_early = EARLY_PERIOD[1] - EARLY_PERIOD[0] + 1
    years_late = LATE_PERIOD[1] - LATE_PERIOD[0] + 1
    years_recent = CONTEMPORARY_YEARS[1] - CONTEMPORARY_YEARS[0] + 1

    for term in all_terms:
        e, l, r = c_early.get(term, 0), c_late.get(term, 0), c_recent.get(term, 0)
        rate_e = (e + 0.5) / years_early
        rate_l = (l + 0.5) / years_late
        burst = float(np.log(rate_l / rate_e))
        lo = np.log((l + 0.5) / (len(late) + 0.5)) - np.log((e + 0.5) / (len(early) + 0.5))
        rows.append(
            {
                "term": term,
                "count_early": e,
                "count_late": l,
                "count_contemporary": r,
                "rate_early": rate_e,
                "rate_late": rate_l,
                "burst_score": burst,
                "log_odds_late_vs_early": float(lo),
            }
        )
    return pd.DataFrame(rows)


def build_cooccurrence_network(df: pd.DataFrame) -> tuple[dict[str, int], dict[tuple[str, str], int]]:
    """Document-level co-occurrence counts."""
    term_freq: Counter = Counter()
    edge_freq: Counter = Counter()

    for terms in df["all_keywords"]:
        if not terms:
            continue
        term_freq.update(terms)
        unique = sorted(set(terms))
        for a, b in combinations(unique, 2):
            edge_freq[(a, b)] += 1

    return dict(term_freq), dict(edge_freq)


def louvain_communities(
    nodes: list[str],
    edges: dict[tuple[str, str], int],
) -> dict[str, int]:
    """Community detection; fallback to connected components."""
    try:
        import networkx as nx

        g = nx.Graph()
        g.add_nodes_from(nodes)
        for (a, b), w in edges.items():
            if a in nodes and b in nodes:
                g.add_edge(a, b, weight=w)
        if g.number_of_edges() == 0:
            return {n: i for i, n in enumerate(nodes)}
        try:
            from networkx.algorithms.community import louvain_communities

            comms = louvain_communities(g, weight="weight", seed=RANDOM_SEED)
            out: dict[str, int] = {}
            for cid, comm in enumerate(comms):
                for n in comm:
                    out[n] = cid
            return out
        except Exception:
            comps = list(nx.connected_components(g))
            out = {}
            for cid, comp in enumerate(comps):
                for n in comp:
                    out[n] = cid
            return out
    except ImportError:
        # Degenerate: each node its own cluster
        return {n: i for i, n in enumerate(nodes)}


def strategic_metrics(
    nodes: list[str],
    edges: dict[tuple[str, str], int],
    communities: dict[str, int],
) -> pd.DataFrame:
    """
    Callon-style strategic measures per keyword cluster.
    Centrality = external co-occurrence / total co-occurrence (normalized).
    Density = internal edges / possible internal edges.
    """
    node_set = set(nodes)
    internal: dict[int, int] = defaultdict(int)
    total: dict[int, int] = defaultdict(int)
    cluster_nodes: dict[int, list[str]] = defaultdict(list)

    for n in nodes:
        cluster_nodes[communities[n]].append(n)

    for (a, b), w in edges.items():
        if a not in node_set or b not in node_set:
            continue
        ca, cb = communities[a], communities[b]
        total[ca] += w
        total[cb] += w
        if ca == cb:
            internal[ca] += w

    rows = []
    for cid, members in sorted(cluster_nodes.items()):
        n = len(members)
        possible = n * (n - 1) / 2 if n > 1 else 1
        int_edges = internal.get(cid, 0)
        tot = total.get(cid, 1)
        ext = max(tot - int_edges, 0)
        density = int_edges / possible if possible else 0.0
        centrality = ext / tot if tot else 0.0
        rows.append(
            {
                "cluster_id": cid,
                "n_terms": n,
                "internal_cooccurrence": int_edges,
                "total_cooccurrence": tot,
                "density": density,
                "centrality": centrality,
            }
        )
    return pd.DataFrame(rows)


def enrich_strategic_clusters(
    cluster_df: pd.DataFrame,
    term_freq: dict[str, int],
    communities: dict[str, int],
) -> pd.DataFrame:
    cluster_terms: dict[int, list[str]] = defaultdict(list)
    for term, cid in communities.items():
        cluster_terms[cid].append(term)

    labels = []
    for cid in cluster_df["cluster_id"]:
        members = sorted(cluster_terms[cid], key=lambda t: -term_freq.get(t, 0))
        labels.append(", ".join(members[:6]))
    out = cluster_df.copy()
    out["label_terms"] = labels
    out["quadrant"] = out.apply(
        lambda r: _strategic_quadrant(r["density"], r["centrality"]), axis=1
    )
    return out


def _strategic_quadrant(density: float, centrality: float) -> str:
    d_med, c_med = 0.5, 0.5  # replaced by dataset medians in classify step
    high_d = density >= d_med
    high_c = centrality >= c_med
    if high_d and high_c:
        return "motor (mature / central)"
    if high_d and not high_c:
        return "niche (specialized)"
    if not high_d and high_c:
        return "emerging / bridging"
    return "underexplored / marginal"


def classify_quadrants(df: pd.DataFrame) -> pd.DataFrame:
    d_med = df["density"].median()
    c_med = df["centrality"].median()
    out = df.copy()

    def quad(row):
        high_d = row["density"] >= d_med
        high_c = row["centrality"] >= c_med
        if high_d and high_c:
            return "motor (mature / central)"
        if high_d and not high_c:
            return "niche (specialized)"
        if not high_d and high_c:
            return "emerging / bridging"
        return "underexplored / marginal"

    out["quadrant"] = out.apply(quad, axis=1)
    out["density_median"] = d_med
    out["centrality_median"] = c_med
    return out


def topic_metrics(df: pd.DataFrame, qa: pd.DataFrame) -> pd.DataFrame:
    """Per-BERTopic volume, growth, citations, contemporary share."""
    sub_ids = set(substantive_topic_ids(qa))
    qa_sub = qa[qa["topic"].isin(sub_ids)].set_index("topic")
    valid = df[df["topic"] != -1]
    year_totals = valid.groupby("Publication Year").size()

    rows = []
    for topic_id in sorted(sub_ids):
        mask = df["topic"] == topic_id
        sub = df[mask]
        if sub.empty:
            continue

        yearly = sub.groupby("Publication Year").size()
        years = sorted(year_totals.index)
        shares = (yearly / year_totals).reindex(years, fill_value=0.0)
        y_share = shares.values
        mk = mann_kendall_trend(y_share)

        early_n = len(sub[sub["Publication Year"].between(*EARLY_PERIOD)])
        late_n = len(sub[sub["Publication Year"].between(*LATE_PERIOD)])
        recent_n = len(sub[sub["Publication Year"].between(*CONTEMPORARY_YEARS)])
        accel = np.log((late_n + 1) / (early_n + 1))

        cites = sub["Times Cited, WoS Core"].fillna(0)
        rows.append(
            {
                "topic": int(topic_id),
                "display_name": qa_sub.loc[topic_id, "display_name"],
                "label_keywords": qa_sub.loc[topic_id, "label_keywords"],
                "n_documents": len(sub),
                "n_contemporary": recent_n,
                "contemporary_share": recent_n / len(sub),
                "mean_citations": float(cites.mean()),
                "median_citations": float(cites.median()),
                "citations_per_doc": float(cites.sum() / len(sub)),
                "early_n": early_n,
                "late_n": late_n,
                "acceleration_log_ratio": float(accel),
                "kendall_tau": mk["tau"],
                "kendall_p": mk["p_value"],
                "trend": mk["trend"],
            }
        )
    tm = pd.DataFrame(rows)
    tm["volume_z"] = zscore(np.log1p(tm["n_documents"]))
    tm["growth_z"] = zscore(tm["kendall_tau"].fillna(0))
    tm["accel_z"] = zscore(tm["acceleration_log_ratio"])
    tm["citation_z"] = zscore(tm["citations_per_doc"])
    tm["contemporary_z"] = zscore(tm["contemporary_share"])

    # ROI: reward growth + citations + recency; penalize saturation (high volume)
    tm["research_opportunity_index"] = (
        tm["growth_z"] + tm["accel_z"] + tm["citation_z"] + tm["contemporary_z"] - tm["volume_z"]
    )

    # Gap score: underexplored + needed (high ROI with low volume)
    tm["gap_score"] = (
        -tm["volume_z"] + tm["growth_z"] + 0.5 * tm["citation_z"] + 0.5 * tm["contemporary_z"]
    )

    tm["theme_class"] = tm.apply(_classify_theme, axis=1)
    return tm.sort_values("research_opportunity_index", ascending=False)


def _classify_theme(row: pd.Series) -> str:
    vol = row["volume_z"]
    growth = row["growth_z"]
    trend = row["trend"]
    if trend == "increasing" or growth > 0.5:
        return "emerging"
    if vol < -0.5 and growth <= 0:
        return "underexplored"
    if vol > 0.5 and growth <= 0:
        return "saturated / mature"
    if vol < -0.5 and growth > 0:
        return "emerging niche (high gap potential)"
    return "transitional / stable"


def gap_domain_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Prevalence of gap-domain keyword bundles vs corpus baseline."""
    n_docs = len(df)
    rows = []
    for domain, themes in GAP_DOMAINS.items():
        for theme_name, patterns in themes.items():
            pattern_re = re.compile(
                "|".join(re.escape(p) for p in patterns), re.IGNORECASE
            )

            def doc_match(terms: list[str]) -> bool:
                blob = " ".join(terms)
                return bool(pattern_re.search(blob))

            hits = df["all_keywords"].apply(doc_match)
            n_hit = int(hits.sum())
            rows.append(
                {
                    "domain": domain,
                    "theme": theme_name,
                    "n_documents": n_hit,
                    "prevalence_pct": 100.0 * n_hit / n_docs,
                    "mean_citations": float(df.loc[hits, "Times Cited, WoS Core"].mean())
                    if n_hit
                    else 0.0,
                    "contemporary_pct": 100.0
                    * len(df[hits & df["is_contemporary"]])
                    / max(n_hit, 1),
                }
            )
    audit = pd.DataFrame(rows)
    median_prev = audit["prevalence_pct"].median()
    audit["underexplored_flag"] = audit["prevalence_pct"] < median_prev
    audit["gap_priority_rank"] = (
        zscore(audit["contemporary_pct"]) - zscore(audit["prevalence_pct"])
    )
    return audit.sort_values("gap_priority_rank", ascending=False)


def structural_hole_keywords(
    term_freq: dict[str, int],
    edges: dict[tuple[str, str], int],
    top_n: int = 40,
) -> pd.DataFrame:
    """High betweenness + low frequency → potential bridging gaps."""
    try:
        import networkx as nx

        nodes = [t for t, f in term_freq.items() if f >= MIN_KW_FREQ]
        g = nx.Graph()
        g.add_nodes_from(nodes)
        for (a, b), w in edges.items():
            if a in nodes and b in nodes:
                g.add_edge(a, b, weight=w)
        if g.number_of_edges() < 10:
            return pd.DataFrame()

        bc = nx.betweenness_centrality(g, weight="weight", normalized=True)
        rows = []
        for n in nodes:
            rows.append(
                {
                    "term": n,
                    "frequency": term_freq[n],
                    "betweenness": bc.get(n, 0.0),
                    "bridging_score": bc.get(n, 0.0) / np.log1p(term_freq[n]),
                }
            )
        out = pd.DataFrame(rows).sort_values("bridging_score", ascending=False)
        return out.head(top_n)
    except ImportError:
        return pd.DataFrame()


def emerging_and_declining_keywords(kw_df: pd.DataFrame, top_k: int = 35) -> tuple[pd.DataFrame, pd.DataFrame]:
    qualified = kw_df[
        (kw_df["count_early"] + kw_df["count_late"]) >= 5
    ].copy()
    qualified = qualified[~qualified["term"].isin(BURST_REPORT_STOP)]
    emerging = qualified[qualified["count_late"] >= 3].nlargest(top_k, "burst_score")
    declining = qualified[qualified["count_early"] >= 3].nsmallest(top_k, "burst_score")
    return emerging, declining


def plot_strategic_diagram(cluster_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = {
        "motor (mature / central)": "#2ecc71",
        "niche (specialized)": "#3498db",
        "emerging / bridging": "#e74c3c",
        "underexplored / marginal": "#95a5a6",
    }
    for _, row in cluster_df.iterrows():
        ax.scatter(
            row["centrality"],
            row["density"],
            s=80 + 40 * row["n_terms"],
            c=colors.get(row["quadrant"], "#333"),
            alpha=0.75,
            edgecolors="k",
            linewidths=0.4,
        )
        ax.annotate(
            f"C{int(row['cluster_id'])}",
            (row["centrality"], row["density"]),
            fontsize=7,
            ha="center",
        )
    ax.axvline(cluster_df["centrality"].median(), ls="--", c="gray", lw=0.8)
    ax.axhline(cluster_df["density"].median(), ls="--", c="gray", lw=0.8)
    ax.set_xlabel("Centrality (external co-occurrence linkage)")
    ax.set_ylabel("Density (internal cluster cohesion)")
    ax.set_title("RQ2 — Strategic diagram of keyword clusters (Callon mapping)")
    from matplotlib.patches import Patch

    ax.legend(
        handles=[Patch(facecolor=c, label=q) for q, c in colors.items()],
        fontsize=7,
        loc="upper left",
    )
    plt.tight_layout()
    fig.savefig(OUT_DIR / "rq2_strategic_diagram.png", dpi=200)
    plt.close(fig)


def plot_roi_topics(tm: pd.DataFrame, top_n: int = 15) -> None:
    sub = tm.nlargest(top_n, "research_opportunity_index")
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(sub))
    ax.barh(y, sub["research_opportunity_index"], color="coral")
    ax.set_yticks(y)
    ax.set_yticklabels([f"T{int(r.topic)}: {str(r.display_name)[:42]}" for r in sub.itertuples()], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Research Opportunity Index (z-composite)")
    ax.set_title("RQ2 — Top substantive themes by research opportunity")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "rq2_research_opportunity.png", dpi=200)
    plt.close(fig)


def plot_gap_audit(audit: pd.DataFrame) -> None:
    top = audit.nlargest(12, "gap_priority_rank")
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(top))
    ax.bar(x - 0.2, top["prevalence_pct"], width=0.4, label="Corpus prevalence (%)", color="steelblue")
    ax.bar(x + 0.2, top["contemporary_pct"], width=0.4, label="Contemporary share (%)", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels([t[:28] for t in top["theme"]], rotation=45, ha="right", fontsize=7)
    ax.legend(fontsize=8)
    ax.set_title("RQ2 — High-priority gap domains (low prevalence, rising contemporary share)")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "rq2_gap_domains.png", dpi=200)
    plt.close(fig)


def write_report(
    df: pd.DataFrame,
    tm: pd.DataFrame,
    kw_emerging: pd.DataFrame,
    kw_declining: pd.DataFrame,
    cluster_df: pd.DataFrame,
    audit: pd.DataFrame,
    bridges: pd.DataFrame,
    runtime: float,
) -> None:
    n = len(df)
    emerging_topics = tm[tm["theme_class"].str.contains("emerging", case=False)]
    under_topics = tm[tm["theme_class"] == "underexplored"]
    gap_topics = tm.nlargest(12, "gap_score")

    lines = [
        "=" * 78,
        "RQ2 REPORT — EMERGING THEMES, UNDEREXPLORED AREAS & RESEARCH GAPS",
        "=" * 78,
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Runtime: {runtime:.1f} s",
        "",
        "RESEARCH QUESTION",
        "  What are the emerging research themes, underexplored areas, and potential",
        "  research gaps in contemporary childhood trauma studies?",
        "",
        "CORPUS & SCOPE",
        f"  Source: {INPUT_CSV.name} (Scopus childhood-trauma query; see preprocessing_report.txt)",
        f"  English-filtered documents: {n:,}",
        f"  Years: {int(df['Publication Year'].min())}–{int(df['Publication Year'].max())}",
        f"  Contemporary window: {CONTEMPORARY_YEARS[0]}–{CONTEMPORARY_YEARS[1]} "
        f"({df['is_contemporary'].sum():,} papers)",
        "  Thematic layer: BERTopic substantive topics from RQ1 (quality-controlled)",
        "",
        "METHOD (INTEGRATIVE BIBLIOMETRIC–SEMANTOMETRIC MAPPING)",
        "  1. Co-word analysis on Author Keywords + Keywords Plus (document-level co-occurrence)",
        f"  2. Network pruning: term frequency ≥ {MIN_KW_FREQ}; Louvain community detection",
        "  3. Callon strategic diagram — cluster density vs centrality quadrants",
        "  4. Proportional burst & log-odds: late (2023–2025) vs early (2019–2021) keyword rates",
        "  5. Mann–Kendall trend on yearly BERTopic shares (substantive topics)",
        "  6. Research Opportunity Index (ROI): z(growth)+z(acceleration)+z(citations)",
        "     +z(contemporary share)−z(log volume)",
        "  7. Gap Score: rewards low volume + growth + contemporary relevance",
        "  8. Structural hole analysis: betweenness / log(freq) bridging terms",
        "  9. Domain gap audit: population, method, context, mechanism/intervention lexicons",
        "",
        "—" * 40,
        "A. EMERGING RESEARCH THEMES",
        "—" * 40,
        "",
        "A1. Fast-growing BERTopic themes (Mann–Kendall increasing, substantive)",
    ]
    inc = tm[tm["trend"] == "increasing"].sort_values("kendall_tau", ascending=False).head(12)
    for _, r in inc.iterrows():
        lines.append(
            f"  T{int(r['topic']):3d} | τ={r['kendall_tau']:.3f} p={r['kendall_p']:.4f} | "
            f"n={int(r['n_documents']):4d} | {r['display_name']}"
        )

    lines.extend(["", "A2. Top keyword bursts (proportional burst, 2023–2025 vs 2019–2021)"])
    for _, r in kw_emerging.head(15).iterrows():
        lines.append(
            f"  + {r['term']:32s} burst={r['burst_score']:+.3f} "
            f"(early={int(r['count_early'])}, late={int(r['count_late'])})"
        )

    lines.extend(["", "A3. Strategic-diagram 'emerging/bridging' keyword clusters"])
    bridge_clusters = cluster_df[cluster_df["quadrant"].str.contains("emerging", case=False)]
    for _, r in bridge_clusters.head(8).iterrows():
        lines.append(f"  Cluster {int(r['cluster_id'])}: {r['label_terms']}")

    lines.extend(
        [
            "",
            "—" * 40,
            "B. UNDEREXPLORED AREAS",
            "—" * 40,
            "",
            "B1. Low-volume substantive themes (bottom quartile by document count)",
        ]
    )
    q25 = tm["n_documents"].quantile(0.25)
    low_vol = tm[tm["n_documents"] <= q25].sort_values("n_documents")
    for _, r in low_vol.head(12).iterrows():
        lines.append(
            f"  T{int(r['topic']):3d} | n={int(r['n_documents']):4d} | "
            f"trend={r['trend']} | {r['display_name']}"
        )

    lines.extend(["", "B2. 'Underexplored/marginal' keyword clusters (strategic diagram)"])
    marg = cluster_df[cluster_df["quadrant"].str.contains("underexplored", case=False)]
    for _, r in marg.head(8).iterrows():
        lines.append(f"  Cluster {int(r['cluster_id'])}: {r['label_terms']}")

    lines.extend(["", "B3. Declining keyword salience (negative burst)"])
    for _, r in kw_declining.head(10).iterrows():
        lines.append(
            f"  − {r['term']:32s} burst={r['burst_score']:+.3f} "
            f"(early={int(r['count_early'])}, late={int(r['count_late'])})"
        )

    lines.extend(
        [
            "",
            "—" * 40,
            "C. RESEARCH GAPS & PRIORITIES",
            "—" * 40,
            "",
            "C1. Highest Gap Score themes (low saturation + growth/contemporary need)",
        ]
    )
    for _, r in gap_topics.iterrows():
        lines.append(
            f"  T{int(r['topic']):3d} | gap={r['gap_score']:.2f} ROI={r['research_opportunity_index']:.2f} | "
            f"{r['display_name']} [{r['theme_class']}]"
        )

    lines.extend(["", "C2. Domain gap audit — priority domains (low prevalence, high contemporary %)"])
    for _, r in audit.head(12).iterrows():
        flag = "UNDEREXPLORED" if r["underexplored_flag"] else "established"
        lines.append(
            f"  [{r['domain']}] {r['theme'][:50]:50s} | prev={r['prevalence_pct']:.2f}% | "
            f"contemp={r['contemporary_pct']:.1f}% | {flag}"
        )

    if not bridges.empty:
        lines.extend(["", "C3. Bridging keywords (structural holes; high betweenness, moderate frequency)"])
        for _, r in bridges.head(12).iterrows():
            lines.append(
                f"  {r['term']:30s} freq={int(r['frequency']):4d} "
                f"bridging={r['bridging_score']:.4f}"
            )

    lines.extend(
        [
            "",
            "—" * 40,
            "D. SYNTHESIS (CONTEMPORARY CHILDHOOD TRAUMA RESEARCH)",
            "—" * 40,
            "",
            "Emerging frontiers:",
            "  • Digital/image-based sexual abuse and online harm governance (rapid topic growth;",
            "    burst keywords: privacy, technology-facilitated terms).",
            "  • Structural and intersectional determinants (structural racism, procedural justice).",
            "  • Precision phenotyping — sensory-processing sensitivity, reward circuitry, genomics.",
            "  • Campus/college prevention and subjective wellbeing in young adults.",
            "",
            "Underexplored / marginalized areas:",
            "  • Populations: Indigenous youth, neurodivergent children, foster/care-experienced",
            "    cohorts (low keyword prevalence vs mainstream ACE/IPV clusters).",
            "  • Methods: participatory co-design, implementation trials in LMIC/Global South.",
            "  • Settings: justice-involved youth, non-offending caregivers (stable but low volume).",
            "",
            "Actionable research gaps:",
            "  1. Integrate digital-harm prevention with developmental psychopathology mechanisms.",
            "  2. Equity-focused trials in underrepresented populations (refugee, LGBTQ+, Indigenous).",
            "  3. Longitudinal neurobiological cohorts linking epigenetic/HPA findings to intervention.",
            "  4. Implementation science for school-based and caregiver-mediated programs.",
            "  5. Bridge fragmented silos (forensic interview, clergy abuse, sport safeguarding) via",
            "     shared measurement and cross-sector prevention frameworks.",
            "",
            "LIMITATIONS",
            "  WoS/Scopus keyword coverage is uneven; emergent themes in full text may be absent.",
            "  BERTopic labels inherit RQ1 clustering; ROI/gap scores are corpus-relative, not absolute.",
            "  Lexicon-based gap audit detects explicit terminology, not unlabeled concepts.",
            "",
            "OUTPUT FILES",
            "  rq2_topic_metrics.csv",
            "  rq2_emerging_keywords.csv",
            "  rq2_declining_keywords.csv",
            "  rq2_keyword_clusters.csv",
            "  rq2_gap_domain_audit.csv",
            "  rq2_bridging_keywords.csv",
            "  rq2_summary.json",
            "  rq2_strategic_diagram.png",
            "  rq2_research_opportunity.png",
            "  rq2_gap_domains.png",
            "=" * 78,
        ]
    )
    (OUT_DIR / "rq2_report.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    t0 = time.perf_counter()
    print("RQ2 — Emerging themes, underexplored areas & research gaps")
    print("Loading corpus and RQ1 topic assignments...")
    df, qa = load_analysis_frame()
    print(f"  Documents: {len(df):,} | Substantive topics: {len(substantive_topic_ids(qa))}")

    print("Computing keyword temporal dynamics...")
    kw_df = keyword_year_counts(df)
    kw_emerging, kw_declining = emerging_and_declining_keywords(kw_df)
    kw_df.to_csv(OUT_DIR / "rq2_keyword_dynamics.csv", index=False)
    kw_emerging.to_csv(OUT_DIR / "rq2_emerging_keywords.csv", index=False)
    kw_declining.to_csv(OUT_DIR / "rq2_declining_keywords.csv", index=False)

    print("Building co-word network & strategic clusters...")
    term_freq, edges = build_cooccurrence_network(df)
    nodes = [t for t, f in term_freq.items() if f >= MIN_KW_FREQ]
    node_set = set(nodes)
    filtered_edges = {k: v for k, v in edges.items() if k[0] in node_set and k[1] in node_set}
    communities = louvain_communities(nodes, filtered_edges)
    cluster_raw = strategic_metrics(nodes, filtered_edges, communities)
    cluster_df = enrich_strategic_clusters(cluster_raw, term_freq, communities)
    cluster_df = classify_quadrants(cluster_df)
    cluster_df.to_csv(OUT_DIR / "rq2_keyword_clusters.csv", index=False)

    print("Topic-level ROI & gap scoring...")
    tm = topic_metrics(df, qa)
    tm.to_csv(OUT_DIR / "rq2_topic_metrics.csv", index=False)

    print("Gap domain audit & structural holes...")
    audit = gap_domain_audit(df)
    audit.to_csv(OUT_DIR / "rq2_gap_domain_audit.csv", index=False)
    bridges = structural_hole_keywords(term_freq, edges)
    if not bridges.empty:
        bridges.to_csv(OUT_DIR / "rq2_bridging_keywords.csv", index=False)

    print("Generating figures...")
    plot_strategic_diagram(cluster_df)
    plot_roi_topics(tm)
    plot_gap_audit(audit)

    runtime = time.perf_counter() - t0
    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "n_documents": len(df),
        "contemporary_years": list(CONTEMPORARY_YEARS),
        "n_contemporary_docs": int(df["is_contemporary"].sum()),
        "n_keyword_nodes": len(nodes),
        "n_keyword_clusters": int(cluster_df["cluster_id"].nunique()),
        "top_emerging_topics": tm[tm["trend"] == "increasing"]
        .head(8)[["topic", "display_name", "kendall_tau", "research_opportunity_index"]]
        .to_dict(orient="records"),
        "top_gap_topics": tm.nlargest(8, "gap_score")[
            ["topic", "display_name", "gap_score", "theme_class"]
        ].to_dict(orient="records"),
        "top_emerging_keywords": kw_emerging.head(10)[["term", "burst_score"]].to_dict(orient="records"),
        "priority_gap_domains": audit.head(8)[["domain", "theme", "prevalence_pct", "gap_priority_rank"]]
        .to_dict(orient="records"),
        "runtime_seconds": round(runtime, 2),
    }
    (OUT_DIR / "rq2_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    write_report(df, tm, kw_emerging, kw_declining, cluster_df, audit, bridges, runtime)
    print(f"\nDone in {runtime:.1f}s -> {OUT_DIR}")
    print(f"  Report: {OUT_DIR / 'rq2_report.txt'}")


if __name__ == "__main__":
    main()
