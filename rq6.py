"""
RQ6 — Dominant research contexts and thematic clusters (2020–2025): LDA + noun phrases.

Research question:
  What are the dominant research contexts and thematic clusters in peer-reviewed
  childhood trauma literature across 2020–2025 as revealed by LDA topic modelling
  and noun-phrase extraction?

Pipeline:
  1. English-only corpus (rq1.filter_english_corpus)
  2. Publication years 2020–2025 inclusive
  3. Peer-reviewed filter on Document Type (Articles + Reviews; excludes meeting abstracts, editorials)
  4. LDA on CountVectorizer bag-of-words (preprocessed_text)
  5. Noun-phrase extraction from title+abstract (fast noun-headed heuristic; optional NLTK subsample)
  6. Thematic clusters = LDA topics; labels from top terms + top NPs per topic
  7. Research contexts = WoS/Scopus Research Areas (and Keywords Plus summary)
"""

from __future__ import annotations

import json
import re
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

warnings.filterwarnings("ignore", category=FutureWarning)

from rq1 import filter_english_corpus

BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "data_preprocessed.csv"
OUT_DIR = BASE_DIR / "rq6_output"
OUT_DIR.mkdir(exist_ok=True)

YEAR_START, YEAR_END = 2020, 2025
N_TOPICS = 32
TOP_TERMS = 12
TOP_NPS_TOPIC = 15
TOP_NPS_GLOBAL = 40
NP_TEXT_CHARS = 1200
MAX_NPS_PER_DOC = 40
MIN_NP_LEN = 2
MAX_NP_TOKENS = 5
RANDOM_SEED = 42

NOUN_SUFFIXES = (
    "tion", "sion", "ment", "ness", "ity", "ism", "ist", "ence", "ance",
    "ing", "age", "ure", "dom", "ship", "hood",
)

# Generic NP fragments to exclude from dominance tables (methods boilerplate / empty shells)
GENERIC_NP_PREFIXES = (
    "the ", "and ", "this ", "with ", "for ", "our ", "their ", "these ",
    "those ", "that ", "from ", "into ", "upon ",
)
GENERIC_NP_EXACT = {
    "logistic regression", "structural equation", "structural equation modeling",
    "the relationship", "the development", "the mediating", "the importance",
    "the influence", "the presence", "the experience", "the quality",
    "the nature", "the majority", "and setting", "and depression", "and violence",
}

# Macro-tradition labels from LDA top terms (interpretive, transparent rules)
MACRO_THEME_RULES: list[tuple[str, list[str]]] = [
    ("Policy, services & child protection systems", ["policy", "protection", "worker", "healthcare", "law", "service", "state"]),
    ("Qualitative care & lived experience", ["qualitative", "interview", "lived", "approach", "people", "class"]),
    ("Attachment, emotion regulation & relationships", ["attachment", "emotion regulation", "mediate", "mediation", "relationship"]),
    ("Intimate partner & domestic violence", ["ipv", "intimate partner", "domestic violence", "partner violence"]),
    ("ACE & epidemiological adversity", ["ace", "adverse experience", "aor", "household", "regression", "exposure"]),
    ("Preclinical stress biology (animal models)", ["rat", "mouse", "maternal separation", "cortisol", "hpa"]),
    ("PTSD, CSA & post-traumatic psychopathology", ["ptsd", "posttraumatic", "csa", "traumatic", "trauma"]),
    ("Family, maternal & infant development", ["family", "mother", "infant", "caregiver", "intergenerational"]),
    ("Depression, anxiety & COVID-19 mental health", ["depression", "anxiety", "covid", "pandemic", "resilience"]),
    ("Neurocognitive threat & memory", ["memory", "executive", "threat", "deprivation", "psychopathology"]),
    ("Education & prevention (students/teachers)", ["student", "education", "teacher", "prevention", "school"]),
    ("Epigenetics & HPA axis", ["methylation", "epigenetic", "hpa", "dna methylation", "gene"]),
    ("Pediatric injury & obesity", ["injury", "pediatric", "obesity", "fracture", "infant"]),
    ("Substance use, pain & HIV", ["substance", "alcohol", "opioid", "hiv", "cannabis"]),
    ("Suicide & self-harm", ["suicide", "suicidal", "ideation", "nssi"]),
    ("Forensic disclosure & digital abuse", ["forensic", "disclosure", "medium", "metoo", "perpetrator"]),
    ("Psychosis & dissociation", ["psychosis", "schizophrenia", "dissociation", "dissociative"]),
]


def load_stopwords() -> set[str]:
    path = BASE_DIR / "manual_stopwords.txt"
    if not path.exists():
        return set()
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().lower()
        if line and not line.startswith("#"):
            out.add(line)
    return out


STOPWORDS = load_stopwords()


def load_corpus() -> pd.DataFrame:
    usecols = [
        "UT (Unique WOS ID)",
        "Article Title",
        "Abstract",
        "preprocessed_text",
        "Publication Year",
        "Language",
        "Document Type",
        "Author Keywords",
        "Keywords Plus",
        "Research Areas",
    ]
    df = pd.read_csv(INPUT_CSV, usecols=usecols, low_memory=False)
    df = df.dropna(subset=["preprocessed_text", "Publication Year"])
    df = df[df["preprocessed_text"].str.strip().astype(bool)]
    df["Publication Year"] = df["Publication Year"].astype(int)
    return df.reset_index(drop=True)


def is_peer_reviewed(document_type) -> bool:
    """Conservative WoS/Scopus-style filter for empirical peer-reviewed types."""
    s = str(document_type).lower().strip()
    if not s or s == "nan":
        return True  # keep if missing
    if "meeting abstract" in s:
        return False
    if "editorial" in s and "article" not in s:
        return False
    if "news" in s and "article" not in s:
        return False
    if "book review" in s and "article" not in s:
        return False
    # Primary peer-reviewed forms
    if "article" in s or "review" in s:
        return True
    if "proceedings paper" in s or "conference paper" in s:
        return True
    return False


def _looks_noun_like(token: str) -> bool:
    if len(token) < 4:
        return False
    return any(token.endswith(s) for s in NOUN_SUFFIXES)


def extract_noun_phrases_fast(text: str) -> list[str]:
    if not isinstance(text, str) or len(text.strip()) < 10:
        return []
    text = re.sub(r"[^a-zA-Z0-9\s\-]", " ", text.lower())
    text = re.sub(r"\s+", " ", text).strip()[:NP_TEXT_CHARS]
    words = [w for w in text.split() if w not in STOPWORDS and len(w) > 2 and w.isalpha()]
    if len(words) < 2:
        return []
    seen: set[str] = set()
    phrases: list[str] = []
    for n in range(MIN_NP_LEN, MAX_NP_TOKENS + 1):
        for i in range(len(words) - n + 1):
            chunk = words[i : i + n]
            if not _looks_noun_like(chunk[-1]):
                continue
            phrase = " ".join(chunk)
            if phrase not in seen:
                seen.add(phrase)
                phrases.append(phrase)
            if len(phrases) >= MAX_NPS_PER_DOC:
                return phrases
    return phrases


def parse_research_areas(val) -> list[str]:
    if not isinstance(val, str) or not val.strip():
        return []
    parts = re.split(r"[;,]", val)
    return [p.strip() for p in parts if len(p.strip()) > 2]


def lda_top_terms(lda: LatentDirichletAllocation, feature_names: np.ndarray, top_n: int) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for k in range(lda.n_components):
        idx = lda.components_[k].argsort()[::-1][:top_n]
        out[k] = [str(feature_names[i]) for i in idx]
    return out


def auto_label(terms: list[str]) -> str:
    return ", ".join(terms[:4]).title()


def is_generic_np(phrase: str) -> bool:
    p = phrase.strip().lower()
    if not p or p in GENERIC_NP_EXACT:
        return True
    return any(p.startswith(pref) for pref in GENERIC_NP_PREFIXES)


def filter_np_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if not is_generic_np(r["phrase"])]


def assign_macro_theme(label_terms: str) -> str:
    joined = label_terms.lower()
    best_name, best_hits = "Other / mixed", 0
    for name, keys in MACRO_THEME_RULES:
        hits = sum(1 for k in keys if k in joined)
        if hits > best_hits:
            best_hits = hits
            best_name = name
    return best_name if best_hits >= 2 else "Other / mixed"


def plot_year_heatmap(yx_counts: pd.DataFrame, topic_df: pd.DataFrame, out_path: Path) -> None:
    top_topics = topic_df.sort_values("n_documents", ascending=False).head(12)["topic"].tolist()
    sub = yx_counts[yx_counts["topic"].isin(top_topics)].copy()
    pivot = sub.pivot_table(index="topic", columns="Publication Year", values="count", fill_value=0)
    pivot = pivot.reindex(top_topics)
    name_map = topic_df.set_index("topic")["display_label"].to_dict()

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([int(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"T{t}: {name_map.get(t, '')[:35]}" for t in pivot.index], fontsize=8)
    ax.set_xlabel("Year")
    ax.set_title("RQ6 — Top LDA clusters by year (2020–2025)")
    plt.colorbar(im, ax=ax, label="Documents")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def postprocess_outputs() -> None:
    """Regenerate filtered NP tables, cluster profiles, heatmap, synthesis (no LDA refit)."""
    topic_df = pd.read_csv(OUT_DIR / "rq6_lda_topics.csv")
    np_topic = pd.read_csv(OUT_DIR / "rq6_topic_noun_phrases.csv")
    np_global = pd.read_csv(OUT_DIR / "rq6_global_noun_phrases.csv")
    yx = pd.read_csv(OUT_DIR / "rq6_year_topic_counts.csv")
    contexts = pd.read_csv(OUT_DIR / "rq6_research_contexts.csv")

    # Filtered global NPs
    g_rows = filter_np_rows(np_global.to_dict(orient="records"))
    pd.DataFrame(g_rows).to_csv(OUT_DIR / "rq6_global_noun_phrases_filtered.csv", index=False)

    # Cluster profiles: LDA terms + substantive NPs + macro-tradition
    profiles = []
    for _, row in topic_df.iterrows():
        tid = int(row["topic"])
        nps = np_topic[np_topic["topic"] == tid].sort_values("count", ascending=False)
        nps_f = [r["phrase"] for r in filter_np_rows(nps.to_dict(orient="records"))][:8]
        profiles.append(
            {
                "topic": tid,
                "display_label": row["display_label"],
                "n_documents": int(row["n_documents"]),
                "macro_tradition": assign_macro_theme(str(row["label_terms"])),
                "lda_top_terms": row["label_terms"],
                "top_noun_phrases": ", ".join(nps_f),
            }
        )
    prof_df = pd.DataFrame(profiles).sort_values("n_documents", ascending=False)
    prof_df.to_csv(OUT_DIR / "rq6_cluster_profiles.csv", index=False)

    # Macro-tradition aggregation
    macro = (
        prof_df.groupby("macro_tradition", as_index=False)
        .agg(n_topics=("topic", "count"), n_documents=("n_documents", "sum"))
        .sort_values("n_documents", ascending=False)
    )
    macro.to_csv(OUT_DIR / "rq6_macro_traditions.csv", index=False)

    plot_year_heatmap(yx, topic_df, OUT_DIR / "rq6_year_topic_heatmap.png")

    # Extended synthesis report section
    lines = [
        "",
        "=" * 78,
        "RQ6 SYNTHESIS — DOMINANT CONTEXTS & THEMATIC CLUSTERS (POST-PROCESSED)",
        "=" * 78,
        "",
        "SUBSTANTIVE NOUN PHRASES (generic shells removed)",
    ]
    for r in g_rows[:18]:
        lines.append(f"  {r['phrase'][:55]:55s}  n={r['count']}")

    lines.extend(["", "MACRO-TRADITIONS (LDA clusters grouped by top-term rules)"])
    for _, r in macro.iterrows():
        lines.append(f"  {r['macro_tradition']:45s} | topics={int(r['n_topics']):2d} | docs={int(r['n_documents']):5d}")

    lines.extend(["", "CLUSTER PROFILES (LDA + noun phrases)"])
    for _, r in prof_df.head(12).iterrows():
        lines.append(f"  T{int(r['topic']):2d} | {r['macro_tradition']}")
        lines.append(f"       LDA: {r['lda_top_terms']}")
        if r["top_noun_phrases"]:
            lines.append(f"       NP:  {r['top_noun_phrases']}")

    lines.extend(["", "DOMINANT RESEARCH CONTEXTS (Research Areas)"])
    for _, r in contexts.head(12).iterrows():
        lines.append(
            f"  {r['research_area'][:45]:45s} n={int(r['n_documents']):5d} "
            f"-> T{int(r['dominant_lda_topic'])}"
        )

    lines.append("=" * 78)
    (OUT_DIR / "rq6_synthesis.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"  Post-process done -> {OUT_DIR / 'rq6_synthesis.txt'}", flush=True)


def main() -> None:
    from tqdm import tqdm

    t0 = time.perf_counter()
    print("RQ6 — LDA + noun phrases (2020–2025, peer-reviewed)", flush=True)

    df_raw = load_corpus()
    df, removed = filter_english_corpus(df_raw)
    df = df[(df["Publication Year"] >= YEAR_START) & (df["Publication Year"] <= YEAR_END)]
    df = df[df["Document Type"].apply(is_peer_reviewed)]

    print(f"  After English + year + peer-review: {len(df):,} documents", flush=True)

    docs = df["preprocessed_text"].tolist()
    vec = CountVectorizer(
        max_features=12_000,
        min_df=8,
        max_df=0.92,
        ngram_range=(1, 2),
        dtype=np.float64,
    )
    print("  Vectorizing (CountVectorizer)...", flush=True)
    X = vec.fit_transform(docs)
    names = vec.get_feature_names_out()

    lda = LatentDirichletAllocation(
        n_components=N_TOPICS,
        max_iter=25,
        learning_method="online",
        batch_size=1024,
        random_state=RANDOM_SEED,
        n_jobs=1,
        evaluate_every=0,
    )
    print(f"  Fitting LDA (K={N_TOPICS})...", flush=True)
    doc_topic_dist = lda.fit_transform(X)
    doc_labels = doc_topic_dist.argmax(axis=1)

    terms_by_topic = lda_top_terms(lda, names, TOP_TERMS)

    # Save document-level output
    id_col = "UT (Unique WOS ID)"
    out_docs = df[[id_col, "Publication Year", "Document Type"]].copy()
    out_docs = out_docs.rename(columns={id_col: "UT"})
    out_docs["lda_topic"] = doc_labels
    out_docs["lda_probability"] = doc_topic_dist.max(axis=1)
    out_docs.to_csv(OUT_DIR / "rq6_document_topics.csv", index=False)

    # Topic table
    rows = []
    for k in range(N_TOPICS):
        rows.append(
            {
                "topic": k,
                "label_terms": ", ".join(terms_by_topic[k]),
                "display_label": auto_label(terms_by_topic[k]),
                "n_documents": int((doc_labels == k).sum()),
                "mean_prob": float(doc_topic_dist[:, k].mean()),
            }
        )
    topic_df = pd.DataFrame(rows).sort_values("n_documents", ascending=False)
    topic_df.to_csv(OUT_DIR / "rq6_lda_topics.csv", index=False)
    label_by_topic = topic_df.set_index("topic")["display_label"].to_dict()

    # Year x topic counts
    yx = df[["Publication Year"]].copy()
    yx["topic"] = doc_labels
    yx_counts = yx.groupby(["Publication Year", "topic"]).size().reset_index(name="count")
    yx_counts.to_csv(OUT_DIR / "rq6_year_topic_counts.csv", index=False)

    # Noun phrases
    print("  Noun phrase extraction...", flush=True)
    df["lexical_text"] = (
        df["Article Title"].fillna("").astype(str) + " " + df["Abstract"].fillna("").astype(str)
    ).str.slice(0, NP_TEXT_CHARS)
    nps_by_doc = [extract_noun_phrases_fast(t) for t in tqdm(df["lexical_text"], desc="NP", leave=False)]

    global_np: Counter = Counter()
    for nps in nps_by_doc:
        global_np.update(nps)

    topic_np: dict[int, Counter] = defaultdict(Counter)
    for nps, t in zip(nps_by_doc, doc_labels):
        topic_np[int(t)].update(nps)

    np_global_rows = [{"phrase": p, "count": c} for p, c in global_np.most_common(TOP_NPS_GLOBAL * 2)]
    pd.DataFrame(np_global_rows).to_csv(OUT_DIR / "rq6_global_noun_phrases.csv", index=False)
    np_global_filtered = filter_np_rows(np_global_rows)
    pd.DataFrame(np_global_filtered[:TOP_NPS_GLOBAL]).to_csv(
        OUT_DIR / "rq6_global_noun_phrases_filtered.csv", index=False
    )

    np_topic_rows = []
    for k in range(N_TOPICS):
        top = topic_np[k].most_common(TOP_NPS_TOPIC)
        for phrase, cnt in top:
            np_topic_rows.append({"topic": k, "phrase": phrase, "count": cnt})
    pd.DataFrame(np_topic_rows).to_csv(OUT_DIR / "rq6_topic_noun_phrases.csv", index=False)

    # Research Areas (dominant contexts)
    area_counter: Counter = Counter()
    area_topic: dict[str, Counter] = defaultdict(Counter)
    for areas, lab in zip(df["Research Areas"].tolist(), doc_labels):
        for a in parse_research_areas(areas):
            area_counter[a] += 1
            area_topic[a][int(lab)] += 1

    area_rows = []
    for area, tot in area_counter.most_common(80):
        top_t = area_topic[area].most_common(1)[0][0] if area_topic[area] else -1
        area_rows.append(
            {
                "research_area": area,
                "n_documents": tot,
                "dominant_lda_topic": int(top_t),
                "dominant_topic_label": label_by_topic.get(top_t, ""),
            }
        )
    pd.DataFrame(area_rows).to_csv(OUT_DIR / "rq6_research_contexts.csv", index=False)

    # Cluster profiles + macro-traditions
    profiles = []
    for _, row in topic_df.iterrows():
        tid = int(row["topic"])
        top_nps = [
            r["phrase"]
            for r in filter_np_rows(
                [
                    {"phrase": p, "count": c}
                    for p, c in topic_np[tid].most_common(TOP_NPS_TOPIC * 2)
                ]
            )
        ][:8]
        profiles.append(
            {
                "topic": tid,
                "display_label": row["display_label"],
                "n_documents": int(row["n_documents"]),
                "macro_tradition": assign_macro_theme(str(row["label_terms"])),
                "lda_top_terms": row["label_terms"],
                "top_noun_phrases": ", ".join(top_nps),
            }
        )
    prof_df = pd.DataFrame(profiles)
    prof_df.to_csv(OUT_DIR / "rq6_cluster_profiles.csv", index=False)
    prof_df.groupby("macro_tradition", as_index=False).agg(
        n_topics=("topic", "count"), n_documents=("n_documents", "sum")
    ).sort_values("n_documents", ascending=False).to_csv(OUT_DIR / "rq6_macro_traditions.csv", index=False)

    # Plot: topic sizes
    fig, ax = plt.subplots(figsize=(10, 6))
    sub = topic_df.sort_values("n_documents", ascending=True).tail(25)
    ax.barh(range(len(sub)), sub["n_documents"].values, color="steelblue")
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels([f"T{r.topic}: {r.display_label[:40]}" for r in sub.itertuples()], fontsize=8)
    ax.set_xlabel("Number of documents")
    ax.set_title("RQ6 — Largest LDA thematic clusters (2020–2025)")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "rq6_topic_sizes.png", dpi=200)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8, 7))
    topg = np_global_rows[:20]
    ax2.barh(range(len(topg)), [r["count"] for r in topg], color="darkslategray")
    ax2.set_yticks(range(len(topg)))
    ax2.set_yticklabels([r["phrase"][:45] for r in topg], fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("Frequency")
    ax2.set_title("RQ6 — Dominant noun phrases (title + abstract)")
    plt.tight_layout()
    fig2.savefig(OUT_DIR / "rq6_global_nps.png", dpi=200)
    plt.close(fig2)

    plot_year_heatmap(yx_counts, topic_df, OUT_DIR / "rq6_year_topic_heatmap.png")

    runtime = time.perf_counter() - t0

    # Report
    lines = [
        "=" * 78,
        "RQ6 REPORT — RESEARCH CONTEXTS & THEMATIC CLUSTERS (2020–2025)",
        "=" * 78,
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Runtime: {runtime:.1f} s",
        "",
        "CORPUS",
        f"  Source: {INPUT_CSV.name}",
        f"  English-filtered + years {YEAR_START}–{YEAR_END} + peer-reviewed types: {len(df):,} documents",
        f"  Removed non-English (from raw window): {len(removed):,}",
        "",
        "METHOD",
        f"  LDA: K={N_TOPICS}, CountVectorizer (max_features=12000, min_df=8, ngram 1-2)",
        "  Noun phrases: fast noun-headed phrases from title+abstract",
        "  Research contexts: Research Areas field (WoS/Scopus), ranked by frequency",
        "",
        "LARGEST LDA CLUSTERS (by document count)",
    ]
    for _, r in topic_df.head(15).iterrows():
        lines.append(f"  T{int(r['topic']):2d} | n={int(r['n_documents']):5d} | {r['display_label']}")
        lines.append(f"       Terms: {r['label_terms']}")

    lines.extend(["", "TOP GLOBAL NOUN PHRASES (substantive, filtered)"])
    for r in np_global_filtered[:20]:
        lines.append(f"  {r['phrase'][:55]:55s}  n={r['count']}")

    lines.extend(["", "MACRO-TRADITIONS (grouped LDA clusters)"])
    macro_tbl = pd.read_csv(OUT_DIR / "rq6_macro_traditions.csv")
    for _, r in macro_tbl.iterrows():
        lines.append(
            f"  {r['macro_tradition']:45s} | topics={int(r['n_topics']):2d} | docs={int(r['n_documents']):5d}"
        )

    lines.extend(["", "TOP RESEARCH AREAS (CONTEXTS)"])
    for r in area_rows[:15]:
        lines.append(
            f"  {r['research_area'][:50]:50s}  n={r['n_documents']:5d}  "
            f"→ dominant T{r['dominant_lda_topic']} ({r['dominant_topic_label'][:35]})"
        )

    lines.extend(
        [
            "",
            "OUTPUT FILES",
            "  rq6_document_topics.csv",
            "  rq6_lda_topics.csv",
            "  rq6_year_topic_counts.csv",
            "  rq6_global_noun_phrases.csv",
            "  rq6_topic_noun_phrases.csv",
            "  rq6_research_contexts.csv",
            "  rq6_topic_sizes.png",
            "  rq6_global_nps.png",
            "  rq6_year_topic_heatmap.png",
            "  rq6_cluster_profiles.csv",
            "  rq6_macro_traditions.csv",
            "  rq6_global_noun_phrases_filtered.csv",
            "=" * 78,
        ]
    )
    (OUT_DIR / "rq6_report.txt").write_text("\n".join(lines), encoding="utf-8")
    postprocess_outputs()  # writes rq6_synthesis.txt from saved tables

    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "n_documents": len(df),
        "years": [YEAR_START, YEAR_END],
        "n_topics": N_TOPICS,
        "top_topics": topic_df.head(8).to_dict(orient="records"),
        "top_global_nps": np_global_rows[:12],
        "top_research_areas": area_rows[:10],
        "runtime_seconds": round(runtime, 2),
    }
    (OUT_DIR / "rq6_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nDone in {runtime:.1f}s -> {OUT_DIR}", flush=True)
    print(f"  Report: {OUT_DIR / 'rq6_report.txt'}", flush=True)


if __name__ == "__main__":
    import sys

    if "--post" in sys.argv:
        postprocess_outputs()
    else:
        main()
