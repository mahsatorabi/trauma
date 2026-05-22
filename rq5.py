"""
RQ5 — Intellectual network of themes (macro-traditions / conceptual architecture).

Research question:
  How do the discovered themes relate to one another in a structured intellectual
  network, forming clusters or macro-traditions that define the discipline's
  conceptual architecture?

Approach (intellectual structure at theme level):
  1. Use RQ1 BERTopic assignments as the theme layer (substantive topics only).
  2. Primary: topic–reference bibliographic coupling (shared cited references).
  3. Fallback (if cited references absent): topic–keyword coupling using Author Keywords
     + Keywords Plus (shared conceptual vocabulary).
  4. Compute topic–topic similarity (cosine on term-count vectors).
  4. Build a sparse topic network: keep top-k strongest links per topic + min similarity.
  5. Detect macro-traditions via Louvain community detection.
  6. Characterize architecture:
     - within-cluster cohesion, between-cluster connectivity
     - centrality (strength, betweenness)
     - bridging themes (high betweenness, cross-cluster edges)
  7. Export tables, figures, and an academic report.

Why coupling?
  It operationalizes an "intellectual" relationship: themes are connected when their
  papers build on similar bodies of work (references) or share stable conceptual
  vocabularies (keywords), revealing macro-traditions.
"""

from __future__ import annotations

import json
import math
import re
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore", category=FutureWarning)

from rq1 import filter_english_corpus, substantive_topic_ids

BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "data_preprocessed.csv"
RQ1_DIR = BASE_DIR / "rq1_output"
OUT_DIR = BASE_DIR / "rq5_output"
OUT_DIR.mkdir(exist_ok=True)

# Network construction
MIN_TERM_LEN = 3
MIN_TERM_DF = 25  # term must appear in at least this many documents (reduces noise)
MAX_TERMS_PER_DOC = 50  # cap extremely long keyword lists
TOPIC_MIN_DOCS = 40  # ignore very small topics for stability
TOP_K_EDGES_PER_TOPIC = 8
MIN_EDGE_SIM = 0.06
RANDOM_SEED = 42


def load_corpus_rq5() -> pd.DataFrame:
    usecols = [
        "UT (Unique WOS ID)",
        "Article Title",
        "Abstract",
        "preprocessed_text",
        "Publication Year",
        "Language",
        "Author Keywords",
        "Keywords Plus",
        "Times Cited, WoS Core",
    ]
    df = pd.read_csv(INPUT_CSV, usecols=usecols, low_memory=False)
    df = df.dropna(subset=["preprocessed_text", "Publication Year"])
    df = df[df["preprocessed_text"].str.strip().astype(bool)]
    df["Publication Year"] = df["Publication Year"].astype(int)
    df = df[(df["Publication Year"] >= 2019) & (df["Publication Year"] <= 2026)]
    return df.reset_index(drop=True)


def load_topic_assignments(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    doc_path = RQ1_DIR / "rq1_document_topics.csv"
    qa_path = RQ1_DIR / "rq1_topic_qa.csv"
    if not doc_path.exists() or not qa_path.exists():
        raise FileNotFoundError("RQ1 outputs required. Run `python rq1.py` first.")

    doc_topics = pd.read_csv(doc_path)
    qa = pd.read_csv(qa_path)
    merged = df.merge(
        doc_topics[["UT (Unique WOS ID)", "topic", "topic_type", "display_name"]],
        on="UT (Unique WOS ID)",
        how="inner",
    )
    return merged, qa


def parse_keyword_field(value) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    raw = re.split(r";|,|\n|\r", value)
    out = []
    for r in raw:
        r = re.sub(r"\s+", " ", r).strip()
        if len(r) < MIN_TERM_LEN:
            continue
        r = r.strip(" .;:,").lower()
        out.append(r)
    # dedupe but preserve order
    dedup = list(dict.fromkeys(out))
    return dedup[:MAX_TERMS_PER_DOC]


def build_terms_by_doc(df: pd.DataFrame) -> tuple[list[list[str]], Counter]:
    """
    Document-level term lists. Uses Author Keywords + Keywords Plus.
    """
    terms_by_doc: list[list[str]] = []
    df_counter: Counter = Counter()
    ak = df.get("Author Keywords", pd.Series([""] * len(df))).fillna("").tolist()
    kp = df.get("Keywords Plus", pd.Series([""] * len(df))).fillna("").tolist()
    for a, k in zip(ak, kp):
        terms = parse_keyword_field(a) + parse_keyword_field(k)
        terms = list(dict.fromkeys(terms))
        terms_by_doc.append(terms)
        df_counter.update(set(terms))
    return terms_by_doc, df_counter


def topic_term_matrix(
    df: pd.DataFrame,
    terms_by_doc: list[list[str]],
    qa: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, list[int], list[str]]:
    """
    Returns:
      topic_meta (topic_id, name, n_docs),
      matrix (topic x ref) counts,
      topic_ids,
      ref_vocab
    """
    sub_ids = set(substantive_topic_ids(qa))

    # topic sizes
    counts = df[df["topic"].isin(sub_ids) & (df["topic"] >= 0)].groupby("topic").size()
    keep_topics = sorted([int(t) for t, n in counts.items() if int(n) >= TOPIC_MIN_DOCS])

    qa_idx = qa.set_index("topic")
    topic_meta = pd.DataFrame(
        {
            "topic": keep_topics,
            "display_name": [qa_idx.loc[t, "display_name"] if t in qa_idx.index else str(t) for t in keep_topics],
            "n_documents": [int(counts.get(t, 0)) for t in keep_topics],
        }
    )

    all_terms = [terms_by_doc[i] for i in range(len(df))]
    df_counter: Counter = Counter()
    for terms in all_terms:
        df_counter.update(set(terms))
    vocab = sorted([t for t, c in df_counter.items() if c >= MIN_TERM_DF])
    index = {t: i for i, t in enumerate(vocab)}

    mat = np.zeros((len(keep_topics), len(vocab)), dtype=np.float32)
    topic_to_row = {t: i for i, t in enumerate(keep_topics)}

    topics = df["topic"].tolist()
    for i, t in enumerate(topics):
        if t not in topic_to_row:
            continue
        row = topic_to_row[t]
        for term in terms_by_doc[i]:
            j = index.get(term)
            if j is not None:
                mat[row, j] += 1.0

    return topic_meta, mat, keep_topics, vocab


def build_topic_network(
    topic_meta: pd.DataFrame,
    sim: np.ndarray,
    top_k: int = TOP_K_EDGES_PER_TOPIC,
    min_sim: float = MIN_EDGE_SIM,
) -> pd.DataFrame:
    """
    Sparse edge list: top-k edges per node (undirected), filtered by min_sim.
    """
    n = sim.shape[0]
    edges = set()
    for i in range(n):
        order = np.argsort(sim[i])[::-1]
        picked = 0
        for j in order:
            if j == i:
                continue
            if sim[i, j] < min_sim:
                break
            a, b = (i, j) if i < j else (j, i)
            edges.add((a, b))
            picked += 1
            if picked >= top_k:
                break

    rows = []
    for i, j in sorted(edges):
        rows.append(
            {
                "source": int(topic_meta.iloc[i]["topic"]),
                "target": int(topic_meta.iloc[j]["topic"]),
                "weight": float(sim[i, j]),
            }
        )
    return pd.DataFrame(rows)


def louvain_clusters(topic_meta: pd.DataFrame, edges: pd.DataFrame) -> dict[int, int]:
    try:
        import networkx as nx
        from networkx.algorithms.community import louvain_communities

        g = nx.Graph()
        for t in topic_meta["topic"].tolist():
            g.add_node(int(t))
        for _, r in edges.iterrows():
            g.add_edge(int(r["source"]), int(r["target"]), weight=float(r["weight"]))

        if g.number_of_edges() == 0:
            return {int(t): 0 for t in topic_meta["topic"].tolist()}

        comms = louvain_communities(g, weight="weight", seed=RANDOM_SEED)
        out: dict[int, int] = {}
        for cid, comm in enumerate(comms):
            for node in comm:
                out[int(node)] = int(cid)
        return out
    except Exception:
        return {int(t): 0 for t in topic_meta["topic"].tolist()}


def network_metrics(topic_meta: pd.DataFrame, edges: pd.DataFrame, cluster_map: dict[int, int]) -> pd.DataFrame:
    import networkx as nx

    g = nx.Graph()
    for t in topic_meta["topic"].tolist():
        g.add_node(int(t))
    for _, r in edges.iterrows():
        g.add_edge(int(r["source"]), int(r["target"]), weight=float(r["weight"]))

    strength = {n: 0.0 for n in g.nodes}
    for u, v, d in g.edges(data=True):
        w = float(d.get("weight", 1.0))
        strength[u] += w
        strength[v] += w

    bet = nx.betweenness_centrality(g, weight="weight", normalized=True) if g.number_of_edges() else {n: 0.0 for n in g.nodes}
    deg = dict(g.degree())

    rows = []
    for t in topic_meta["topic"].tolist():
        rows.append(
            {
                "topic": int(t),
                "cluster": int(cluster_map.get(int(t), -1)),
                "degree": int(deg.get(int(t), 0)),
                "strength": float(strength.get(int(t), 0.0)),
                "betweenness": float(bet.get(int(t), 0.0)),
            }
        )
    return pd.DataFrame(rows)


def cluster_characterization(topic_meta: pd.DataFrame, metrics: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    name_map = topic_meta.set_index("topic")["display_name"].to_dict()
    cluster_sizes = metrics.groupby("cluster")["topic"].count().rename("n_topics")

    # within vs between weights
    within = defaultdict(float)
    between = defaultdict(float)
    for _, r in edges.iterrows():
        a, b, w = int(r["source"]), int(r["target"]), float(r["weight"])
        ca = int(metrics.loc[metrics["topic"] == a, "cluster"].iloc[0])
        cb = int(metrics.loc[metrics["topic"] == b, "cluster"].iloc[0])
        if ca == cb:
            within[ca] += w
        else:
            between[ca] += w
            between[cb] += w

    rows = []
    for cid, n in cluster_sizes.items():
        sub = metrics[metrics["cluster"] == cid].copy()
        top_cent = sub.sort_values("betweenness", ascending=False).head(3)["topic"].tolist()
        top_str = sub.sort_values("strength", ascending=False).head(3)["topic"].tolist()
        rows.append(
            {
                "cluster": int(cid),
                "n_topics": int(n),
                "within_weight": float(within.get(cid, 0.0)),
                "between_weight": float(between.get(cid, 0.0)),
                "cohesion_ratio_within_over_total": float(within.get(cid, 0.0) / (within.get(cid, 0.0) + between.get(cid, 0.0) + 1e-9)),
                "top_strength_topics": "; ".join([f"T{t} {name_map.get(t, '')[:35]}" for t in top_str]),
                "top_bridge_topics": "; ".join([f"T{t} {name_map.get(t, '')[:35]}" for t in top_cent]),
            }
        )
    return pd.DataFrame(rows).sort_values("n_topics", ascending=False)


def plot_network(topic_meta: pd.DataFrame, edges: pd.DataFrame, metrics: pd.DataFrame, out_path: Path) -> None:
    import networkx as nx

    g = nx.Graph()
    for _, r in topic_meta.iterrows():
        g.add_node(int(r["topic"]), label=str(r["display_name"]))
    for _, r in edges.iterrows():
        g.add_edge(int(r["source"]), int(r["target"]), weight=float(r["weight"]))

    clusters = metrics.set_index("topic")["cluster"].to_dict()
    strengths = metrics.set_index("topic")["strength"].to_dict()

    # deterministic layout
    pos = nx.spring_layout(g, seed=RANDOM_SEED, k=0.8 / math.sqrt(max(g.number_of_nodes(), 1)))

    # color map
    uniq = sorted(set(clusters.values()))
    palette = plt.cm.get_cmap("tab20", max(len(uniq), 1))
    color = {c: palette(i) for i, c in enumerate(uniq)}

    fig, ax = plt.subplots(figsize=(12, 9))
    # edges
    widths = [0.5 + 4.0 * float(d.get("weight", 0.0)) for _, _, d in g.edges(data=True)]
    nx.draw_networkx_edges(g, pos, ax=ax, width=widths, alpha=0.25, edge_color="#222")
    # nodes
    node_sizes = [120 + 900 * float(strengths.get(n, 0.0)) / (max(strengths.values()) + 1e-9) for n in g.nodes]
    node_colors = [color.get(int(clusters.get(n, -1)), (0.6, 0.6, 0.6, 1.0)) for n in g.nodes]
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=node_sizes, node_color=node_colors, linewidths=0.4, edgecolors="k")

    # label only top strength nodes to keep readable
    top = metrics.sort_values("strength", ascending=False).head(20)["topic"].tolist()
    labels = {int(n): f"T{int(n)}" for n in top}
    nx.draw_networkx_labels(g, pos, labels=labels, font_size=8, ax=ax)

    ax.set_title("RQ5 — Intellectual network of themes (bibliographic coupling)", fontsize=14)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def write_report(
    topic_meta: pd.DataFrame,
    edges: pd.DataFrame,
    metrics: pd.DataFrame,
    clusters: pd.DataFrame,
    runtime: float,
) -> None:
    name_map = topic_meta.set_index("topic")["display_name"].to_dict()
    top_bridges = metrics.sort_values("betweenness", ascending=False).head(12)
    top_core = metrics.sort_values("strength", ascending=False).head(12)

    lines = [
        "=" * 78,
        "RQ5 REPORT — INTELLECTUAL THEME NETWORK & MACRO-TRADITIONS",
        "=" * 78,
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Runtime: {runtime:.1f} s",
        "",
        "RESEARCH QUESTION",
        "  How do the discovered themes relate to one another in a structured intellectual",
        "  network, forming clusters or macro-traditions that define the discipline's",
        "  conceptual architecture?",
        "",
        "METHOD",
        "  1. Theme layer: BERTopic substantive topics (RQ1 topic QA)",
        "  2. Intellectual links: bibliographic coupling (shared cited references)",
        f"  3. Similarity: cosine(topic×term count vectors), term DF>={MIN_TERM_DF}",
        f"  4. Network sparsification: top-k={TOP_K_EDGES_PER_TOPIC} per topic, min_sim={MIN_EDGE_SIM}",
        "  5. Macro-traditions: Louvain community detection on weighted network",
        "",
        "NETWORK SUMMARY",
        f"  Topics included: {len(topic_meta):,} (topic min docs={TOPIC_MIN_DOCS})",
        f"  Edges retained:  {len(edges):,}",
        f"  Macro-traditions (clusters): {int(metrics['cluster'].nunique())}",
        "",
        "—" * 40,
        "A. MACRO-TRADITIONS (CLUSTERS)",
        "—" * 40,
    ]

    for _, r in clusters.iterrows():
        lines.append(
            f"  Cluster {int(r['cluster']):2d} | n_topics={int(r['n_topics']):2d} | "
            f"cohesion={r['cohesion_ratio_within_over_total']:.2f}"
        )
        lines.append(f"    Core (strength):  {r['top_strength_topics']}")
        lines.append(f"    Bridges (between): {r['top_bridge_topics']}")

    lines.extend(
        [
            "",
            "—" * 40,
            "B. ARCHITECTURE: CORE THEMES & BRIDGES",
            "—" * 40,
            "",
            "B1. Core themes (highest strength; many strong intellectual links)",
        ]
    )
    for _, r in top_core.iterrows():
        t = int(r["topic"])
        lines.append(
            f"  T{t:3d} | strength={r['strength']:.3f} | degree={int(r['degree'])} | {name_map.get(t, '')}"
        )

    lines.extend(["", "B2. Bridging themes (highest betweenness; connect macro-traditions)"])
    for _, r in top_bridges.iterrows():
        t = int(r["topic"])
        lines.append(
            f"  T{t:3d} | betweenness={r['betweenness']:.3f} | cluster={int(r['cluster'])} | {name_map.get(t, '')}"
        )

    lines.extend(
        [
            "",
            "INTERPRETATION (CONCEPTUAL ARCHITECTURE)",
            "  - Clusters represent macro-traditions grounded in shared cited literatures.",
            "  - Core nodes anchor dominant traditions; bridging nodes indicate integrative",
            "    areas where methods, populations, or applied settings connect otherwise",
            "    separated literatures (e.g., mental health outcomes ↔ policy/forensic ↔ digital harm).",
            "",
            "OUTPUT FILES",
            "  rq5_topic_meta.csv",
            "  rq5_topic_edges.csv",
            "  rq5_topic_metrics.csv",
            "  rq5_cluster_summary.csv",
            "  rq5_report.txt",
            "  rq5_network.png",
            "=" * 78,
        ]
    )

    (OUT_DIR / "rq5_report.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    t0 = time.perf_counter()
    print("RQ5 — Intellectual network of themes (macro-traditions)")

    df_raw = load_corpus_rq5()
    df, _ = filter_english_corpus(df_raw)
    df, qa = load_topic_assignments(df)

    print(f"  Documents: {len(df):,}")
    terms_by_doc, df_counter = build_terms_by_doc(df)
    topic_meta, mat, topic_ids, vocab = topic_term_matrix(df, terms_by_doc, qa)
    topic_meta.to_csv(OUT_DIR / "rq5_topic_meta.csv", index=False)

    print(f"  Topics included (n>= {TOPIC_MIN_DOCS}): {len(topic_meta)}")
    print(f"  Vocab size (DF>={MIN_TERM_DF}): {len(vocab)}")
    if mat.shape[1] == 0:
        raise ValueError("No terms available for coupling. Check keyword fields in data_preprocessed.csv.")
    print("  Computing topic–topic similarity...")
    sim = cosine_similarity(mat)

    edges = build_topic_network(topic_meta, sim)
    edges.to_csv(OUT_DIR / "rq5_topic_edges.csv", index=False)

    print("  Community detection...")
    cluster_map = louvain_clusters(topic_meta, edges)
    metrics = network_metrics(topic_meta, edges, cluster_map)
    metrics = metrics.merge(topic_meta, on="topic", how="left")
    metrics.to_csv(OUT_DIR / "rq5_topic_metrics.csv", index=False)

    clusters = cluster_characterization(topic_meta, metrics, edges)
    clusters.to_csv(OUT_DIR / "rq5_cluster_summary.csv", index=False)

    print("  Plotting network...")
    plot_network(topic_meta, edges, metrics, OUT_DIR / "rq5_network.png")

    runtime = time.perf_counter() - t0
    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "n_documents": int(len(df)),
        "n_topics_included": int(len(topic_meta)),
        "n_edges": int(len(edges)),
        "n_clusters": int(metrics["cluster"].nunique()),
        "runtime_seconds": round(runtime, 2),
        "top_core": metrics.sort_values("strength", ascending=False)
        .head(8)[["topic", "display_name", "cluster", "strength", "betweenness"]]
        .to_dict(orient="records"),
        "top_bridges": metrics.sort_values("betweenness", ascending=False)
        .head(8)[["topic", "display_name", "cluster", "strength", "betweenness"]]
        .to_dict(orient="records"),
    }
    (OUT_DIR / "rq5_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    write_report(topic_meta, edges, metrics, clusters, runtime)
    print(f"\nDone in {runtime:.1f}s -> {OUT_DIR}")
    print(f"  Clusters: {summary['n_clusters']} | Edges: {summary['n_edges']} | Topics: {summary['n_topics_included']}")
    print(f"  Report: {OUT_DIR / 'rq5_report.txt'}")


if __name__ == "__main__":
    main()

