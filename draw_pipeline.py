"""
Render full study pipeline as PNG — clean layout, no overlaps.
Run: python draw_pipeline.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BASE = Path(__file__).resolve().parent
OUT = BASE / "pipeline_full.png"

C_DATA = "#BBDEFB"
C_PROC = "#FFE0B2"
C_CORE = "#C8E6C9"
C_RQ = "#E1BEE7"
C_RQ6 = "#F8BBD9"
C_META = "#ECEFF1"
C_EDGE = "#37474F"
C_TITLE = "#0D47A1"
C_LABEL = "#1565C0"
C_TEXT = "#212121"


def rounded_box(ax, x, y, w, h, text, fc, fs=9, bold=False, ec="#78909C", lw=1.2):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.015,rounding_size=0.06",
            linewidth=lw, edgecolor=ec, facecolor=fc,
        )
    )
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fs, color=C_TEXT,
        weight="bold" if bold else "normal", linespacing=1.3,
    )
    return {"cx": x + w / 2, "left": x, "right": x + w, "bottom": y, "top": y + h}


def arr(ax, x0, y0, x1, y1, rad=0.0, color=C_EDGE, lw=1.4, **kwargs):
    del kwargs  # unused
    cs = f"arc3,rad={rad}" if rad else "arc3,rad=0"
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", color=color, linewidth=lw,
        mutation_scale=14, connectionstyle=cs, shrinkA=6, shrinkB=6,
    ))


def phase_title(ax, x, y, text):
    ax.text(x, y, text, fontsize=10, weight="bold", color=C_LABEL, ha="left", va="bottom")


def main() -> None:
    W, H = 20.0, 14.5
    fig, ax = plt.subplots(figsize=(W, H), dpi=200)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.set_facecolor("white")

    ax.text(W / 2, H - 0.25, "Childhood Trauma Literature — Full Analytical Pipeline",
            ha="center", va="top", fontsize=17, weight="bold", color=C_TITLE)
    ax.text(W / 2, H - 0.72,
            "Scopus  →  preprocessing  →  English QC  →  BERTopic (RQ1)  →  RQ2–RQ5  |  RQ6 parallel branch",
            ha="center", va="top", fontsize=9.5, color="#546E7A")

    mx, mw = 0.5, 6.0

    # Fixed Y coordinates (top → bottom) — no overlap
    y_scopus = 12.55
    y_data = 11.65
    y_pre = 9.85
    y_csv = 8.35
    y_eng = 7.15
    y_rq1 = 5.75
    y_hub = 4.55

    phase_title(ax, mx, y_scopus + 0.82, "Phase 0 — Data acquisition")
    b0 = rounded_box(ax, mx, y_scopus, mw, 0.72, "Scopus search (childhood-trauma query)", C_DATA)
    b1 = rounded_box(ax, mx, y_data, mw, 0.72, "data.csv  (~19,799 records)", C_DATA)
    arr(ax, b0["cx"], b0["bottom"], b1["cx"], b1["top"])

    phase_title(ax, mx, y_pre + 0.82, "Phase 1 — preprocess.py")
    b_pre = rounded_box(ax, mx, y_pre, mw, 1.25,
        "Filter · combine fields · clean · tokenize · lemmatize\n"
        "Stopwords: NLTK + manual_stopwords.txt", C_PROC, fs=8.5)
    arr(ax, b1["cx"], b1["bottom"], b_pre["cx"], b_pre["top"])

    b_csv = rounded_box(ax, mx, y_csv, mw, 0.78,
        "data_preprocessed.csv\n(preprocessed_text, tokens, metadata)", C_PROC, fs=8.5, bold=True)
    arr(ax, b_pre["cx"], b_pre["bottom"], b_csv["cx"], b_csv["top"])

    phase_title(ax, mx, y_eng + 0.82, "Phase 2 — English QC")
    b_eng = rounded_box(ax, mx, y_eng, mw, 0.72,
        "rq1.filter_english_corpus\nlangdetect + Spanish-token heuristic", C_CORE, fs=8.5)
    arr(ax, b_csv["cx"], b_csv["bottom"], b_eng["cx"], b_eng["top"])

    phase_title(ax, mx, y_rq1 + 0.95, "Phase 3 — rq1.py (RQ1)")
    b_rq1 = rounded_box(ax, mx, y_rq1, mw, 1.45,
        "TF-IDF → SVD (128d)\n"
        "BERTopic: UMAP + HDBSCAN + c-TF-IDF\n"
        "topics_over_time · Topic QA · substantive topics", C_CORE, fs=8.5)
    arr(ax, b_eng["cx"], b_eng["bottom"], b_rq1["cx"], b_rq1["top"])

    hub = rounded_box(ax, mx, y_hub, mw, 0.62,
        "Output: rq1_output/  (document_topics, topic_qa, topic_info, …)", C_CORE, fs=8.5, bold=True)
    arr(ax, b_rq1["cx"], b_rq1["bottom"], hub["cx"], hub["top"])

    # RQ1 temporal (right column, y 8.5–12.3)
    rx, rw = 7.2, 5.9
    phase_title(ax, rx, 12.35, "RQ1 — temporal & driver outputs")
    r1_ys = [11.45, 10.65, 9.85, 9.05, 8.25]
    for txt, ry in zip([
        "Yearly topic shares",
        "Mann–Kendall + Theil–Sen",
        "Early (2019–21) vs late (2023–25)",
        "Drivers: volume, citations, % reviews",
        "Keywords Plus log-odds",
    ], r1_ys):
        rounded_box(ax, rx, ry, rw, 0.58, txt, "#A5D6A7", fs=8.5)
    arr(ax, hub["right"] + 0.05, (hub["top"] + hub["bottom"]) / 2, rx, 10.0)

    # RQ6 parallel (top-right, y 8.5–12)
    r6x, r6w = 13.55, 6.0
    phase_title(ax, r6x, 12.35, "Parallel — rq6.py")
    r6a = rounded_box(ax, r6x, 11.35, r6w, 0.58, "From data_preprocessed.csv", C_RQ6, fs=8.5)
    r6b = rounded_box(ax, r6x, 10.55, r6w, 0.68,
        "English QC · 2020–2025 · peer-reviewed", C_RQ6, fs=8.5)
    r6c = rounded_box(ax, r6x, 9.15, r6w, 1.15,
        "LDA K=32 · noun phrases\nResearch Areas · macro-traditions\n→ rq6_output/", C_RQ6, fs=8.5, bold=True)
    arr(ax, b_csv["right"], (b_csv["top"] + b_csv["bottom"]) / 2, r6a["left"], r6a["top"] + 0.1,
        rad=0.15, color="#C2185B", lw=1.6)
    arr(ax, r6a["cx"], r6a["bottom"], r6b["cx"], r6b["top"], color="#C2185B")
    arr(ax, r6b["cx"], r6b["bottom"], r6c["cx"], r6c["top"], color="#C2185B")
    ax.text(r6x + r6w / 2, 8.85, "Optional: rq6.py --post", ha="center", fontsize=7.5,
            color="#AD1457", style="italic")

    # Downstream RQ2–RQ5 (bottom, y 0.35–3.85) — well below hub (hub top ≈ 5.17)
    phase_title(ax, 0.5, 4.05, "Downstream (after rq1.py) — all use rq1_output/")
    bw, bh_t, bh_b = 4.55, 0.55, 1.35
    y_t, y_b = 3.15, 1.65
    branches = [
        ("RQ2 · rq2.py", "Emerging themes & gaps",
         "Co-word · Louvain · strategic diagram\nBurst keywords · ROI / Gap Score"),
        ("RQ3 · rq3.py", "Prevalence & composition",
         "Periods P1–P4 · MK · z-test (BH-FDR)\nJS divergence · meta-theme"),
        ("RQ4 · rq4.py", "Cross-paradigm robustness",
         "BERTopic · LDA · NMF · K-means\nARI/NMI · Hungarian · NP triangulation"),
        ("RQ5 · rq5.py", "Intellectual network",
         "Keyword coupling*\nLouvain · macro-traditions · centrality"),
    ]
    for i, (code, sub, body) in enumerate(branches):
        x = 0.5 + i * 4.85
        rounded_box(ax, x, y_t, bw, bh_t, f"{code}\n{sub}", C_RQ, fs=8.5, bold=True)
        rounded_box(ax, x, y_b, bw, bh_b, body, C_RQ, fs=7.8)
        arr(ax, hub["cx"], hub["bottom"], x + bw / 2, y_t + bh_t, rad=-0.06)

    ax.text(0.5, 1.35,
            "* No cited references in Scopus export → keyword coupling (not bibliographic coupling).",
            fontsize=7.5, color="#546E7A", style="italic")

    # Footer
    rounded_box(ax, 0.5, 0.15, 11.5, 0.95,
        "Run order:  (1) preprocess.py  →  (2) rq1.py  →  (3) rq2 · rq3 · rq4 · rq5  |  "
        "(4) rq6.py after step 1  →  (5) rq6.py --post optional",
        C_META, fs=8.5, ec="#90A4AE")

    for i, (c, lab) in enumerate([
        (C_DATA, "Data"), (C_PROC, "Preprocess"), (C_CORE, "RQ1 BERTopic"),
        (C_RQ, "RQ2–5"), (C_RQ6, "RQ6"),
    ]):
        lx = 12.5 + (i % 3) * 2.45
        ly = 0.55 - (i // 3) * 0.38
        ax.add_patch(FancyBboxPatch((lx, ly), 0.3, 0.24, boxstyle="round,pad=0.01",
                                    facecolor=c, edgecolor="#90A4AE"))
        ax.text(lx + 0.38, ly + 0.12, lab, fontsize=8, va="center")

    fig.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=0.2, facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
