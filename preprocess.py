"""
Full text preprocessing pipeline for Scopus childhood-trauma corpus.
"""

from __future__ import annotations

import json
import re
import string
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path

import nltk
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "data.csv"
OUTPUT_CSV = BASE_DIR / "data_preprocessed.csv"
OUTPUT_TOKENS_CSV = BASE_DIR / "data_preprocessed_tokens.csv"
REPORT_PATH = BASE_DIR / "preprocessing_report.txt"
MANUAL_STOPWORDS_PATH = BASE_DIR / "manual_stopwords.txt"
STOPWORDS_EXPORT_PATH = BASE_DIR / "all_stopwords_used.txt"
STATS_JSON_PATH = BASE_DIR / "preprocessing_stats.json"

TEXT_COLUMNS = [
    "Article Title",
    "Abstract",
    "Author Keywords",
    "Keywords Plus",
]

# Records must have all of these non-empty; keywords = at least one keyword field
REQUIRED_TITLE_COL = "Article Title"
REQUIRED_ABSTRACT_COL = "Abstract"
KEYWORD_COLUMNS = ["Author Keywords", "Keywords Plus"]

MIN_TOKEN_LEN = 2
MAX_TOKEN_LEN = 40

# ---------------------------------------------------------------------------
# NLTK resources
# ---------------------------------------------------------------------------
def ensure_nltk_data() -> None:
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


# ---------------------------------------------------------------------------
# Stopwords
# ---------------------------------------------------------------------------
def load_manual_stopwords(path: Path) -> set[str]:
    words: set[str] = set()
    if not path.exists():
        return words
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().lower()
        if not line or line.startswith("#"):
            continue
        words.add(line)
    return words


def build_stopword_set() -> set[str]:
    english = set(nltk_stopwords.words("english"))
    manual = load_manual_stopwords(MANUAL_STOPWORDS_PATH)
    combined = english | manual
    # Single-character tokens
    combined |= set(string.ascii_lowercase)
    return combined


# ---------------------------------------------------------------------------
# Text cleaning helpers
# ---------------------------------------------------------------------------
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\S+@\S+")
DOI_RE = re.compile(r"\b10\.\d{4,}/\S+", re.IGNORECASE)
NON_ALPHA_RE = re.compile(r"[^a-z\s\-]")
MULTISPACE_RE = re.compile(r"\s+")
DIGIT_RE = re.compile(r"^\d+$")

LEMMATIZER = WordNetLemmatizer()
POS_MAP = {
    "J": "a",
    "V": "v",
    "N": "n",
    "R": "r",
}


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def strip_html(text: str) -> str:
    if "<" in text and ">" in text:
        return BeautifulSoup(text, "html.parser").get_text(" ")
    return text


def clean_raw_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = normalize_unicode(text)
    text = strip_html(text)
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = DOI_RE.sub(" ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text


def get_wordnet_pos(tag: str) -> str:
    if tag.startswith("J"):
        return POS_MAP["J"]
    if tag.startswith("V"):
        return POS_MAP["V"]
    if tag.startswith("N"):
        return POS_MAP["N"]
    if tag.startswith("R"):
        return POS_MAP["R"]
    return "n"


def tokenize_and_filter(text: str, stopwords: set[str]) -> list[str]:
    if not text:
        return []
    tokens = word_tokenize(text)
    filtered: list[str] = []
    for tok in tokens:
        tok = tok.strip("-")
        if len(tok) < MIN_TOKEN_LEN or len(tok) > MAX_TOKEN_LEN:
            continue
        if DIGIT_RE.match(tok):
            continue
        if tok in stopwords:
            continue
        filtered.append(tok)
    return filtered


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    if not tokens:
        return []
    tagged = nltk.pos_tag(tokens)
    lemmas: list[str] = []
    for word, tag in tagged:
        pos = get_wordnet_pos(tag)
        lemma = LEMMATIZER.lemmatize(word, pos)
        if lemma:
            lemmas.append(lemma)
    return lemmas


def preprocess_text(text: str, stopwords: set[str]) -> tuple[str, list[str]]:
    cleaned = clean_raw_text(text)
    tokens = tokenize_and_filter(cleaned, stopwords)
    lemmas = lemmatize_tokens(tokens)
    return " ".join(lemmas), lemmas


def combine_fields(row: pd.Series) -> str:
    parts: list[str] = []
    for col in TEXT_COLUMNS:
        val = row.get(col, "")
        if pd.notna(val) and str(val).strip():
            parts.append(str(val))
    return " ".join(parts)


def is_empty_series(series: pd.Series) -> pd.Series:
    return series.isna() | (series.astype(str).str.strip() == "")


def filter_incomplete_records(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop rows with empty title, abstract, or both keyword fields."""
    empty_title = is_empty_series(df[REQUIRED_TITLE_COL])
    empty_abstract = is_empty_series(df[REQUIRED_ABSTRACT_COL])
    empty_keywords = is_empty_series(df[KEYWORD_COLUMNS[0]]) & is_empty_series(
        df[KEYWORD_COLUMNS[1]]
    )
    remove_mask = empty_title | empty_abstract | empty_keywords

    stats = {
        "input_records": len(df),
        "removed_empty_title": int(empty_title.sum()),
        "removed_empty_abstract": int(empty_abstract.sum()),
        "removed_empty_keywords": int(empty_keywords.sum()),
        "removed_total": int(remove_mask.sum()),
        "kept_records": int((~remove_mask).sum()),
    }
    return df.loc[~remove_mask].reset_index(drop=True), stats


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def top_ngrams(tokens_series: pd.Series, n: int = 20, top_k: int = 30) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for tokens in tokens_series:
        if not isinstance(tokens, list):
            continue
        if n == 1:
            counter.update(tokens)
        else:
            counter.update([" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)])
    return counter.most_common(top_k)


def write_report(
    df: pd.DataFrame,
    stopwords: set[str],
    manual_count: int,
    nltk_count: int,
    elapsed_sec: float,
    filter_stats: dict[str, int],
) -> None:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("SCOPUS CHILDHOOD TRAUMA CORPUS — PREPROCESSING REPORT")
    lines.append("=" * 72)
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Input file: {INPUT_CSV.name}")
    lines.append(f"Records (after filtering): {len(df):,}")
    lines.append(f"Runtime: {elapsed_sec:.1f} seconds")
    lines.append("")

    lines.append("--- RECORD FILTERING (INCOMPLETE TEXT REMOVED) ---")
    lines.append(f"  Input records: {filter_stats['input_records']:,}")
    lines.append(f"  Removed — empty title: {filter_stats['removed_empty_title']:,}")
    lines.append(f"  Removed — empty abstract: {filter_stats['removed_empty_abstract']:,}")
    lines.append(
        f"  Removed — empty keywords (both Author Keywords & Keywords Plus): "
        f"{filter_stats['removed_empty_keywords']:,}"
    )
    lines.append(f"  Total removed: {filter_stats['removed_total']:,}")
    lines.append(f"  Records kept: {filter_stats['kept_records']:,}")
    lines.append(
        "  Rule: keep only records with non-empty title, abstract, and at least one keyword field"
    )
    lines.append("")

    lines.append("--- SEARCH QUERY (SOURCE) ---")
    lines.append(
        'TS=("childhood trauma" OR "childhood maltreatment" OR "childhood adversity" '
        'OR "early life stress" OR "childhood abuse" OR "childhood neglect" '
        'OR "physical abuse" OR "sexual abuse" OR "emotional abuse" '
        'OR "psychological abuse" OR "physical neglect" OR "emotional neglect")'
    )
    lines.append("")

    lines.append("--- TEXT FIELDS COMBINED ---")
    for col in TEXT_COLUMNS:
        non_null = df[col].notna().sum() if col in df.columns else 0
        lines.append(f"  - {col}: {non_null:,} non-empty")
    lines.append("  Combined column: combined_text (raw) -> preprocessed_text (cleaned)")
    lines.append("")

    lines.append("--- PREPROCESSING STEPS APPLIED ---")
    steps = [
        "1. Filter: drop records with empty title, abstract, or both keyword fields",
        "2. Field combination: Article Title + Abstract + Author Keywords + Keywords Plus",
        "3. Unicode normalization (NFKC)",
        "4. HTML tag removal (BeautifulSoup)",
        "5. Lowercasing",
        "6. URL, email, and DOI pattern removal",
        "7. Non-alphabetic character removal (hyphens retained for tokenization)",
        "8. Whitespace normalization",
        "9. NLTK word_tokenize",
        f"10. Token length filter: keep {MIN_TOKEN_LEN}–{MAX_TOKEN_LEN} characters",
        "11. Numeric-only token removal",
        "12. Stopword removal (NLTK English + manual domain list)",
        "13. POS-aware WordNet lemmatization",
    ]
    lines.extend(steps)
    lines.append("")

    lines.append("--- STOPWORDS ---")
    lines.append(f"  NLTK English stopwords: {nltk_count:,}")
    lines.append(f"  Manual stopwords: {manual_count:,}")
    lines.append(f"  Total unique stopwords: {len(stopwords):,}")
    lines.append(f"  Exported list: {STOPWORDS_EXPORT_PATH.name}")
    lines.append("")

    lines.append("--- CORPUS STATISTICS ---")
    lines.append(f"  Empty combined_text: {(df['combined_text'].str.strip() == '').sum():,}")
    lines.append(f"  Empty preprocessed_text: {(df['preprocessed_text'].str.strip() == '').sum():,}")
    lines.append(f"  Mean tokens per document: {df['token_count'].mean():.1f}")
    lines.append(f"  Median tokens per document: {df['token_count'].median():.0f}")
    lines.append(f"  Min tokens: {df['token_count'].min()}")
    lines.append(f"  Max tokens: {df['token_count'].max()}")
    lines.append(f"  Total tokens (corpus): {df['token_count'].sum():,}")
    lines.append(f"  Unique lemmas (corpus): {len(set(t for toks in df['tokens'] for t in toks)):,}")
    lines.append("")

    lines.append("--- TOP 30 UNIGRAMS (after preprocessing) ---")
    for word, cnt in top_ngrams(df["tokens"], n=1):
        lines.append(f"  {word:25s} {cnt:>8,}")
    lines.append("")

    lines.append("--- TOP 20 BIGRAMS ---")
    for bigram, cnt in top_ngrams(df["tokens"], n=2, top_k=20):
        lines.append(f"  {bigram:35s} {cnt:>6,}")
    lines.append("")

    lines.append("--- OUTPUT FILES ---")
    lines.append(f"  {OUTPUT_CSV.name} — full metadata + combined/preprocessed text")
    lines.append(f"  {OUTPUT_TOKENS_CSV.name} — UT ID + preprocessed text + token list (compact)")
    lines.append(f"  {STATS_JSON_PATH.name} — machine-readable summary statistics")
    lines.append("=" * 72)

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    import time

    t0 = time.perf_counter()
    ensure_nltk_data()

    print("Loading data...")
    df_raw = pd.read_csv(INPUT_CSV, low_memory=False)
    print("Filtering incomplete records (empty title / abstract / keywords)...")
    df, filter_stats = filter_incomplete_records(df_raw)
    print(
        f"  Kept {filter_stats['kept_records']:,} / {filter_stats['input_records']:,} "
        f"(removed {filter_stats['removed_total']:,})"
    )
    n_records = len(df)

    print("Building stopword set...")
    manual_sw = load_manual_stopwords(MANUAL_STOPWORDS_PATH)
    nltk_sw = set(nltk_stopwords.words("english"))
    stopwords = build_stopword_set()
    STOPWORDS_EXPORT_PATH.write_text(
        "\n".join(sorted(stopwords)), encoding="utf-8"
    )

    print("Combining text fields...")
    df["combined_text"] = df.apply(combine_fields, axis=1)

    print("Preprocessing (tokenize, stopwords, lemmatize)...")
    preprocessed_texts: list[str] = []
    token_lists: list[list[str]] = []

    for text in tqdm(df["combined_text"], total=n_records, desc="Preprocessing"):
        proc, toks = preprocess_text(str(text) if pd.notna(text) else "", stopwords)
        preprocessed_texts.append(proc)
        token_lists.append(toks)

    df["preprocessed_text"] = preprocessed_texts
    df["tokens"] = token_lists
    df["token_count"] = df["tokens"].apply(len)

    elapsed = time.perf_counter() - t0

    print("Saving outputs...")
    # Main output: all columns + new fields (tokens as string for CSV compatibility)
    out_df = df.copy()
    out_df["tokens"] = out_df["tokens"].apply(lambda t: " ".join(t))
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    # Compact file for downstream modeling
    id_col = "UT (Unique WOS ID)" if "UT (Unique WOS ID)" in df.columns else None
    compact_cols = ["preprocessed_text", "token_count", "tokens"]
    if id_col:
        compact_cols = [id_col] + compact_cols
    if "Article Title" in df.columns:
        compact_cols.insert(1 if id_col else 0, "Article Title")
    if "Publication Year" in df.columns:
        compact_cols.append("Publication Year")
    df[compact_cols].to_csv(OUTPUT_TOKENS_CSV, index=False, encoding="utf-8-sig")

    stats = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "input_file": str(INPUT_CSV),
        "filtering": filter_stats,
        "n_records": n_records,
        "runtime_seconds": round(elapsed, 2),
        "nltk_stopwords": len(nltk_sw),
        "manual_stopwords": len(manual_sw),
        "total_stopwords": len(stopwords),
        "empty_preprocessed": int((df["preprocessed_text"].str.strip() == "").sum()),
        "mean_tokens": round(float(df["token_count"].mean()), 2),
        "median_tokens": int(df["token_count"].median()),
        "total_tokens": int(df["token_count"].sum()),
        "unique_lemmas": len(set(t for toks in token_lists for t in toks)),
        "top_unigrams": top_ngrams(pd.Series(token_lists), n=1, top_k=30),
    }
    STATS_JSON_PATH.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    write_report(df, stopwords, len(manual_sw), len(nltk_sw), elapsed, filter_stats)

    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Input records: {filter_stats['input_records']:,}")
    print(f"  Removed: {filter_stats['removed_total']:,}")
    print(f"  Records kept: {n_records:,}")
    print(f"  Report: {REPORT_PATH}")
    print(f"  Preprocessed data: {OUTPUT_CSV}")
    print(f"  Compact export: {OUTPUT_TOKENS_CSV}")


if __name__ == "__main__":
    main()
