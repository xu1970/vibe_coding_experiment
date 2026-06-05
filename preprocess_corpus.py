"""
Unified LDA preprocessing pipeline.

Consolidates ingestion, text cleaning/tokenization, vocabulary filtering,
and Gensim dictionary/corpus construction behind a single config dict.
"""

from __future__ import annotations

import glob
import re
from collections import Counter
from itertools import chain
from pathlib import Path
from typing import Any

import emoji
import jieba.posseg as pseg
import pandas as pd
from gensim import corpora
from opencc import OpenCC

from functions import add_replacements, add_words_from_file, replace_w_rules


# ---------------------------------------------------------------------------
# Default configuration — adjust filters in one place
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: dict[str, Any] = {
    # --- Ingestion ---
    "data_dir": "/Users/xiangningxu/Documents/vibe_coding/Scrape",
    "file_pattern": "comments_sampled_*.csv",
    "text_column": "content",
    "vote_values": None,  # e.g. [1, 2]; None keeps all rows

    # --- Resource files (paths relative to project root or absolute) ---
    "custom_words_file": "data/custom_words.txt",
    "stop_words_file": "stop_words.txt",
    "filtered_words_file": "filtered_words.txt",
    "low_frequency_file": "low_frequency.txt",
    "replacement_rules_file": "replacement_rules2.txt",
    "megatoken_file": "megatoken.txt",

    # --- Text cleaning ---
    "min_chunk_len": 3,
    "simplify_traditional": True,

    # --- Token / vocabulary filtering ---
    "min_token_freq": 2,          # keep tokens appearing more than this count corpus-wide
    "exclude_tokens": set(),      # hard-coded removals before dictionary step
    "lda_exclude_tokens": {       # removed only for LDA vocabulary (optional second pass)
        "孩子", "小孩子", "生孩子", "没有", "不想", "不生", "问题", "时候",
    },

    # --- Gensim Dictionary.filter_extremes ---
    "min_doc_freq": 2,            # no_below
    "max_doc_freq": 0.8,         # no_above (proportion of documents)
    "keep_n": 800,

    # --- Validation ---
    "drop_empty_docs": True,
    "min_tokens_per_doc": 1,
}


# ---------------------------------------------------------------------------
# Helpers (existing notebook logic, grouped by stage)
# ---------------------------------------------------------------------------
def _load_word_list(path: str | Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _convert_emoji(sentence: str) -> str:
    sentence = emoji.demojize(sentence)
    sentence = re.sub(r":cow_face::horse_face:", "牛马", sentence)
    sentence = re.sub(r"\[[^\[\]]*\]", "", sentence)
    return sentence


def _preprocess_text(sentence: str) -> str:
    sentence = re.sub(r"回复|[\u4e00-\u9fff]@[^:]*:", "", sentence)
    sentence = re.sub(r"@[^:]* ", "", sentence)
    sentence = re.sub(r"[\s]", "，", sentence)
    sentence = re.sub(r"[\n\t\s]*", "", sentence)
    sentence = re.sub(r"我觉得", "", sentence)
    sentence = re.sub(r"[“+”（）]", "", sentence)
    sentence = re.sub(r"%", "百分之", sentence)
    return sentence


def _split_chunks(sentence: str, min_len: int) -> list[str]:
    chunks = re.split(r"[。！？?，, （）\[\];:、……]", sentence)
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    return [chunk for chunk in chunks if len(chunk) > min_len]


def _refine_tokens(chunk_texts: list[str]) -> list[str]:
    """Pattern-based token refinement (replaces former Stanza step)."""
    result: list[str] = []
    child_words = {
        "孩子", "小孩", "宝宝", "小孩儿", "崽", "崽崽", "男孩", "女孩",
        "崽子", "儿子", "一个人", "宠物", "猫", "狗",
    }
    verb_words = {"给", "有", "生", "没", "抱", "带", "养", "爱", "不要", "要"}
    keep_flags = {"n", "v", "a", "vn", "an", "nr", "nt", "nz", "t"}

    for token_text in chunk_texts:
        if not token_text.strip():
            continue
        pos_tokens = list(pseg.cut(token_text))
        pattern_tokens: list[str] = []
        i = 0
        while i < len(pos_tokens):
            word, flag = pos_tokens[i].word, pos_tokens[i].flag
            if word in {"不", "没"} and i + 1 < len(pos_tokens):
                next_word, next_flag = pos_tokens[i + 1].word, pos_tokens[i + 1].flag
                if next_flag in {"v", "vn"} and len(next_word) <= 3:
                    pattern_tokens.append(word + next_word)
                    i += 2
                    continue
            if word in verb_words and i + 1 < len(pos_tokens):
                next_word = pos_tokens[i + 1].word
                if next_word in child_words:
                    pattern_tokens.append(word + next_word)
                    i += 2
                    continue
            if flag in keep_flags:
                pattern_tokens.append(word)
            i += 1
        result.extend(pattern_tokens)
    return result


def _tokenize_chunk(
    text: str,
    stopwords: set[str],
    filtered_words: set[str],
    low_frequency: set[str],
) -> str:
    tokens = [word.word for word in pseg.cut(text)]
    tokens = [t for t in tokens if t not in stopwords]
    tokens = [t for t in tokens if t not in filtered_words]
    tokens = [t for t in tokens if t not in low_frequency]
    return " ".join(tokens)


def _apply_replacement_dict(tokens: list[str], rules: dict[str, str]) -> list[str]:
    return [rules.get(t, t) for t in tokens]


# ---------------------------------------------------------------------------
# Stage functions
# ---------------------------------------------------------------------------
def load_documents(config: dict[str, Any]) -> pd.DataFrame:
    """Load and optionally filter comment CSVs."""
    data_dir = Path(config["data_dir"])
    pattern = str(data_dir / config["file_pattern"])
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {pattern}")

    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)

    text_col = config["text_column"]
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found. Columns: {list(df.columns)}")

    vote_values = config.get("vote_values")
    if vote_values is not None and "vote" in df.columns:
        df = df[df["vote"].isin(vote_values)]

    return df


def tokenize_documents(
    comments: list[str],
    config: dict[str, Any],
    *,
    stopwords: set[str],
    filtered_words: set[str],
    low_frequency: set[str],
    replacement_rules: dict[str, str],
    megatoken: dict[str, str],
    simplify_fn,
) -> list[list[str]]:
    """Run per-document cleaning, chunking, tokenization, and refinement."""
    min_chunk = config["min_chunk_len"]
    docs: list[list[str]] = []

    for comment in comments:
        sentence = _preprocess_text(comment)
        sentence = _convert_emoji(sentence)
        if config["simplify_traditional"]:
            sentence = simplify_fn(sentence)

        chunks = _split_chunks(sentence, min_chunk)
        chunk_strings = [
            _tokenize_chunk(c, stopwords, filtered_words, low_frequency)
            for c in chunks
        ]
        tokens = _refine_tokens(chunk_strings)
        tokens = _apply_replacement_dict(tokens, replacement_rules)
        tokens = _apply_replacement_dict(tokens, megatoken)
        docs.append(tokens)

    return docs


def build_vocabulary(
    tokenized_docs: list[list[str]],
    config: dict[str, Any],
) -> tuple[set[str], list[list[str]]]:
    """Corpus-wide frequency filter + optional hard exclusions."""
    flat = list(chain.from_iterable(tokenized_docs))
    counts = Counter(flat)
    min_freq = config["min_token_freq"]
    vocab = {t for t, c in counts.items() if c > min_freq}

    exclude = set(config.get("exclude_tokens", set()))
    vocab -= exclude

    filtered_docs = [[t for t in doc if t in vocab] for doc in tokenized_docs]
    return vocab, filtered_docs


def build_gensim_corpus(
    tokenized_docs: list[list[str]],
    config: dict[str, Any],
    *,
    lda_vocab_exclude: set[str] | None = None,
) -> tuple[corpora.Dictionary, list[list[tuple[int, int]]], list[list[str]]]:
    """Build Dictionary, apply filter_extremes, return corpus + docs used."""
    if lda_vocab_exclude:
        allowed = None  # determined after first pass
        docs = [
            [t for t in doc if t not in lda_vocab_exclude]
            for doc in tokenized_docs
        ]
    else:
        docs = tokenized_docs

    if config["drop_empty_docs"]:
        docs = [d for d in docs if len(d) >= config["min_tokens_per_doc"]]

    dictionary = corpora.Dictionary(docs)
    dictionary.filter_extremes(
        no_below=config["min_doc_freq"],
        no_above=config["max_doc_freq"],
        keep_n=config["keep_n"],
    )

    corpus = [dictionary.doc2bow(doc) for doc in docs]
    return dictionary, corpus, docs


def validate_corpus(
    dictionary: corpora.Dictionary,
    corpus: list[list[tuple[int, int]]],
    tokenized_docs: list[list[str]],
) -> dict[str, Any]:
    """Light validation before modeling."""
    empty = sum(1 for doc in corpus if len(doc) == 0)
    return {
        "n_documents": len(tokenized_docs),
        "n_nonempty_bow_docs": len(corpus) - empty,
        "n_empty_bow_docs": empty,
        "vocabulary_size": len(dictionary),
        "total_token_instances": sum(len(d) for d in tokenized_docs),
    }


# ---------------------------------------------------------------------------
# Single entry point
# ---------------------------------------------------------------------------
def preprocess_corpus(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    End-to-end preprocessing for LDA.

    Returns
    -------
    dict with keys:
        documents      – raw comment strings
        tokenized_docs – list of token lists (post vocab filter)
        lda_docs       – token lists after lda_exclude_tokens (if any)
        dictionary     – gensim Dictionary
        corpus         – list of BoW vectors
        vocab          – set of retained type-level tokens
        stats          – validation summary
        config         – resolved config used
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    # Resources
    if Path(cfg["custom_words_file"]).exists():
        add_words_from_file(cfg["custom_words_file"])

    stopwords = set(_load_word_list(cfg["stop_words_file"]))
    filtered_words = set(_load_word_list(cfg["filtered_words_file"]))
    low_frequency = set(_load_word_list(cfg["low_frequency_file"]))

    replacement_rules: dict[str, str] = {}
    megatoken: dict[str, str] = {}
    add_replacements(cfg["replacement_rules_file"], replacement_rules)
    add_replacements(cfg["megatoken_file"], megatoken)

    cc = OpenCC("t2s") if cfg["simplify_traditional"] else None
    simplify_fn = (lambda s: cc.convert(s)) if cc else (lambda s: s)

    # 1. Ingestion
    df = load_documents(cfg)
    comments = df[cfg["text_column"]].astype(str).tolist()

    # 2. Tokenization
    raw_tokenized = tokenize_documents(
        comments,
        cfg,
        stopwords=stopwords,
        filtered_words=filtered_words,
        low_frequency=low_frequency,
        replacement_rules=replacement_rules,
        megatoken=megatoken,
        simplify_fn=simplify_fn,
    )

    # 3. Vocabulary filter (corpus-wide frequency)
    vocab, tokenized_docs = build_vocabulary(raw_tokenized, cfg)

    # 4. LDA-specific vocabulary trim + Gensim objects
    lda_exclude = set(cfg.get("lda_exclude_tokens") or [])
    dictionary, corpus, lda_docs = build_gensim_corpus(
        tokenized_docs,
        cfg,
        lda_vocab_exclude=lda_exclude if lda_exclude else None,
    )

    stats = validate_corpus(dictionary, corpus, lda_docs)

    return {
        "documents": comments,
        "tokenized_docs": tokenized_docs,
        "lda_docs": lda_docs,
        "dictionary": dictionary,
        "corpus": corpus,
        "vocab": vocab,
        "stats": stats,
        "config": cfg,
    }


def format_bow_doc(
    bow: list[tuple[int, int]],
    dictionary: corpora.Dictionary,
) -> list[tuple[str, int]]:
    """Human-readable BoW: (Chinese token, count)."""
    return [(dictionary[i], c) for i, c in bow]
