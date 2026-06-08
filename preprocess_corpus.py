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
    "data_dir": "/Users/xiangningxu/Documents/vibe_coding/Scraping",
    "file_pattern": "comments_sampled_*.csv",
    "text_column": "comment_text",
    "vote_values": None,  # e.g. [1, 2]; None keeps all rows

    # --- Resource files (paths relative to project root or absolute) ---
    "custom_words_file": "custom_words.txt",
    "stop_words_file": "stop_words.txt",
    "filtered_words_file": "filtered_words.txt",
    "low_frequency_file": "low_frequency.txt",
    "replacement_rules_file": "replacement_rules2.txt",
    "megatoken_file": "megatoken.txt",

    # --- Comment filtering (before tokenization) ---
    "deduplicate_comments": True,
    "max_emojis_per_comment": 10,
    "max_links_per_comment": 1,  # drop comments with more than this many webpage links

    # --- Text cleaning ---
    "min_chunk_len": 3,
    "simplify_traditional": True,
    "min_consecutive_dup_run": 3,  # collapse N+ identical consecutive tokens to one
    "bilibili_sticker_words": {"辣眼睛", "藏狐", "吃瓜"},

    # --- Token / vocabulary filtering ---
    "min_token_freq": 2,          # keep tokens appearing more than this count corpus-wide
    "exclude_tokens": set(),      # hard-coded removals before dictionary step
    "lda_exclude_tokens": {       # removed only for LDA vocabulary (optional second pass)
        "孩子", "小孩子", "生孩子", "没有", "不想", "不生", "问题", "时候",
    },

    # --- Gensim Dictionary.filter_extremes ---
    "min_doc_freq": 2,            # no_below
    "max_doc_freq": 0.8,         # no_above (proportion of documents)
    "keep_n": None,               # None: use all tokens; int: keep top-N by frequency

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


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _is_noise_token(token: str, custom_words: set[str] | None = None) -> bool:
    """Drop punctuation, digit-only, and bare ASCII symbol tokens."""
    if not token or not token.strip():
        return True
    if custom_words and token in custom_words:
        return False
    if _CJK_RE.search(token):
        return False
    if token.isdigit():
        return True
    if re.fullmatch(r"[\W_]+", token):
        return True
    if len(token) == 1 and ord(token) < 128 and not token.isalnum():
        return True
    return False


def _count_emojis(text: str) -> int:
    """Count Unicode emojis and Bilibili-style bracket tags like [微笑]."""
    unicode_count = emoji.emoji_count(text)
    bracket_count = len(re.findall(r"\[[^\[\]]+\]", text))
    return unicode_count + bracket_count


_WEBPAGE_LINK_RE = re.compile(
    r"https?://[^\s\]\)）」』\"'<>，,。；;]+|www\.[^\s\]\)）」』\"'<>，,。；;]+",
    re.IGNORECASE,
)


def _count_webpage_links(text: str) -> int:
    """Count http(s):// and www. webpage links in comment text."""
    return len(_WEBPAGE_LINK_RE.findall(text))


def prepare_comments(
    comments: list[str],
    config: dict[str, Any],
) -> tuple[list[str], dict[str, int]]:
    """
    Drop emoji-heavy, multi-link, and duplicate comments before tokenization.

    Order: emoji filter, link filter, then deduplication (keeps first occurrence).
    """
    stats = {
        "n_loaded": len(comments),
        "n_dropped_emojis": 0,
        "n_dropped_multi_link": 0,
        "n_dropped_duplicates": 0,
        "n_kept": 0,
    }

    max_emojis = config.get("max_emojis_per_comment")
    if max_emojis is not None:
        kept: list[str] = []
        for comment in comments:
            if _count_emojis(comment) > max_emojis:
                stats["n_dropped_emojis"] += 1
            else:
                kept.append(comment)
        comments = kept

    max_links = config.get("max_links_per_comment")
    if max_links is not None:
        kept = []
        for comment in comments:
            if _count_webpage_links(comment) > max_links:
                stats["n_dropped_multi_link"] += 1
            else:
                kept.append(comment)
        comments = kept

    if config.get("deduplicate_comments", True):
        seen: set[str] = set()
        unique: list[str] = []
        for comment in comments:
            if comment in seen:
                stats["n_dropped_duplicates"] += 1
            else:
                seen.add(comment)
                unique.append(comment)
        comments = unique

    stats["n_kept"] = len(comments)
    return comments, stats


def _convert_emoji(sentence: str) -> str:
    sentence = emoji.demojize(sentence)
    sentence = re.sub(r":cow_face::horse_face:", "牛马", sentence)
    sentence = re.sub(r"\[[^\[\]]*\]", "", sentence)
    return sentence


def _strip_sticker_words(sentence: str, sticker_words: set[str]) -> str:
    """Remove Bilibili sticker vocabulary left as literal text after bracket stripping."""
    for word in sorted(sticker_words, key=len, reverse=True):
        sentence = sentence.replace(word, "")
    return sentence


def _preprocess_text(sentence: str) -> str:
    sentence = re.sub(r"回复|[\u4e00-\u9fff]@[^:]*:", "", sentence)
    sentence = re.sub(r"@[^:]* ", "", sentence)
    sentence = re.sub(r"[\s]", "，", sentence)
    sentence = re.sub(r"[\n\t\s]*", "", sentence)
    sentence = re.sub(r"我觉得", "", sentence)
    sentence = re.sub(r"[“+”（）]", "", sentence)
    sentence = re.sub(r"%", "百分之", sentence)
    # Strip remaining punctuation/symbols; keep Chinese, letters, and digits.
    sentence = re.sub(r"[^\w]", "", sentence)
    return sentence


def _split_chunks(sentence: str, min_len: int) -> list[str]:
    chunks = re.split(r"[。！？?，, （）\[\];:、……]", sentence)
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    return [chunk for chunk in chunks if len(chunk) > min_len]


def _refine_tokens(
    chunk_texts: list[str],
    custom_words: set[str] | None = None,
    sticker_words: set[str] | None = None,
) -> list[str]:
    """Pattern-based token refinement (replaces former Stanza step)."""
    result: list[str] = []
    custom_words = custom_words or set()
    sticker_words = sticker_words or set()
    child_words = {
        "孩子", "小孩", "宝宝", "小孩儿", "崽", "崽崽", "男孩", "女孩",
        "崽子", "儿子", "一个人", "宠物", "猫", "狗",
    }
    verb_words = {"给", "有", "生", "没", "抱", "带", "养", "爱", "不要", "要"}
    keep_flags = {"n", "v", "a", "vn", "an", "nr", "nt", "nz", "t"}

    for token_text in chunk_texts:
        if not token_text.strip():
            continue
        # Re-tag chunk tokens individually so custom-dictionary words stay intact.
        pos_tokens = []
        for word in token_text.split():
            if (
                not word.strip()
                or _is_noise_token(word, custom_words)
                or word in sticker_words
            ):
                continue
            tagged = list(pseg.cut(word))
            pos_tokens.extend(tagged if tagged else [])
        pattern_tokens: list[str] = []
        i = 0
        while i < len(pos_tokens):
            word, flag = pos_tokens[i].word, pos_tokens[i].flag
            if _is_noise_token(word, custom_words) or word in sticker_words:
                i += 1
                continue
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
            if flag in keep_flags or (flag == "x" and word in custom_words):
                pattern_tokens.append(word)
            i += 1
        result.extend(pattern_tokens)
    return result


def _tokenize_chunk(
    text: str,
    stopwords: set[str],
    filtered_words: set[str],
    low_frequency: set[str],
    custom_words: set[str] | None = None,
    sticker_words: set[str] | None = None,
) -> str:
    custom_words = custom_words or set()
    sticker_words = sticker_words or set()
    tokens = [word.word for word in pseg.cut(text)]
    tokens = [t for t in tokens if t not in stopwords]
    tokens = [t for t in tokens if t not in filtered_words]
    tokens = [t for t in tokens if t not in low_frequency]
    tokens = [t for t in tokens if t not in sticker_words]
    tokens = [t for t in tokens if not _is_noise_token(t, custom_words)]
    return " ".join(tokens)


def _apply_replacement_dict(tokens: list[str], rules: dict[str, str]) -> list[str]:
    return [rules.get(t, t) for t in tokens]


def _collapse_consecutive_duplicate_tokens(
    tokens: list[str],
    min_run: int = 3,
) -> list[str]:
    """Collapse consecutive identical tokens or token sequences repeated min_run+ times."""
    if not tokens or min_run < 2:
        return tokens

    # Pass 1: identical single-token runs (e.g. 好可爱 好可爱 好可爱 -> 好可爱)
    collapsed: list[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        j = i + 1
        while j < n and tokens[j] == tokens[i]:
            j += 1
        run_len = j - i
        if run_len >= min_run:
            collapsed.append(tokens[i])
        else:
            collapsed.extend(tokens[i:j])
        i = j

    # Pass 2: repeating multi-token patterns (e.g. 好 可爱 好 可爱 好 可爱 -> 好 可爱)
    n = len(collapsed)
    if n < min_run:
        return collapsed

    out: list[str] = []
    i = 0
    while i < n:
        matched = False
        max_pat_len = (n - i) // min_run
        for pat_len in range(1, max_pat_len + 1):
            pattern = collapsed[i : i + pat_len]
            run = 1
            pos = i + pat_len
            while pos + pat_len <= n and collapsed[pos : pos + pat_len] == pattern:
                run += 1
                pos += pat_len
            if run >= min_run:
                out.extend(pattern)
                i = pos
                matched = True
                break
        if not matched:
            out.append(collapsed[i])
            i += 1
    return out


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
    custom_words: set[str] | None = None,
) -> list[list[str]]:
    """Run per-document cleaning, chunking, tokenization, and refinement."""
    min_chunk = config["min_chunk_len"]
    sticker_words = set(config.get("bilibili_sticker_words") or ())
    docs: list[list[str]] = []

    for comment in comments:
        sentence = _convert_emoji(comment)
        sentence = _preprocess_text(sentence)
        sentence = _strip_sticker_words(sentence, sticker_words)
        if config["simplify_traditional"]:
            sentence = simplify_fn(sentence)

        chunks = _split_chunks(sentence, min_chunk)
        chunk_strings = [
            _tokenize_chunk(
                c,
                stopwords,
                filtered_words,
                low_frequency,
                custom_words,
                sticker_words,
            )
            for c in chunks
        ]
        tokens = _refine_tokens(chunk_strings, custom_words, sticker_words)
        tokens = _apply_replacement_dict(tokens, replacement_rules)
        tokens = _apply_replacement_dict(tokens, megatoken)
        tokens = _collapse_consecutive_duplicate_tokens(
            tokens,
            min_run=config.get("min_consecutive_dup_run", 3),
        )
        tokens = [t for t in tokens if t not in sticker_words]
        docs.append(tokens)

    return docs


def strip_blocked_tokens(
    tokenized_docs: list[list[str]],
    *,
    stopwords: set[str],
    filtered_words: set[str],
) -> list[list[str]]:
    """Remove stop_words and filtered_words before LDA vocabulary/corpus construction."""
    blocked = stopwords | filtered_words
    return [[t for t in doc if t not in blocked] for doc in tokenized_docs]


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
    filter_kwargs: dict[str, Any] = {
        "no_below": config["min_doc_freq"],
        "no_above": config["max_doc_freq"],
    }
    keep_n = config.get("keep_n")
    if keep_n is not None:
        filter_kwargs["keep_n"] = keep_n
    dictionary.filter_extremes(**filter_kwargs)

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
    custom_words: set[str] = set()
    if Path(cfg["custom_words_file"]).exists():
        custom_words = set(_load_word_list(cfg["custom_words_file"]))
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

    # 2. Drop emoji-heavy and duplicate comments before cleaning/tokenization
    comments, filter_stats = prepare_comments(comments, cfg)

    # 3. Tokenization
    raw_tokenized = tokenize_documents(
        comments,
        cfg,
        stopwords=stopwords,
        filtered_words=filtered_words,
        low_frequency=low_frequency,
        replacement_rules=replacement_rules,
        megatoken=megatoken,
        simplify_fn=simplify_fn,
        custom_words=custom_words,
    )

    # 4. Final strip of stop_words / filtered_words (catches tokens reintroduced in refine)
    raw_tokenized = strip_blocked_tokens(
        raw_tokenized,
        stopwords=stopwords,
        filtered_words=filtered_words,
    )

    # 5. Vocabulary filter (corpus-wide frequency)
    vocab, tokenized_docs = build_vocabulary(raw_tokenized, cfg)

    # 6. LDA-specific vocabulary trim + Gensim objects
    lda_exclude = set(cfg.get("lda_exclude_tokens") or [])
    dictionary, corpus, lda_docs = build_gensim_corpus(
        tokenized_docs,
        cfg,
        lda_vocab_exclude=lda_exclude if lda_exclude else None,
    )

    # Align raw documents with corpus rows after LDA trim / empty-doc removal
    aligned_comments: list[str] = []
    for comment, tokens in zip(comments, tokenized_docs):
        trimmed = [t for t in tokens if t not in lda_exclude] if lda_exclude else tokens
        if cfg["drop_empty_docs"] and len(trimmed) < cfg["min_tokens_per_doc"]:
            continue
        aligned_comments.append(comment)
    comments = aligned_comments

    stats = validate_corpus(dictionary, corpus, lda_docs)
    stats.update(filter_stats)

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
