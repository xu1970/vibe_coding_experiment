"""
Entry point for fertility-comment LDA analysis.

Edit the configuration block below to change data paths and hyper-parameters.
"""

from __future__ import annotations

from pathlib import Path

from lda_utils import LDASettings, print_lda_topics, print_topic_tokens, run_lda_pipeline
from output_utils import (
    save_lda_model,
    save_top_comments,
    save_topic_coordinates,
    save_topic_tokens,
)
from preprocess_corpus import DEFAULT_CONFIG, preprocess_corpus

# =============================================================================
# Configuration — change paths and parameters here
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = "/Users/xiangningxu/Documents/vibe_coding/Scraping"
FILE_PATTERN = "comments_sampled_*育*.csv"
TEXT_COLUMN = "comment_text"
VOTE_VALUES = None  # e.g. [1, 2]; None keeps all rows

# Resource files (under preprocess_lists/)
PREPROCESS_LISTS_DIR = PROJECT_ROOT / "preprocess_lists"
CUSTOM_WORDS_FILE = PREPROCESS_LISTS_DIR / "custom_words.txt"
STOP_WORDS_FILE = PREPROCESS_LISTS_DIR / "stop_words.txt"
FILTERED_WORDS_FILE = PREPROCESS_LISTS_DIR / "filtered_words.txt"
LOW_FREQUENCY_FILE = PREPROCESS_LISTS_DIR / "low_frequency.txt"
REPLACEMENT_RULES_FILE = PREPROCESS_LISTS_DIR / "replacement_rules2.txt"
MEGATOKEN_FILE = PREPROCESS_LISTS_DIR / "megatoken.txt"
PRODUCT_SALES_TERMS_FILE = PREPROCESS_LISTS_DIR / "product_sales_terms.txt"
AI_COMMENT_TERMS_FILE = PREPROCESS_LISTS_DIR / "ai_comment_terms.txt"
BLOCKED_COMMENT_TERMS_FILE = PREPROCESS_LISTS_DIR / "blocked_comment_terms.txt"
ENGAGEMENT_SPAM_TERMS_FILE = PREPROCESS_LISTS_DIR / "engagement_spam_terms.txt"
REPETITION_SPAM_TERMS_FILE = PREPROCESS_LISTS_DIR / "repetition_spam_terms.txt"
META_DISCUSSION_TERMS_FILE = PREPROCESS_LISTS_DIR / "meta_discussion_terms.txt"

# Preprocessing filters
DEDUPLICATE_COMMENTS = True
MAX_EMOJIS_PER_COMMENT = 10
MAX_LINKS_PER_COMMENT = 1  # drop comments with more than one webpage link
FILTER_BRACKET_SPAM_COMMENTS = True
MAX_BRACKET_PAIRS_PER_COMMENT = 3  # drop if <> or 《》 pair count exceeds this (>3 → 4+ pairs)
FILTER_BLOCKED_COMMENT_TERMS = True
MAX_BLOCKED_TERM_MENTIONS = 1  # drop if any listed term appears more than this (>3 → 4+ hits)
FILTER_PRODUCT_SALES_COMMENTS = True
MAX_PRODUCT_TERM_MENTIONS = 2  # drop if product/sales term count exceeds this (>2 → 3+ hits)
FILTER_AI_COMMENTS = True
FILTER_SHORT_ENGAGEMENT_COMMENTS = True
SHORT_ENGAGEMENT_MIN_TOKENS = 10  # drop only when comment has fewer than this many jieba tokens
SHORT_ENGAGEMENT_MAX_TERM_MENTIONS = 1  # drop if 点赞/赞同/关注 hits exceed this (>1 → 2+)
FILTER_REPETITION_SPAM_COMMENTS = True
MAX_REPETITION_TERM_MENTIONS = 7  # drop if any listed term appears more than this (>10 → 11+)
FILTER_META_DISCUSSION_COMMENTS = True
META_DISCUSSION_MIN_TERM_MENTIONS = 5  # drop if combined 数据/臆测/捏造/观点/神棍 hits reach this
MIN_CHUNK_LEN = 3
MIN_CONSECUTIVE_DUP_RUN = 3
MIN_TOKEN_FREQ = 2
FILTER_SINGLE_CHAR_TOKENS = True
SINGLE_CHAR_TOKENS_FILE = PREPROCESS_LISTS_DIR / "single_char_tokens.txt"
SINGLE_CHAR_TOP_PCT = 20
SINGLE_CHAR_MID_LOW_PCT = 20
SINGLE_CHAR_MID_HIGH_PCT = 80
LDA_EXCLUDE_TOKENS = {
    "孩子", "小孩子", "生孩子", "没有", "不想", "不生", "问题", "时候",
}
MIN_DOC_FREQ = 50
MAX_DOC_FREQ = 0.8
KEEP_N = None  # int: keep top-N tokens by frequency; None: use all tokens after preprocessing
LIKES_COLUMN = "like_count"  # Scraping CSVs; use "numlikes" for comments_oid323836485.csv

# Corpus reweighting before LDA
REWEIGHT_BY_LIKES = True
LIKES_USE_LOG = True  # False: weight = 1 + LIKES_ALPHA * num_likes; True: weight = 1 + log(num_likes)
LIKES_ALPHA = 0.1      # used only when LIKES_USE_LOG is False

# LDA hyper-parameters
NUM_TOPICS = 19
PASSES = 20
RANDOM_STATE = 46
TOPN_TOPICS_PRINT = 10
TOPN_TOPIC_TOKENS = 20
TOPN_DOCS_PER_TOPIC = 7

# CSV output settings
OUTPUT_DIR = PROJECT_ROOT / "outputs"
NUM_TOPIC_TOKENS = 15
NUM_TOP_COMMENTS = 10
TOPIC_TOKENS_FILENAME = OUTPUT_DIR / "topic_tokens.csv"
TOP_COMMENTS_FILENAME = OUTPUT_DIR / "top_comments.csv"
LDA_MODEL_FILENAME = OUTPUT_DIR / "model" / "lda.model"
TOPIC_COORDINATES_FILENAME = OUTPUT_DIR / "topic_coordinates.csv"
MDS_METHOD = "mmds"  # topic distance projection: 'mmds', 'pcoa', or 'tsne'


def build_preprocess_config() -> dict:
    """Assemble preprocessing config from module-level settings."""
    cfg = {
        **DEFAULT_CONFIG,
        "data_dir": DATA_DIR,
        "file_pattern": FILE_PATTERN,
        "text_column": TEXT_COLUMN,
        "likes_column": LIKES_COLUMN,
        "vote_values": VOTE_VALUES,
        "custom_words_file": str(CUSTOM_WORDS_FILE),
        "stop_words_file": str(STOP_WORDS_FILE),
        "filtered_words_file": str(FILTERED_WORDS_FILE),
        "low_frequency_file": str(LOW_FREQUENCY_FILE),
        "replacement_rules_file": str(REPLACEMENT_RULES_FILE),
        "megatoken_file": str(MEGATOKEN_FILE),
        "deduplicate_comments": DEDUPLICATE_COMMENTS,
        "max_emojis_per_comment": MAX_EMOJIS_PER_COMMENT,
        "max_links_per_comment": MAX_LINKS_PER_COMMENT,
        "filter_bracket_spam_comments": FILTER_BRACKET_SPAM_COMMENTS,
        "max_bracket_pairs_per_comment": MAX_BRACKET_PAIRS_PER_COMMENT,
        "filter_blocked_comment_terms": FILTER_BLOCKED_COMMENT_TERMS,
        "max_blocked_term_mentions": MAX_BLOCKED_TERM_MENTIONS,
        "blocked_comment_terms_file": str(BLOCKED_COMMENT_TERMS_FILE),
        "filter_product_sales_comments": FILTER_PRODUCT_SALES_COMMENTS,
        "max_product_term_mentions": MAX_PRODUCT_TERM_MENTIONS,
        "product_sales_terms_file": str(PRODUCT_SALES_TERMS_FILE),
        "filter_ai_comments": FILTER_AI_COMMENTS,
        "ai_comment_terms_file": str(AI_COMMENT_TERMS_FILE),
        "filter_short_engagement_comments": FILTER_SHORT_ENGAGEMENT_COMMENTS,
        "short_engagement_min_tokens": SHORT_ENGAGEMENT_MIN_TOKENS,
        "short_engagement_max_term_mentions": SHORT_ENGAGEMENT_MAX_TERM_MENTIONS,
        "engagement_spam_terms_file": str(ENGAGEMENT_SPAM_TERMS_FILE),
        "filter_repetition_spam_comments": FILTER_REPETITION_SPAM_COMMENTS,
        "max_repetition_term_mentions": MAX_REPETITION_TERM_MENTIONS,
        "repetition_spam_terms_file": str(REPETITION_SPAM_TERMS_FILE),
        "filter_meta_discussion_comments": FILTER_META_DISCUSSION_COMMENTS,
        "meta_discussion_min_term_mentions": META_DISCUSSION_MIN_TERM_MENTIONS,
        "meta_discussion_terms_file": str(META_DISCUSSION_TERMS_FILE),
        "min_chunk_len": MIN_CHUNK_LEN,
        "min_consecutive_dup_run": MIN_CONSECUTIVE_DUP_RUN,
        "min_token_freq": MIN_TOKEN_FREQ,
        "filter_single_char_tokens": FILTER_SINGLE_CHAR_TOKENS,
        "single_char_tokens_file": str(SINGLE_CHAR_TOKENS_FILE),
        "single_char_top_pct": SINGLE_CHAR_TOP_PCT,
        "single_char_mid_low_pct": SINGLE_CHAR_MID_LOW_PCT,
        "single_char_mid_high_pct": SINGLE_CHAR_MID_HIGH_PCT,
        "lda_exclude_tokens": LDA_EXCLUDE_TOKENS,
        "min_doc_freq": MIN_DOC_FREQ,
        "max_doc_freq": MAX_DOC_FREQ,
    }
    cfg["keep_n"] = KEEP_N  # None uses all tokens after preprocessing
    return cfg


def build_lda_settings() -> LDASettings:
    """Assemble LDA settings from module-level hyper-parameters."""
    return LDASettings(
        num_topics=NUM_TOPICS,
        passes=PASSES,
        random_state=RANDOM_STATE,
        topn_topics_print=TOPN_TOPICS_PRINT,
        topn_topic_tokens=TOPN_TOPIC_TOKENS,
        topn_docs_per_topic=TOPN_DOCS_PER_TOPIC,
    )


def main() -> dict:
    """Run preprocessing, train LDA, and print diagnostic summaries."""
    print("=" * 80)
    print("Step 1: Preprocess corpus")
    print("=" * 80)

    preprocessed = preprocess_corpus(build_preprocess_config())
    stats = preprocessed["stats"]
    for key, value in stats.items():
        if key.startswith("n_single_char_") or key == "single_char_report":
            continue
        print(f"  {key}: {value}")

    print(
        f"\n  Bracket-spam filter: {'ON' if FILTER_BRACKET_SPAM_COMMENTS else 'OFF'}"
        + (
            f" (drop if >{MAX_BRACKET_PAIRS_PER_COMMENT} <> or 《》 pairs)"
            if FILTER_BRACKET_SPAM_COMMENTS
            else ""
        )
    )
    print(
        f"  Blocked-term filter: {'ON' if FILTER_BLOCKED_COMMENT_TERMS else 'OFF'}"
        + (
            f" (drop if any listed term appears >{MAX_BLOCKED_TERM_MENTIONS} times)"
            if FILTER_BLOCKED_COMMENT_TERMS
            else ""
        )
    )
    print(
        f"  Product-sales filter: {'ON' if FILTER_PRODUCT_SALES_COMMENTS else 'OFF'}"
        + (
            f" (drop if >{MAX_PRODUCT_TERM_MENTIONS} term hits)"
            if FILTER_PRODUCT_SALES_COMMENTS
            else ""
        )
    )
    print(
        f"  AI-comment filter: {'ON' if FILTER_AI_COMMENTS else 'OFF'}"
    )
    print(
        f"  Short-engagement filter: {'ON' if FILTER_SHORT_ENGAGEMENT_COMMENTS else 'OFF'}"
        + (
            f" (drop if <{SHORT_ENGAGEMENT_MIN_TOKENS} tokens and"
            f" >{SHORT_ENGAGEMENT_MAX_TERM_MENTIONS} 点赞/赞同/关注 hits)"
            if FILTER_SHORT_ENGAGEMENT_COMMENTS
            else ""
        )
    )
    print(
        f"  Repetition-spam filter: {'ON' if FILTER_REPETITION_SPAM_COMMENTS else 'OFF'}"
        + (
            f" (drop if any listed term appears >{MAX_REPETITION_TERM_MENTIONS} times)"
            if FILTER_REPETITION_SPAM_COMMENTS
            else ""
        )
    )
    print(
        f"  Meta-discussion filter: {'ON' if FILTER_META_DISCUSSION_COMMENTS else 'OFF'}"
        + (
            f" (drop if combined 数据/臆测/捏造/观点/神棍 hits ≥{META_DISCUSSION_MIN_TERM_MENTIONS})"
            if FILTER_META_DISCUSSION_COMMENTS
            else ""
        )
    )

    dictionary = preprocessed["dictionary"]
    if KEEP_N is None:
        print(
            f"\n  keep_n not set — using all {stats['vocabulary_size']} "
            "tokens after preprocessing for LDA"
        )
    else:
        print(f"\n  keep_n={KEEP_N} — vocabulary capped to {stats['vocabulary_size']} tokens")
    corpus = preprocessed["corpus"]
    documents = preprocessed["documents"]
    lda_docs = preprocessed["lda_docs"]

    print("\nExample preprocessed document (tokens):")
    if lda_docs:
        print(f"  {lda_docs[0][:20]}")

    print("\n" + "=" * 80)
    print("Step 2: Train LDA")
    print("=" * 80)

    document_likes = preprocessed.get("document_likes")
    if REWEIGHT_BY_LIKES:
        if document_likes is None:
            raise ValueError(
                f"REWEIGHT_BY_LIKES is enabled but likes column "
                f"{LIKES_COLUMN!r} was not found or loaded"
            )
        if LIKES_USE_LOG:
            print("  Like reweighting: ON (log mode, weight = 1 + log(num_likes))")
        else:
            print(
                f"  Like reweighting: ON (scale mode, alpha={LIKES_ALPHA}, "
                f"weight = 1 + {LIKES_ALPHA} * num_likes)"
            )
    else:
        print("  Like reweighting: OFF")

    lda_result = run_lda_pipeline(
        dictionary=dictionary,
        corpus=corpus,
        documents=documents,
        settings=build_lda_settings(),
        reweight_by_likes=REWEIGHT_BY_LIKES,
        likes_alpha=LIKES_ALPHA,
        likes_use_log=LIKES_USE_LOG,
        document_likes=document_likes,
    )

    lda_model = lda_result["lda_model"]
    print(f"  topics: {lda_model.num_topics}")
    print(f"  vocabulary: {len(dictionary)}")

    print("\n" + "=" * 80)
    print("Step 3: Topic summaries")
    print("=" * 80)
    print_lda_topics(lda_model, num_words=TOPN_TOPICS_PRINT)

    print("\n" + "=" * 80)
    print("Step 5: Save CSV outputs")
    print("=" * 80)

    topic_tokens_path = save_topic_tokens(
        lda_model,
        NUM_TOPICS,
        top_n=NUM_TOPIC_TOKENS,
        filename=TOPIC_TOKENS_FILENAME,
    )
    top_comments_path = save_top_comments(
        lda_model,
        corpus,
        documents,
        NUM_TOPICS,
        top_n=NUM_TOP_COMMENTS,
        filename=TOP_COMMENTS_FILENAME,
    )
    model_path = save_lda_model(lda_model, filename=LDA_MODEL_FILENAME)
    coordinates_path = save_topic_coordinates(
        lda_model,
        lda_result["corpus_tfidf"],
        dictionary,
        filename=TOPIC_COORDINATES_FILENAME,
        mds=MDS_METHOD,
        sort_topics=False,
    )
    print(f"  topic tokens: {topic_tokens_path}")
    print(f"  top comments: {top_comments_path}")
    print(f"  lda model: {model_path}")
    print(f"  topic coordinates: {coordinates_path}")

    return {
        "preprocessed": preprocessed,
        "lda": lda_result,
        "outputs": {
            "topic_tokens_csv": topic_tokens_path,
            "top_comments_csv": top_comments_path,
            "lda_model": model_path,
            "topic_coordinates_csv": coordinates_path,
        },
    }


if __name__ == "__main__":
    main()
