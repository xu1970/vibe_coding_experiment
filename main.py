"""
Entry point for fertility-comment LDA analysis.

Edit the configuration block below to change data paths and hyper-parameters.
"""

from __future__ import annotations

from pathlib import Path

from lda_utils import LDASettings, print_lda_topics, print_topic_tokens, run_lda_pipeline
from output_utils import save_top_comments, save_topic_tokens
from preprocess_corpus import DEFAULT_CONFIG, preprocess_corpus

# =============================================================================
# Configuration — change paths and parameters here
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = "/Users/xiangningxu/Documents/vibe_coding/Scraping"
FILE_PATTERN = "comments_sampled_*.csv"
TEXT_COLUMN = "comment_text"
VOTE_VALUES = None  # e.g. [1, 2]; None keeps all rows

# Resource files (relative to project root)
CUSTOM_WORDS_FILE = PROJECT_ROOT / "custom_words.txt"
STOP_WORDS_FILE = PROJECT_ROOT / "stop_words.txt"
FILTERED_WORDS_FILE = PROJECT_ROOT / "filtered_words.txt"
LOW_FREQUENCY_FILE = PROJECT_ROOT / "low_frequency.txt"
REPLACEMENT_RULES_FILE = PROJECT_ROOT / "replacement_rules2.txt"
MEGATOKEN_FILE = PROJECT_ROOT / "megatoken.txt"

# Preprocessing filters
DEDUPLICATE_COMMENTS = True
MAX_EMOJIS_PER_COMMENT = 10
MAX_LINKS_PER_COMMENT = 1  # drop comments with more than one webpage link
MIN_CHUNK_LEN = 3
MIN_CONSECUTIVE_DUP_RUN = 3
MIN_TOKEN_FREQ = 2
LDA_EXCLUDE_TOKENS = {
    "孩子", "小孩子", "生孩子", "没有", "不想", "不生", "问题", "时候",
}
MIN_DOC_FREQ = 80
MAX_DOC_FREQ = 0.7
KEEP_N = None  # int: keep top-N tokens by frequency; None: use all tokens after preprocessing

# LDA hyper-parameters
NUM_TOPICS = 25
PASSES = 25
RANDOM_STATE = 42
TOPN_TOPICS_PRINT = 10
TOPN_TOPIC_TOKENS = 20
TOPN_DOCS_PER_TOPIC = 7

# CSV output settings
OUTPUT_DIR = PROJECT_ROOT / "outputs"
NUM_TOPIC_TOKENS = 15
NUM_TOP_COMMENTS = 5
TOPIC_TOKENS_FILENAME = OUTPUT_DIR / "topic_tokens.csv"
TOP_COMMENTS_FILENAME = OUTPUT_DIR / "top_comments.csv"


def build_preprocess_config() -> dict:
    """Assemble preprocessing config from module-level settings."""
    cfg = {
        **DEFAULT_CONFIG,
        "data_dir": DATA_DIR,
        "file_pattern": FILE_PATTERN,
        "text_column": TEXT_COLUMN,
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
        "min_chunk_len": MIN_CHUNK_LEN,
        "min_consecutive_dup_run": MIN_CONSECUTIVE_DUP_RUN,
        "min_token_freq": MIN_TOKEN_FREQ,
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
        print(f"  {key}: {value}")

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

    lda_result = run_lda_pipeline(
        dictionary=dictionary,
        corpus=corpus,
        documents=documents,
        settings=build_lda_settings(),
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
    print(f"  topic tokens: {topic_tokens_path}")
    print(f"  top comments: {top_comments_path}")

    return {
        "preprocessed": preprocessed,
        "lda": lda_result,
        "outputs": {
            "topic_tokens_csv": topic_tokens_path,
            "top_comments_csv": top_comments_path,
        },
    }


if __name__ == "__main__":
    main()
