"""
Topic-count sweep: preprocess once, then train LDA for K in [18, 45] with 3 seeds each.

Reuses preprocessing / BoW corpus from main.py settings. Writes two CSVs per K:
  - topic_tokens_k{K}.csv   — top tokens for every topic in each of 3 runs
  - top_comments_k{K}.csv   — top comments for every topic in each of 3 runs
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import pandas as pd
from gensim.models import LdaModel

from lda_utils import LDASettings, run_lda_pipeline
from main import (
    LIKES_ALPHA,
    LIKES_USE_LOG,
    NUM_TOP_COMMENTS,
    NUM_TOPIC_TOKENS,
    PASSES,
    PROJECT_ROOT,
    RANDOM_STATE,
    REWEIGHT_BY_LIKES,
    build_preprocess_config,
)
from preprocess_corpus import preprocess_corpus

# =============================================================================
# Sweep settings
# =============================================================================

TOPIC_MIN = 18
TOPIC_MAX = 45  # inclusive
NUM_RUNS = 3
RUN_SEEDS = [RANDOM_STATE + i for i in range(NUM_RUNS)]

SWEEP_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "topic_sweep"


def _lda_settings_for_run(*, num_topics: int, random_state: int) -> LDASettings:
    return LDASettings(
        num_topics=num_topics,
        passes=PASSES,
        random_state=random_state,
    )


def _topic_token_rows(
    model: LdaModel,
    *,
    run_id: int,
    num_topics: int,
    top_n: int,
) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    for topic_id in range(num_topics):
        topic_words = model.show_topic(topic_id, topn=top_n)
        tokens_str = " ".join(word for word, _ in topic_words)
        rows.append(
            {
                "run": run_id,
                "seed": RUN_SEEDS[run_id - 1],
                "topic_id": topic_id + 1,
                "tokens": tokens_str,
            }
        )
    return rows


def _top_comment_rows(
    model: LdaModel,
    corpus: list[list[tuple[int, int]]],
    documents: list[str],
    *,
    run_id: int,
    num_topics: int,
    top_n: int,
    minimum_probability: float = 0.0,
) -> list[dict[str, int | float | str]]:
    doc_topic_probs: list[list[float]] = []
    for doc in corpus:
        doc_topics = model.get_document_topics(doc, minimum_probability=minimum_probability)
        doc_topic_probs.append([prob for _, prob in doc_topics])

    topic_df = pd.DataFrame(doc_topic_probs)
    topic_df["comment"] = documents
    topic_df["doc_index"] = range(len(documents))

    rows: list[dict[str, int | float | str]] = []
    for topic_id in range(num_topics):
        col = topic_id
        sorted_docs = topic_df.sort_values(by=col, ascending=False).head(top_n)
        for rank, (_, row) in enumerate(sorted_docs.iterrows(), start=1):
            rows.append(
                {
                    "run": run_id,
                    "seed": RUN_SEEDS[run_id - 1],
                    "topic_id": topic_id + 1,
                    "rank": rank,
                    "doc_index": int(row["doc_index"]),
                    "topic_weight": round(float(row[col]), 6),
                    "comment": row["comment"],
                }
            )
    return rows


def save_sweep_topic_tokens(
    models: list[LdaModel],
    *,
    num_topics: int,
    top_n: int,
    filename: Path,
) -> Path:
    """Combine top tokens from each LDA run into one CSV (run column distinguishes runs)."""
    rows: list[dict[str, int | str]] = []
    for run_id, model in enumerate(models, start=1):
        rows.extend(
            _topic_token_rows(model, run_id=run_id, num_topics=num_topics, top_n=top_n)
        )

    filename.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(filename, index=False, encoding="utf-8-sig")
    return filename


def save_sweep_top_comments(
    models: list[LdaModel],
    corpus: list[list[tuple[int, int]]],
    documents: list[str],
    *,
    num_topics: int,
    top_n: int,
    filename: Path,
) -> Path:
    """Combine top comments from each LDA run into one CSV."""
    filename.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run", "seed", "topic_id", "rank", "doc_index", "topic_weight", "comment"]

    with open(filename, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for run_id, model in enumerate(models, start=1):
            if run_id > 1:
                f.write("\n")

            rows = _top_comment_rows(
                model,
                corpus,
                documents,
                run_id=run_id,
                num_topics=num_topics,
                top_n=top_n,
            )
            for row in rows:
                writer.writerow(row)

    return filename


def run_sweep() -> dict[int, dict[str, Path]]:
    """Preprocess once, sweep topic counts, return output paths keyed by K."""
    print("=" * 80)
    print("Step 1: Preprocess corpus (once)")
    print("=" * 80)
    t0 = time.perf_counter()
    preprocessed = preprocess_corpus(build_preprocess_config())
    print(f"  documents: {len(preprocessed['documents'])}")
    print(f"  vocabulary: {len(preprocessed['dictionary'])}")
    print(f"  elapsed: {time.perf_counter() - t0:.1f}s")

    dictionary = preprocessed["dictionary"]
    corpus = preprocessed["corpus"]
    documents = preprocessed["documents"]
    document_likes = preprocessed.get("document_likes")

    if REWEIGHT_BY_LIKES and document_likes is None:
        raise ValueError("REWEIGHT_BY_LIKES is enabled but document_likes was not loaded")

    outputs: dict[int, dict[str, Path]] = {}
    topic_range = range(TOPIC_MIN, TOPIC_MAX + 1)
    total_fits = len(topic_range) * NUM_RUNS
    fit_idx = 0

    print("\n" + "=" * 80)
    print(f"Step 2: LDA sweep K={TOPIC_MIN}..{TOPIC_MAX}, {NUM_RUNS} runs each")
    print(f"  seeds: {RUN_SEEDS}")
    print(f"  output: {SWEEP_OUTPUT_DIR}")
    print("=" * 80)

    for num_topics in topic_range:
        models: list[LdaModel] = []
        k_t0 = time.perf_counter()

        for run_id, seed in enumerate(RUN_SEEDS, start=1):
            fit_idx += 1
            print(f"  [{fit_idx}/{total_fits}] K={num_topics} run={run_id} seed={seed} ...", end=" ", flush=True)
            run_t0 = time.perf_counter()

            lda_result = run_lda_pipeline(
                dictionary=dictionary,
                corpus=corpus,
                documents=documents,
                settings=_lda_settings_for_run(num_topics=num_topics, random_state=seed),
                reweight_by_likes=REWEIGHT_BY_LIKES,
                likes_alpha=LIKES_ALPHA,
                likes_use_log=LIKES_USE_LOG,
                document_likes=document_likes,
            )
            models.append(lda_result["lda_model"])
            print(f"{time.perf_counter() - run_t0:.1f}s")

        tokens_path = save_sweep_topic_tokens(
            models,
            num_topics=num_topics,
            top_n=NUM_TOPIC_TOKENS,
            filename=SWEEP_OUTPUT_DIR / f"topic_tokens_k{num_topics}.csv",
        )
        comments_path = save_sweep_top_comments(
            models,
            corpus,
            documents,
            num_topics=num_topics,
            top_n=NUM_TOP_COMMENTS,
            filename=SWEEP_OUTPUT_DIR / f"top_comments_k{num_topics}.csv",
        )
        outputs[num_topics] = {
            "topic_tokens": tokens_path,
            "top_comments": comments_path,
        }
        print(
            f"  K={num_topics} saved → {tokens_path.name}, {comments_path.name} "
            f"({time.perf_counter() - k_t0:.1f}s)"
        )

    print("\n" + "=" * 80)
    print(f"Sweep complete — {len(outputs)} topic counts, {total_fits} LDA fits")
    print("=" * 80)
    return outputs


if __name__ == "__main__":
    run_sweep()
