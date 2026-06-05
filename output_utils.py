"""
CSV output utilities for LDA results.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from gensim.models import LdaModel


def _ensure_parent_dir(filepath: str | Path) -> Path:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_topic_tokens(
    model: LdaModel,
    num_topics: int,
    *,
    top_n: int = 15,
    filename: str = "topic_tokens.csv",
) -> Path:
    """
    Extract top words per topic and save to CSV.

    Columns: topic_id, rank, token, weight
    """
    rows: list[dict] = []
    for topic_id in range(num_topics):
        topic_words = model.show_topic(topic_id, topn=top_n)
        for rank, (token, weight) in enumerate(topic_words, start=1):
            rows.append(
                {
                    "topic_id": topic_id + 1,
                    "rank": rank,
                    "token": token,
                    "weight": round(weight, 6),
                }
            )

    df = pd.DataFrame(rows)
    out_path = _ensure_parent_dir(filename)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def save_top_comments(
    model: LdaModel,
    corpus: list[list[tuple[int, int]]],
    original_texts: list[str],
    num_topics: int,
    *,
    top_n: int = 5,
    filename: str = "top_comments.csv",
    minimum_probability: float = 0.0,
) -> Path:
    """
    Save the most representative comments per topic.

    Each corpus row is mapped to the comment at the same index in
    original_texts. Columns: topic_id, rank, doc_index, topic_weight, comment
    """
    if len(corpus) != len(original_texts):
        raise ValueError(
            f"corpus length ({len(corpus)}) must match original_texts "
            f"({len(original_texts)})"
        )

    rows: list[dict] = []
    doc_topic_probs: list[list[float]] = []

    for doc in corpus:
        doc_topics = model.get_document_topics(doc, minimum_probability=minimum_probability)
        doc_topic_probs.append([prob for _, prob in doc_topics])

    topic_df = pd.DataFrame(doc_topic_probs)
    topic_df["comment"] = original_texts
    topic_df["doc_index"] = range(len(original_texts))

    for topic_id in range(num_topics):
        col = topic_id
        sorted_docs = topic_df.sort_values(by=col, ascending=False).head(top_n)
        for rank, (_, row) in enumerate(sorted_docs.iterrows(), start=1):
            rows.append(
                {
                    "topic_id": topic_id + 1,
                    "rank": rank,
                    "doc_index": int(row["doc_index"]),
                    "topic_weight": round(float(row[col]), 6),
                    "comment": row["comment"],
                }
            )

    df = pd.DataFrame(rows)
    out_path = _ensure_parent_dir(filename)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path
