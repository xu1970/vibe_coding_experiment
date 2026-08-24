"""
CSV output utilities for LDA results.
"""

from __future__ import annotations

import csv
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

    One row per topic. Top tokens are space-separated in rank order.
    Columns: topic_id, tokens
    """
    rows: list[dict[str, str | int]] = []
    for topic_id in range(num_topics):
        topic_words = model.show_topic(topic_id, topn=top_n)
        tokens_str = " ".join(word for word, _ in topic_words)
        rows.append({"topic_id": topic_id + 1, "tokens": tokens_str})

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

    Each corpus row maps to the comment at the same index in original_texts.
    A blank line separates comments from different topics.
    Columns: topic_id, rank, doc_index, topic_weight, comment
    """
    if len(corpus) != len(original_texts):
        raise ValueError(
            f"corpus length ({len(corpus)}) must match original_texts "
            f"({len(original_texts)})"
        )

    doc_topic_probs: list[list[float]] = []
    for doc in corpus:
        doc_topics = model.get_document_topics(doc, minimum_probability=minimum_probability)
        doc_topic_probs.append([prob for _, prob in doc_topics])

    topic_df = pd.DataFrame(doc_topic_probs)
    topic_df["comment"] = original_texts
    topic_df["doc_index"] = range(len(original_texts))

    out_path = _ensure_parent_dir(filename)
    fieldnames = ["topic_id", "rank", "doc_index", "topic_weight", "comment"]

    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for topic_id in range(num_topics):
            if topic_id > 0:
                f.write("\n")

            col = topic_id
            sorted_docs = topic_df.sort_values(by=col, ascending=False).head(top_n)
            for rank, (_, row) in enumerate(sorted_docs.iterrows(), start=1):
                writer.writerow(
                    {
                        "topic_id": topic_id + 1,
                        "rank": rank,
                        "doc_index": int(row["doc_index"]),
                        "topic_weight": round(float(row[col]), 6),
                        "comment": row["comment"],
                    }
                )

    return out_path


def save_lda_model(
    model: LdaModel,
    *,
    filename: str | Path = "model/lda.model",
) -> Path:
    """
    Persist a trained Gensim LDA model to disk.

    Gensim writes several sidecar files next to ``filename`` (e.g. the model
    state and id2word mapping). Reload later with ``LdaModel.load(filename)``.
    """
    out_path = _ensure_parent_dir(filename)
    model.save(str(out_path))
    return out_path


def save_topic_coordinates(
    model: LdaModel,
    corpus_tfidf: list[list[tuple[int, int]]],
    dictionary,
    *,
    filename: str | Path = "topic_coordinates.csv",
    mds: str = "mmds",
    sort_topics: bool = False,
) -> Path:
    """
    Compute and save the 2D MDS topic coordinates used by pyLDAvis.

    Columns: topic_id, x, y, freq

    ``sort_topics=False`` keeps Gensim's native topic order so ``topic_id``
    aligns 1-to-1 with the topic numbering in ``topic_tokens.csv`` /
    ``top_comments.csv``. Set ``sort_topics=True`` to renumber by prevalence
    (pyLDAvis default), in which case the ids will NOT match those CSVs.

    Requires pyLDAvis (``pip install pyLDAvis``).
    """
    try:
        import pyLDAvis.gensim_models as gensim_vis
    except ImportError:  # pragma: no cover
        try:
            from pyLDAvis import gensim_models as gensim_vis  # type: ignore[attr-defined]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "save_topic_coordinates requires pyLDAvis. Install it with "
                "`pip install pyLDAvis`."
            ) from exc

    vis_data = gensim_vis.prepare(
        model,
        corpus_tfidf,
        dictionary,
        mds=mds,
        sort_topics=sort_topics,
    )

    coords = vis_data.topic_coordinates.copy()
    coords = coords.rename(columns={"topics": "topic_id", "Freq": "freq"})
    coords = coords[["topic_id", "x", "y", "freq"]].sort_values("topic_id")

    out_path = _ensure_parent_dir(filename)
    coords.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path
