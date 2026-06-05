"""
LDA training and analysis utilities for Gensim.

Functions accept preprocessed corpus objects (dictionary + BoW corpus)
and model settings; they do not perform text preprocessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from gensim import corpora, models
from gensim.models import LdaModel


@dataclass
class LDASettings:
    """Hyper-parameters for Gensim LDA training."""

    num_topics: int = 20
    passes: int = 25
    random_state: int = 42
    alpha: str = "auto"
    eta: str = "auto"
    minimum_probability: float = 0.0
    topn_topics_print: int = 10
    topn_topic_tokens: int = 20
    topn_docs_per_topic: int = 7


def build_tfidf_corpus(
    corpus: list[list[tuple[int, int]]],
) -> tuple[models.TfidfModel, list[list[tuple[int, int]]]]:
    """Weight document-term matrix with TF-IDF."""
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    return tfidf, corpus_tfidf


def train_lda_model(
    corpus_tfidf: list[list[tuple[int, int]]],
    dictionary: corpora.Dictionary,
    settings: LDASettings | dict[str, Any],
) -> LdaModel:
    """Train LDA on a TF-IDF-weighted corpus."""
    if isinstance(settings, dict):
        settings = LDASettings(**{k: v for k, v in settings.items() if k in LDASettings.__dataclass_fields__})

    return LdaModel(
        corpus=corpus_tfidf,
        num_topics=settings.num_topics,
        id2word=dictionary,
        passes=settings.passes,
        random_state=settings.random_state,
        alpha=settings.alpha,
        eta=settings.eta,
    )


def print_lda_topics(
    lda_model: LdaModel,
    *,
    num_words: int = 10,
) -> None:
    """Print human-readable topic summaries."""
    for idx, topic in lda_model.print_topics(num_words=num_words):
        print(f"Topic: {idx + 1}")
        print(f"Words: {topic}\n")


def get_document_topic_matrix(
    lda_model: LdaModel,
    corpus: list[list[tuple[int, int]]],
    *,
    minimum_probability: float = 0.0,
) -> list[list[float]]:
    """Return per-document topic probability vectors."""
    doc_topic_dist: list[list[float]] = []
    for doc in corpus:
        doc_topics = lda_model.get_document_topics(doc, minimum_probability=minimum_probability)
        doc_topic_dist.append([prob for _, prob in doc_topics])
    return doc_topic_dist


def build_document_topic_dataframe(
    lda_model: LdaModel,
    corpus: list[list[tuple[int, int]]],
    documents: list[str],
    *,
    minimum_probability: float = 0.0,
) -> pd.DataFrame:
    """Combine topic distributions with original comment text."""
    doc_topic_dist = get_document_topic_matrix(
        lda_model,
        corpus,
        minimum_probability=minimum_probability,
    )
    columns = [f"Topic {i + 1}" for i in range(lda_model.num_topics)]
    df = pd.DataFrame(doc_topic_dist, columns=columns)
    df["Document"] = documents[: len(df)]
    return df


def get_top_documents_per_topic(
    doc_topic_df: pd.DataFrame,
    *,
    top_n: int = 7,
) -> dict[str, pd.DataFrame]:
    """Rank documents by topic weight for each topic."""
    top_docs: dict[str, pd.DataFrame] = {}
    num_topics = sum(1 for c in doc_topic_df.columns if c.startswith("Topic "))
    for topic_num in range(num_topics):
        col = f"Topic {topic_num + 1}"
        sorted_docs = doc_topic_df.sort_values(by=col, ascending=False)
        top_docs[col] = sorted_docs.head(top_n)
    return top_docs


def get_topic_tokens(
    lda_model: LdaModel,
    *,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    """Extract top tokens and weights for each topic."""
    topic_tokens: list[dict[str, Any]] = []
    for topic_id in range(lda_model.num_topics):
        topic_words = lda_model.show_topic(topic_id, topn=top_k)
        topic_tokens.append(
            {
                "topic_id": topic_id + 1,
                "tokens": [word for word, _ in topic_words],
                "weights": [f"{weight:.4f}" for _, weight in topic_words],
                "words_with_weights": [
                    (word, f"{weight:.4f}") for word, weight in topic_words
                ],
            }
        )
    return topic_tokens


def print_topic_tokens(
    lda_model: LdaModel,
    *,
    top_k: int = 20,
) -> None:
    """Print top tokens per topic with Chinese token labels."""
    print("=" * 80)
    print("LDA Topic Tokens (Top Words per Topic)")
    print("=" * 80)
    for topic_data in get_topic_tokens(lda_model, top_k=top_k):
        print(f"\nTopic {topic_data['topic_id']}:")
        print("-" * 80)
        for word, weight in topic_data["words_with_weights"]:
            print(f"  {word:20s} {weight}")


def run_lda_pipeline(
    dictionary: corpora.Dictionary,
    corpus: list[list[tuple[int, int]]],
    documents: list[str],
    settings: LDASettings | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    End-to-end LDA workflow from preprocessed Gensim objects.

    Parameters
    ----------
    dictionary : corpora.Dictionary
        Vocabulary built from tokenized documents.
    corpus : list of BoW vectors
        Document-term matrix from dictionary.doc2bow.
    documents : list of str
        Raw comment text aligned with corpus rows.
    settings : LDASettings or dict, optional
        Model hyper-parameters.

    Returns
    -------
    dict with keys: tfidf, corpus_tfidf, lda_model, doc_topic_df,
                    top_docs_per_topic, topic_tokens
    """
    cfg = settings if settings is not None else LDASettings()
    if isinstance(cfg, dict):
        cfg = LDASettings(**{k: v for k, v in cfg.items() if k in LDASettings.__dataclass_fields__})

    tfidf, corpus_tfidf = build_tfidf_corpus(corpus)
    lda_model = train_lda_model(corpus_tfidf, dictionary, cfg)

    doc_topic_df = build_document_topic_dataframe(
        lda_model,
        corpus,
        documents,
        minimum_probability=cfg.minimum_probability,
    )
    top_docs = get_top_documents_per_topic(
        doc_topic_df,
        top_n=cfg.topn_docs_per_topic,
    )
    topic_tokens = get_topic_tokens(lda_model, top_k=cfg.topn_topic_tokens)

    return {
        "tfidf": tfidf,
        "corpus_tfidf": corpus_tfidf,
        "lda_model": lda_model,
        "doc_topic_df": doc_topic_df,
        "top_docs_per_topic": top_docs,
        "topic_tokens": topic_tokens,
        "settings": cfg,
    }
