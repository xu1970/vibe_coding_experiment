"""
Generate an interactive pyLDAvis HTML visualization for the LDA model in main.py.

Also exports:
  - lda_topic_tokens_en.csv  — top tokens per topic (Chinese + English)
  - lda_top_comments.csv     — top comments per topic

pyLDAvis projects topics into 2D using multidimensional scaling (MDS) by default.
Term labels in the HTML show both Chinese and English (e.g. ``生育 (fertility)``).

Install dependencies first (if needed):
    pip install pyLDAvis deep-translator

Usage:
    python visualize_lda.py
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pyLDAvis
from deep_translator import GoogleTranslator
from gensim.models import LdaModel

from lda_utils import run_lda_pipeline
from main import (
    LIKES_ALPHA,
    LIKES_USE_LOG,
    NUM_TOP_COMMENTS,
    NUM_TOPIC_TOKENS,
    NUM_TOPICS,
    OUTPUT_DIR,
    REWEIGHT_BY_LIKES,
    build_lda_settings,
    build_preprocess_config,
)
from output_utils import save_top_comments
from preprocess_corpus import preprocess_corpus

# pyLDAvis.gensim_models was renamed in newer releases; support both import paths.
try:
    import pyLDAvis.gensim_models as gensim_vis
except ImportError:  # pragma: no cover
    from pyLDAvis import gensim_models as gensim_vis  # type: ignore[attr-defined]

VIS_OUTPUT_DIR = OUTPUT_DIR
VIS_HTML_FILENAME = "lda_pyldavis.html"
TOPIC_TOKENS_EN_FILENAME = "lda_topic_tokens_en.csv"
TOP_COMMENTS_FILENAME = "lda_top_comments.csv"

_TRANSLATOR = GoogleTranslator(source="zh-CN", target="en")
_TRANSLATION_CACHE: dict[str, str] = {}


def build_lda_artifacts() -> dict:
    """
    Reproduce main.py Steps 1–2 and return objects needed for pyLDAvis.

    From main.py:
      - dictionary  ← preprocessed["dictionary"]
      - corpus      ← preprocessed["corpus"]  (BoW; used for doc-topic display)
      - lda_model   ← lda_result["lda_model"]
      - corpus_tfidf← lda_result["corpus_tfidf"] (matrix LDA was trained on)
    """
    print("Preprocessing corpus...")
    t0 = time.perf_counter()
    preprocessed = preprocess_corpus(build_preprocess_config())
    print(
        f"  {len(preprocessed['documents'])} documents, "
        f"vocab {len(preprocessed['dictionary'])} "
        f"({time.perf_counter() - t0:.1f}s)"
    )

    dictionary = preprocessed["dictionary"]
    corpus = preprocessed["corpus"]
    documents = preprocessed["documents"]
    document_likes = preprocessed.get("document_likes")

    print("Training LDA...")
    t0 = time.perf_counter()
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
    print(
        f"  {lda_model.num_topics} topics "
        f"({time.perf_counter() - t0:.1f}s)"
    )

    return {
        "dictionary": dictionary,
        "corpus": corpus,
        "documents": documents,
        "corpus_tfidf": lda_result["corpus_tfidf"],
        "lda_model": lda_model,
    }


def translate_token(token: str) -> str:
    """Translate one Chinese token to English, with in-memory caching."""
    if token in _TRANSLATION_CACHE:
        return _TRANSLATION_CACHE[token]

    try:
        translated = _TRANSLATOR.translate(token)
        english = translated.strip() if translated else token
    except Exception:
        english = token

    _TRANSLATION_CACHE[token] = english
    return english


def translate_tokens(tokens: list[str]) -> list[str]:
    """Translate a list of tokens, reusing cached translations."""
    unique_tokens = list(dict.fromkeys(tokens))
    for token in unique_tokens:
        if token not in _TRANSLATION_CACHE:
            translate_token(token)
    return [_TRANSLATION_CACHE[token] for token in tokens]


def collect_topic_tokens(
    lda_model: LdaModel,
    num_topics: int,
    *,
    top_n: int,
) -> list[dict[str, str | int]]:
    """Return per-topic rows with Chinese and English top tokens."""
    rows: list[dict[str, str | int]] = []
    all_tokens: list[str] = []

    topic_token_lists: list[list[str]] = []
    for topic_id in range(num_topics):
        topic_words = lda_model.show_topic(topic_id, topn=top_n)
        tokens = [word for word, _ in topic_words]
        topic_token_lists.append(tokens)
        all_tokens.extend(tokens)

    print(f"Translating {len(dict.fromkeys(all_tokens))} unique topic tokens...")
    translate_tokens(all_tokens)

    for topic_id, tokens in enumerate(topic_token_lists):
        tokens_en = [_TRANSLATION_CACHE[token] for token in tokens]
        rows.append(
            {
                "topic_id": topic_id + 1,
                "tokens": " ".join(tokens),
                "tokens_en": ", ".join(tokens_en),
            }
        )

    return rows


def save_topic_tokens_translated(
    topic_token_rows: list[dict[str, str | int]],
    *,
    output_path: Path | str = VIS_OUTPUT_DIR / TOPIC_TOKENS_EN_FILENAME,
) -> Path:
    """Save precomputed topic token rows with English translations."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(topic_token_rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def bilingual_term_label(token: str) -> str:
    """Format a token for display as ``中文 (english)``."""
    english = _TRANSLATION_CACHE.get(token, token)
    if english == token:
        return token
    return f"{token} ({english})"


def apply_bilingual_term_labels(vis_data):
    """Patch pyLDAvis prepared data so term bars show Chinese and English."""
    terms = set(vis_data.topic_info["Term"])
    terms.update(vis_data.token_table["Term"])
    translate_tokens(sorted(terms))

    topic_info = vis_data.topic_info.copy()
    token_table = vis_data.token_table.copy()
    topic_info["Term"] = topic_info["Term"].map(bilingual_term_label)
    token_table["Term"] = token_table["Term"].map(bilingual_term_label)

    return vis_data._replace(topic_info=topic_info, token_table=token_table)


def save_pyldavis_html(
    lda_model,
    corpus_tfidf,
    dictionary,
    *,
    output_path: Path | str = VIS_OUTPUT_DIR / VIS_HTML_FILENAME,
    mds: str = "mmds",
    sort_topics: bool = False,
) -> Path:
    """
    Build pyLDAvis data and save an interactive HTML file.

    Parameters
    ----------
    mds : str
        Topic distance projection. ``'mmds'`` (default) uses multidimensional
        scaling; also ``'pcoa'`` or ``'tsne'``.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Preparing pyLDAvis (MDS={mds!r})...")
    vis_data = gensim_vis.prepare(
        lda_model,
        corpus_tfidf,
        dictionary,
        mds=mds,
        sort_topics=sort_topics,
    )
    vis_data = apply_bilingual_term_labels(vis_data)

    print(f"Saving visualization → {output_path}")
    pyLDAvis.save_html(vis_data, str(output_path))
    return output_path


def main() -> dict[str, Path]:
    artifacts = build_lda_artifacts()
    lda_model = artifacts["lda_model"]

    topic_token_rows = collect_topic_tokens(
        lda_model,
        NUM_TOPICS,
        top_n=NUM_TOPIC_TOKENS,
    )
    tokens_path = save_topic_tokens_translated(
        topic_token_rows,
        output_path=VIS_OUTPUT_DIR / TOPIC_TOKENS_EN_FILENAME,
    )
    comments_path = save_top_comments(
        lda_model,
        artifacts["corpus"],
        artifacts["documents"],
        NUM_TOPICS,
        top_n=NUM_TOP_COMMENTS,
        filename=VIS_OUTPUT_DIR / TOP_COMMENTS_FILENAME,
    )
    html_path = save_pyldavis_html(
        lda_model,
        artifacts["corpus_tfidf"],
        artifacts["dictionary"],
        output_path=VIS_OUTPUT_DIR / VIS_HTML_FILENAME,
    )

    print(f"\nSaved topic tokens (EN): {tokens_path.resolve()}")
    print(f"Saved top comments:      {comments_path.resolve()}")
    print(f"Saved pyLDAvis HTML:     {html_path.resolve()}")

    return {
        "html": html_path,
        "topic_tokens_en": tokens_path,
        "top_comments": comments_path,
    }


if __name__ == "__main__":
    paths = main()
    print(f"\nDone. Open in a browser:\n  {paths['html'].resolve()}")
