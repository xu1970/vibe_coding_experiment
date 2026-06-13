"""
Compute gensim C_v topic coherence for topic-count sweep outputs.

Preprocesses once (main.py settings), then scores each K using top tokens
from outputs/topic_sweep/topic_tokens_k{K}.csv (3 runs per K).
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from gensim.models import CoherenceModel

from main import PROJECT_ROOT, build_preprocess_config
from preprocess_corpus import preprocess_corpus

TOPIC_MIN = 22
TOPIC_MAX = 32  # inclusive
SWEEP_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "topic_sweep"
RESULTS_CSV = SWEEP_OUTPUT_DIR / "coherence_cv_k25_35.csv"


def _topics_from_sweep_csv(path: Path, run: int) -> list[list[str]]:
    df = pd.read_csv(path)
    dfr = df[df["run"] == run].sort_values("topic_id")
    return [str(row.tokens).split() for _, row in dfr.iterrows()]


def cv_coherence(
    topics: list[list[str]],
    texts: list[list[str]],
    dictionary,
) -> float:
    cm = CoherenceModel(
        topics=topics,
        texts=texts,
        dictionary=dictionary,
        coherence="c_v",
    )
    return float(cm.get_coherence())


def evaluate_cv_range(
    *,
    topic_min: int = TOPIC_MIN,
    topic_max: int = TOPIC_MAX,
) -> pd.DataFrame:
    print("Preprocessing corpus (once)...")
    t0 = time.perf_counter()
    preprocessed = preprocess_corpus(build_preprocess_config())
    texts = preprocessed["lda_docs"]
    dictionary = preprocessed["dictionary"]
    print(
        f"  {len(texts)} docs, vocab {len(dictionary)} "
        f"({time.perf_counter() - t0:.1f}s)"
    )

    rows: list[dict] = []
    for k in range(topic_min, topic_max + 1):
        path = SWEEP_OUTPUT_DIR / f"topic_tokens_k{k}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing sweep output: {path}")

        run_scores: list[float] = []
        for run in sorted(pd.read_csv(path)["run"].unique()):
            topics = _topics_from_sweep_csv(path, int(run))
            if len(topics) != k:
                raise ValueError(
                    f"{path} run {run}: expected {k} topics, got {len(topics)}"
                )
            score = cv_coherence(topics, texts, dictionary)
            run_scores.append(score)
            rows.append({"k": k, "run": int(run), "c_v": score})
            print(f"  K={k} run={int(run)}  C_v={score:.4f}")

        mean = sum(run_scores) / len(run_scores)
        print(f"  K={k} mean C_v={mean:.4f}  (runs={run_scores})")

    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("k", as_index=False)["c_v"]
        .agg(mean="mean", std="std", min="min", max="max")
        .sort_values("k")
    )
    return detail, summary


def main() -> None:
    SWEEP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detail, summary = evaluate_cv_range()

    detail.to_csv(SWEEP_OUTPUT_DIR / "coherence_cv_k25_35_detail.csv", index=False)
    summary.to_csv(RESULTS_CSV, index=False)

    print("\n" + "=" * 60)
    print("C_v coherence summary (K=25..35)")
    print("=" * 60)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved: {RESULTS_CSV}")
    best = summary.loc[summary["mean"].idxmax()]
    print(
        f"Best mean C_v: K={int(best['k'])} "
        f"(mean={best['mean']:.4f}, std={best['std']:.4f})"
    )


if __name__ == "__main__":
    main()
