"""
Render a labeled MDS topic map from precomputed coordinates + a labeled CSV.

This is the lightweight "Option B" plotting step. It does NOT retrain LDA;
it consumes the artifacts written by main.py:

  - outputs/topic_coordinates.csv  (topic_id, x, y, freq)   ← MDS positions
  - outputs/topic_tokens.csv       (topic_id, tokens[, ...]) ← edit this

Workflow:
  1. Run main.py to produce the two CSVs above (plus the saved model).
  2. Open topic_tokens.csv and add a ``label`` column with a short phrase
     per topic (Chinese or English).
  3. Run this script to draw the map. Each bubble is positioned by MDS,
     sized by topic prevalence, and annotated with an **English-only** label.

Label rules:
  - Prefer the ``label`` column (case-insensitive: label / Label / LABEL).
  - Chinese phrases are translated as a **whole phrase** (not split on spaces).
  - Only the English translation is drawn on the chart.
  - Text is wrapped to roughly match bubble diameter.
  - If ``label`` is empty, falls back to translating the top tokens as a phrase.

Install dependencies first (if needed):
    pip install matplotlib deep-translator

Usage:
    python plot_labeled_mds.py
"""

from __future__ import annotations

import math
import re
import textwrap
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# =============================================================================
# Configuration
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Point at the run folder that holds topic_coordinates.csv + topic_tokens.csv.
# Change this when switching between e.g. 19-topic vs 32-topic drafts.
RUN_DIR = OUTPUT_DIR / "7.1 - ASA draft 32-topic"

COORDINATES_CSV = RUN_DIR / "topic_coordinates.csv"
LABELS_CSV = RUN_DIR / "topic_tokens.csv"
OUTPUT_PNG = RUN_DIR / "lda_labeled_mds.png"

# Column names in the labeled CSV
TOKENS_COLUMN = "tokens"          # space-separated Chinese tokens
TOKENS_EN_COLUMN = "tokens_en"    # optional; comma-separated English tokens
LABEL_COLUMN = "label"            # optional; user-written topic name

N_LABEL_TOKENS = 3                # tokens to use when falling back from empty label
FIGSIZE = (14, 11)
MIN_BUBBLE = 300                  # min marker area (points^2)
MAX_BUBBLE = 6000                 # max marker area (points^2)
BASE_FONTSIZE = 8
MIN_WRAP_CHARS = 8                # never wrap tighter than this
MAX_WRAP_CHARS = 28               # never wrap wider than this
CHARS_PER_POINT = 0.55            # approximate character width relative to fontsize

# CJK-capable fonts (first available wins). 'Arial Unicode MS' ships on macOS.
CJK_FONT_CANDIDATES = [
    "Arial Unicode MS",
    "Heiti TC",
    "Songti SC",
    "Hiragino Sans GB",
    "PingFang SC",
    "SimHei",
]

_TRANSLATION_CACHE: dict[str, str] = {}
_TRANSLATOR = None
_CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")


# =============================================================================
# Translation helpers
# =============================================================================
def _get_translator():
    global _TRANSLATOR
    if _TRANSLATOR is None:
        from deep_translator import GoogleTranslator

        _TRANSLATOR = GoogleTranslator(source="zh-CN", target="en")
    return _TRANSLATOR


def _has_chinese(text: str) -> bool:
    """True if the string contains any CJK ideograph."""
    return bool(_CHINESE_RE.search(text))


def translate_phrase(text: str) -> str:
    """
    Translate a whole phrase to English (no whitespace splitting).

    Cached. On failure, returns the original text and prints a warning.
    """
    phrase = " ".join(text.split())
    if not phrase:
        return ""
    if phrase in _TRANSLATION_CACHE:
        return _TRANSLATION_CACHE[phrase]
    if not _has_chinese(phrase):
        _TRANSLATION_CACHE[phrase] = phrase
        return phrase

    try:
        translated = _get_translator().translate(phrase)
        english = translated.strip() if translated else phrase
    except Exception as exc:
        print(f"  Warning: translation failed for {phrase!r}: {exc}")
        english = phrase

    _TRANSLATION_CACHE[phrase] = english
    return english


# =============================================================================
# Label construction
# =============================================================================
def _split_tokens(value: object) -> list[str]:
    if not isinstance(value, str):
        return []
    return [t for t in value.replace(",", " ").split() if t]


def _split_english(value: object) -> list[str]:
    if not isinstance(value, str):
        return []
    return [t.strip() for t in value.split(",") if t.strip()]


def _get_field(row: pd.Series, name: str) -> object:
    """Case-insensitive column lookup on a row (e.g. 'label' matches 'Label')."""
    value = row.get(name)
    if value is not None and not (isinstance(value, float) and math.isnan(value)):
        return value
    target = name.lower()
    for key in row.index:
        if isinstance(key, str) and key.lower() == target:
            val = row[key]
            if isinstance(val, float) and math.isnan(val):
                return None
            return val
    return None


def build_topic_label(row: pd.Series) -> str:
    """
    Build the English-only annotation text for one topic row.

    Prefer the user ``label`` (translated as a whole phrase if Chinese).
    Fall back to top tokens / tokens_en when label is empty.
    """
    user_label = _get_field(row, LABEL_COLUMN)
    if isinstance(user_label, str) and user_label.strip():
        return translate_phrase(user_label.strip())

    english = _split_english(_get_field(row, TOKENS_EN_COLUMN))[:N_LABEL_TOKENS]
    if english:
        return ", ".join(english)

    tokens = _split_tokens(_get_field(row, TOKENS_COLUMN))[:N_LABEL_TOKENS]
    if not tokens:
        return ""
    return translate_phrase(" ".join(tokens))


def wrap_label_for_bubble(text: str, bubble_area: float) -> str:
    """
    Wrap English label text to roughly match bubble diameter.

    ``bubble_area`` is matplotlib scatter ``s`` (marker area in points^2).
    Text may extend a bit past the bubble; wrapping just keeps it comparable.
    """
    if not text:
        return ""
    diameter = 2.0 * math.sqrt(max(bubble_area, 1.0) / math.pi)
    width = int(diameter * CHARS_PER_POINT)
    width = max(MIN_WRAP_CHARS, min(MAX_WRAP_CHARS, width))
    return textwrap.fill(text, width=width)


# =============================================================================
# Plotting
# =============================================================================
def _configure_cjk_font() -> None:
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = [name for name in CJK_FONT_CANDIDATES if name in available]
    if chosen:
        plt.rcParams["font.sans-serif"] = chosen + plt.rcParams.get(
            "font.sans-serif", []
        )
        plt.rcParams["font.family"] = "sans-serif"
    else:
        print(
            "Warning: no CJK font found; Chinese characters may not render. "
            f"Tried: {CJK_FONT_CANDIDATES}"
        )
    plt.rcParams["axes.unicode_minus"] = False


def _scale_bubbles(freq: pd.Series) -> pd.Series:
    fmin, fmax = float(freq.min()), float(freq.max())
    if fmax <= fmin:
        return pd.Series([(MIN_BUBBLE + MAX_BUBBLE) / 2] * len(freq), index=freq.index)
    norm = (freq - fmin) / (fmax - fmin)
    return MIN_BUBBLE + norm * (MAX_BUBBLE - MIN_BUBBLE)


def load_data(
    coordinates_csv: Path | str = COORDINATES_CSV,
    labels_csv: Path | str = LABELS_CSV,
) -> pd.DataFrame:
    """Load and merge coordinates with the labeled tokens CSV on topic_id."""
    coords = pd.read_csv(coordinates_csv)
    labels = pd.read_csv(labels_csv)

    required = {"topic_id", "x", "y", "freq"}
    missing = required - set(coords.columns)
    if missing:
        raise ValueError(
            f"{coordinates_csv} is missing columns: {sorted(missing)}. "
            "Re-run main.py to regenerate topic_coordinates.csv."
        )
    if "topic_id" not in labels.columns:
        raise ValueError(f"{labels_csv} must have a 'topic_id' column.")

    merged = coords.merge(labels, on="topic_id", how="left")
    return merged


def plot_labeled_mds(
    data: pd.DataFrame,
    *,
    output_path: Path | str = OUTPUT_PNG,
    title: str = "LDA Topic Map (MDS)",
) -> Path:
    """Draw the labeled MDS scatter and save it as an image."""
    _configure_cjk_font()

    sizes = _scale_bubbles(data["freq"])

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.scatter(
        data["x"],
        data["y"],
        s=sizes,
        c=range(len(data)),
        cmap="tab20",
        alpha=0.55,
        edgecolors="black",
        linewidths=0.8,
        zorder=2,
    )

    print("Building English labels (whole-phrase translation)...")
    for idx, (_, row) in enumerate(data.iterrows()):
        english = build_topic_label(row)
        wrapped = wrap_label_for_bubble(english, float(sizes.iloc[idx]))
        annotation = f"#{int(row['topic_id'])}"
        if wrapped:
            annotation += f"\n{wrapped}"
        ax.annotate(
            annotation,
            (row["x"], row["y"]),
            ha="center",
            va="center",
            fontsize=BASE_FONTSIZE,
            zorder=3,
        )

    ax.axhline(0, color="grey", linewidth=0.6, linestyle="--", zorder=1)
    ax.axvline(0, color="grey", linewidth=0.6, linestyle="--", zorder=1)
    ax.set_xlabel("dimension 2")
    ax.set_ylabel("dimension 1")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main() -> Path:
    print(f"Reading coordinates: {COORDINATES_CSV}")
    print(f"Reading labels:      {LABELS_CSV}")
    data = load_data()
    out_path = plot_labeled_mds(data)
    print(f"\nSaved labeled MDS plot → {out_path.resolve()}")
    return out_path


if __name__ == "__main__":
    main()
