"""Tests for ``pipeline.py``.

Coverage map (matches the Day-8 brief in the central improvement plan):

* ``VADER`` scoring on a set of canned positive / negative / neutral texts —
  exercises :func:`pipeline.analyze_sentiment`.
* ``TextRank``-style extractive summarization on a multi-sentence doc —
  exercises :func:`pipeline.textrank_summarize` and its top-k cap.
* ``BERT`` batch-shape sanity check — gated on ``transformers`` being
  importable so the file is safe to keep in the repo until the BERT path
  in the pipeline is actually wired up.

Heavy / network-bound deps are gated with ``pytest.importorskip``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")
pytest.importorskip("pandas")
pytest.importorskip("nltk")
pytest.importorskip("sklearn")
pytest.importorskip("networkx")

import nltk  # noqa: E402

# Pre-fetch the small NLTK resources VADER / sent_tokenize need.  Doing it
# once at module load keeps each test fast and works offline once the
# resources are cached.
for resource in ("vader_lexicon", "punkt", "punkt_tab"):
    try:
        nltk.download(resource, quiet=True)
    except Exception:  # pragma: no cover - network-only failure
        pass

import pandas as pd  # noqa: E402

from pipeline import (  # noqa: E402
    analyze_sentiment,
    generate_review_data,
    textrank_summarize,
)


# ---------------------------------------------------------------------------
# VADER scoring on canned texts
# ---------------------------------------------------------------------------


CANNED_TEXTS = [
    ("Absolutely love this product! Quality is amazing.", "positive"),
    ("This is the best thing I have ever bought, total joy.", "positive"),
    ("Terrible experience. Broke after two days. Awful.", "negative"),
    ("Waste of money. Customer support ignored me entirely.", "negative"),
    ("It is okay. Nothing special but it works.", "neutral"),
]


def _vader_frame() -> pd.DataFrame:
    """Build the canned-text DataFrame the pipeline expects."""
    return pd.DataFrame(
        {
            "review_text": [t for t, _ in CANNED_TEXTS],
            "true_sentiment": [s for _, s in CANNED_TEXTS],
        }
    )


def test_vader_assigns_expected_polarities() -> None:
    """Each canned line gets the polarity it was written to elicit."""
    df = analyze_sentiment(_vader_frame())
    assert "vader_compound" in df.columns
    assert "vader_sentiment" in df.columns

    # Positives score >= 0.05 compound, negatives <= -0.05, the neutral
    # one lands inside the (-0.05, 0.05) band.
    for _, row in df.iterrows():
        if row["true_sentiment"] == "positive":
            assert row["vader_compound"] >= 0.05, row.to_dict()
            assert row["vader_sentiment"] == "positive"
        elif row["true_sentiment"] == "negative":
            assert row["vader_compound"] <= -0.05, row.to_dict()
            assert row["vader_sentiment"] == "negative"
        else:
            assert -0.05 < row["vader_compound"] < 0.05, row.to_dict()
            assert row["vader_sentiment"] == "neutral"


def test_vader_compound_within_range_on_synthetic_corpus() -> None:
    """Every score sits in VADER's documented [-1, 1] range."""
    df = analyze_sentiment(generate_review_data(n_samples=200))
    arr = df["vader_compound"].to_numpy()
    assert (arr >= -1.0).all() and (arr <= 1.0).all()
    assert set(df["vader_sentiment"].unique()).issubset({"positive", "negative", "neutral"})


# ---------------------------------------------------------------------------
# TextRank top-k extractive summarization
# ---------------------------------------------------------------------------


SAMPLE_DOC = (
    "I bought this blender and it is powerful. "
    "The glass pitcher feels premium and heavy. "
    "However, the noise level is unbearable on the highest setting. "
    "The rubber gasket leaked slightly after a couple of weeks. "
    "Customer service sent a replacement gasket but it was a hassle. "
    "Overall it is a decent machine for the price if you do not mind the noise."
)


def test_textrank_returns_at_most_k_sentences() -> None:
    """``textrank_summarize`` honours its ``num_sentences`` top-k cap."""
    for k in (1, 2, 3):
        summary = textrank_summarize(SAMPLE_DOC, num_sentences=k)
        from nltk.tokenize import sent_tokenize

        kept = sent_tokenize(summary)
        assert 1 <= len(kept) <= k, f"k={k}: got {len(kept)} sentences"


def test_textrank_summary_is_a_subset_of_input() -> None:
    """Extractive summarization should never invent sentences."""
    from nltk.tokenize import sent_tokenize

    original = [s.strip() for s in sent_tokenize(SAMPLE_DOC)]
    summary = textrank_summarize(SAMPLE_DOC, num_sentences=2)
    for s in sent_tokenize(summary):
        assert s.strip() in original, s


def test_textrank_passthrough_when_doc_shorter_than_k() -> None:
    """When the doc has <= k sentences, the function returns the doc verbatim."""
    short = "Hello world."
    assert textrank_summarize(short, num_sentences=3) == short


def test_textrank_preserves_original_ordering() -> None:
    """Selected sentences keep their original order (top_indices are sorted)."""
    from nltk.tokenize import sent_tokenize

    original = [s.strip() for s in sent_tokenize(SAMPLE_DOC)]
    summary = textrank_summarize(SAMPLE_DOC, num_sentences=3)
    picked = [s.strip() for s in sent_tokenize(summary)]
    indices = [original.index(s) for s in picked]
    assert indices == sorted(indices), indices


# ---------------------------------------------------------------------------
# BERT batch-shape sanity (gated)
# ---------------------------------------------------------------------------


def test_bert_batch_shape() -> None:
    """If ``transformers`` is installed, encode a small batch through a
    DistilBERT model and assert the output shape is ``(batch, seq, hidden)``.

    This guards future ML work that replaces the VADER scorer with a
    transformer classifier — when that lands, the test exercises the
    batched forward path with a real model.  When transformers isn't
    available the test skips cleanly so the rest of the suite still runs.
    """
    transformers = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")

    model_id = "prajjwal1/bert-tiny"  # ~17 MB, batchable in CI
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    model = transformers.AutoModel.from_pretrained(model_id)
    model.eval()

    batch = [
        "Love this product, fantastic experience.",
        "Terrible quality, will not buy again.",
        "It works, nothing more nothing less.",
    ]
    enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)

    hidden = out.last_hidden_state
    assert hidden.dim() == 3
    assert hidden.shape[0] == len(batch)
    assert hidden.shape[1] == enc["input_ids"].shape[1]
    assert hidden.shape[2] == model.config.hidden_size
