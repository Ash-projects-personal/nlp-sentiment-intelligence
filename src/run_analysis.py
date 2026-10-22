"""Sentiment classification on real IMDb reviews, plus a TextRank
summarizer that is actually a graph method.

Two scorers on the exact same test reviews:
- VADER, a lexicon and rule based scorer, no training.
- A TF-IDF plus logistic regression classifier trained on the
  training reviews only.

Subsampling is by fixed seed and recorded here and in the data card.
The old version of this project claimed a sentiment to sales
correlation. That section is deleted: the sales series there was
generated from the sentiment series, so the correlation was built
in. I have no real paired sales data, so there is no correlation
result in this repo.

The summarizer builds a sentence similarity graph with TF-IDF
cosine similarity and ranks sentences with PageRank. That is
TextRank in the plain sense. There is no T5 or BERT anywhere here.
"""
import json
import pathlib
import re
import time

import networkx as nx
import numpy as np
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
SEED = 42
N_TRAIN_PER_CLASS = 2000
N_TEST_PER_CLASS = 500


def load_split(split, per_class):
    df = pd.read_parquet(ROOT / "data" / f"imdb_{split}.parquet")
    parts = []
    for label in (0, 1):
        parts.append(df[df["label"] == label].sample(n=per_class, random_state=SEED))
    out = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
    out["text"] = out["text"].str.replace(r"<br\s*/?>", " ", regex=True)
    return out[["text", "label"]]


def textrank_summarize(text, n_sentences=2):
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20]
    if len(sentences) <= n_sentences:
        return " ".join(sentences), len(sentences)
    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(sentences)
    sim = (X @ X.T).toarray()
    np.fill_diagonal(sim, 0)
    graph = nx.from_numpy_array(sim)
    scores = nx.pagerank(graph, max_iter=200)
    ranked = sorted(scores, key=scores.get, reverse=True)[:n_sentences]
    return " ".join(sentences[i] for i in sorted(ranked)), len(sentences)


def main():
    OUT.mkdir(exist_ok=True)
    started = time.time()
    train = load_split("train", N_TRAIN_PER_CLASS)
    test = load_split("test", N_TEST_PER_CLASS)
    assert not set(train["text"]).intersection(test["text"])

    # Current NLTK versions refuse to load a lexicon from a project local
    # path (path security check), so I read the lexicon text myself and
    # hand it to VADER exactly the way its own loader would: the file
    # content as a string, then make_lex_dict() parses it.
    from nltk.sentiment.vader import VaderConstants
    vader = SentimentIntensityAnalyzer.__new__(SentimentIntensityAnalyzer)
    vader.lexicon_file = (ROOT / "data" / "vader_lexicon.txt").read_text(encoding="utf-8")
    vader.lexicon = vader.make_lex_dict()
    vader.constants = VaderConstants()
    vader_pred = [1 if vader.polarity_scores(t)["compound"] >= 0 else 0 for t in test["text"]]

    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english")
    X_train = vec.fit_transform(train["text"])
    X_test = vec.transform(test["text"])
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train, train["label"])
    clf_pred = clf.predict(X_test)

    results = {
        "vader_lexicon": {"accuracy": float(accuracy_score(test["label"], vader_pred))},
        "tfidf_logistic_regression": {"accuracy": float(accuracy_score(test["label"], clf_pred))},
    }
    report = classification_report(test["label"], clf_pred, target_names=["neg", "pos"], output_dict=True)

    pd.DataFrame({
        "label": test["label"], "vader_pred": vader_pred, "clf_pred": clf_pred,
        "text_start": [t[:120] for t in test["text"]],
    }).to_csv(OUT / "predictions.csv", index=False)

    demo_text = test[test["label"] == 1]["text"].iloc[0]
    summary, n_sent = textrank_summarize(demo_text, 2)
    demo = {
        "source_words": len(demo_text.split()),
        "source_sentences": n_sent,
        "summary_words": len(summary.split()),
        "summary": summary,
        "note": "One example from the test set. Compression is counted, not a time saving claim. No analyst time was measured.",
    }

    metrics = {
        "dataset": {
            "name": "IMDb movie reviews (Maas et al., 2011), aclImdb",
            "source": "https://huggingface.co/datasets/stanfordnlp/imdb (official Hugging Face distribution of the aclImdb dataset, Maas et al. 2011)",
            "license_note": "Published by the authors for research use. Reviews are user written movie reviews with binary sentiment labels from star ratings.",
            "train_rows_used": int(len(train)),
            "test_rows_used": int(len(test)),
            "sampling": f"seed {SEED}, {N_TRAIN_PER_CLASS} per class from the 25,000 train reviews, {N_TEST_PER_CLASS} per class from the 25,000 test reviews, official train and test splits kept separate",
        },
        "results": results,
        "tfidf_classification_report": report,
        "summarizer_demo": demo,
        "removed_claims": "No sentiment to sales correlation is reported. No BERT, no T5. The summarizer is TF-IDF graph TextRank and is described as such.",
        "runtime_seconds": round(time.time() - started, 1),
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (OUT / "run_log.txt").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print("demo words:", demo["source_words"], "to", demo["summary_words"])


if __name__ == "__main__":
    main()
