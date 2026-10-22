# NLP sentiment intelligence

I rebuilt this project from scratch. The old version's headline was
a correlation between review sentiment and sales volume that the
code manufactured: it built the sales series out of the shifted
sentiment series and then reported the correlation it had planted.
It also claimed a fine tuned BERT scorer and a T5 summarizer that
were never in the code. All of that is removed. What is here now is
measured on real labelled reviews, and the sales correlation claim
is gone entirely because I have no real paired sales data to test
it on.

## Dataset

IMDb movie reviews, Maas et al. 2011. 25,000 labelled train reviews
and 25,000 labelled test reviews, labels from star ratings. I use
the official Hugging Face distribution (stanfordnlp/imdb) because
the original Stanford host does not answer from my machine, and
`src/download_data.py` records the file hashes in
`data/data_card.json`.

Sampling, fixed by seed 42 before any modelling: 2,000 reviews per
class from the train split (4,000 total) to train, 500 per class
from the test split (1,000 total) to evaluate. The official train
and test splits are never mixed.

## Results

Two scorers on the same 1,000 held out test reviews:

| Scorer | Accuracy |
|---|---|
| VADER (lexicon, no training) | 0.730 |
| TF-IDF + logistic regression | 0.849 |

VADER is a word list with rules. It needs no training data and it
shows: it is decent and clearly worse than a simple trained
classifier. The TF-IDF model uses word and bigram features (20,000
max) and plain logistic regression. No BERT, no fine tuning, and I
am not claiming otherwise. Per review predictions are in
`outputs/predictions.csv`, the full classification report is in
`outputs/metrics.json`.

## Summarizer

The summarizer is real TextRank this time: sentences become nodes
in a graph, edges are TF-IDF cosine similarity between sentences,
and PageRank picks the top sentences in their original order. The
old version called a word frequency counter TextRank and mentioned
a T5 that did not exist. There is no T5 here.

One measured demo from a test review: 557 words and its sentences
compressed to a 2 sentence, 25 word summary. That is a word count,
not a time saving claim. I never measured analyst time, so I do
not quote one.

## What is not in this repo

- No sentiment to sales correlation. Testing whether sentiment
  leads sales needs real sales data paired with real reviews over
  time. I do not have that data, so the question stays open and
  unclaimed here.
- No BERT or transformer model. TF-IDF plus logistic regression
  is the trained model, named as such.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/download_data.py
python src/run_analysis.py
python -m pytest tests/
```

## Files

- `src/download_data.py` fetches the IMDb parquet files and the
  VADER lexicon.
- `src/run_analysis.py` samples, scores, trains, summarises, and
  writes `outputs/metrics.json` and `outputs/predictions.csv`.
- `EXPLAIN.md` explains the method and the deleted claim.
