"""Download the two data files this project needs into data/.

1. IMDb movie reviews (Maas et al., 2011). The original Stanford AI
   Lab host (ai.stanford.edu) does not answer from this machine, so
   this fetches the same dataset from its official Hugging Face
   distribution, stanfordnlp/imdb, as parquet files:
   https://huggingface.co/datasets/stanfordnlp/imdb
   25,000 labelled train reviews and 25,000 labelled test reviews.
2. The VADER sentiment lexicon from the NLTK data repository on
   GitHub, saved as a plain text file the code loads directly.
"""
import hashlib
import io
import pathlib
import urllib.request
import zipfile

import requests

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
HF_BASE = "https://huggingface.co/datasets/stanfordnlp/imdb/resolve/main/plain_text"
VADER_URL = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/sentiment/vader_lexicon.zip"


def fetch(url, target):
    r = requests.get(url, timeout=120, headers={"User-Agent": "nlp-sentiment-intelligence rebuild"})
    r.raise_for_status()
    target.write_bytes(r.content)
    print(target.name, "sha256", hashlib.sha256(r.content).hexdigest(), "bytes", len(r.content))


def main():
    DATA.mkdir(exist_ok=True)
    for split in ("train", "test"):
        target = DATA / f"imdb_{split}.parquet"
        if not target.exists():
            fetch(f"{HF_BASE}/{split}-00000-of-00001.parquet", target)
    vader_target = DATA / "vader_lexicon.txt"
    if not vader_target.exists():
        print("downloading VADER lexicon")
        raw = urllib.request.urlopen(VADER_URL).read()
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            name = [n for n in zf.namelist() if n.endswith("vader_lexicon.txt")][0]
            vader_target.write_bytes(zf.read(name))
    print("vader lexicon lines:", sum(1 for _ in open(vader_target)))


if __name__ == "__main__":
    main()
