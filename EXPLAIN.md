# How this works, in plain words

## The comparison

VADER scores a review by looking up words in a fixed lexicon and
applying rules for things like negation and punctuation. It never
sees the training data. The TF-IDF classifier learns from 4,000
labelled reviews which words and pairs of words go with positive
and negative reviews of movies specifically. That is why it wins:
it adapts to the domain. A word like "predictable" is negative for
a movie and harmless elsewhere, and only the trained model can
know that.

Both are tested on reviews neither was built from, from the
official IMDb test split. Same 1,000 reviews, same labels, so the
0.730 against 0.849 comparison is direct.

## The summarizer

TextRank treats each sentence as a node. Two sentences get a
strong edge when their TF-IDF vectors are similar, meaning they
talk about the same things. PageRank then finds the sentences
that are most central to the document's own content, and the top
two are returned in reading order. It is extractive: it only ever
returns sentences from the original text, so it cannot invent
content. That also caps its quality, since a badly written key
sentence stays badly written.

## The deleted correlation, and why

The old pipeline generated a sales series by adding a shifted copy
of the sentiment series to it, then measured the correlation and
the lag and reported both as findings. Any dataset built that way
will show exactly the correlation it was built to show. There is
no fix for that inside the same design. The honest version needs
two independent real series, reviews and sales for the same
product over the same weeks. Without that data, the right move is
to say the question is untested, which is what the README does.

## One code note

Current NLTK releases block loading a lexicon file from a project
folder as a security measure. The code reads the lexicon text
directly and hands the string to VADER, which is what its own
loader returns anyway. The lexicon itself is the unmodified
NLTK VADER lexicon, 7,517 entries.
