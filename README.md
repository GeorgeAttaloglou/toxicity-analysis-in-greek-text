# Toxicity Analysis in Greek text

## About this repo
This is the repository for all the code used for my thesis on toxicity analysis in greek text using NLP.

## Getting Started
### Prerequisites
- Python 3
- Pip, Anaconda or your package manager of choice
- [Pandas](https://pandas.pydata.org/docs/getting_started/install.html)
- [SkLearn](https://scikit-learn.org/stable/install.html)
- [PyTorch](https://pytorch.org/)
- [FastText](https://fasttext.cc/docs/en/support.html)


## Methods used
Two models have been developed so far:
- The baseline model using TF-IDF + Logistic Regression as a performance baseline.
- The intermediate using [greek fasttext](https://fasttext.cc/docs/en/crawl-vectors.html) and Bidirectional Long Short-Term memory neural networks
