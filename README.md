# Toxicity Analysis in Greek text

## About this repo
Toxicity analysis in text is a growing concern in the growing age of anonymity and ai. We've set out to look deeper into the Greek language niche to provide insight for analyzing Greek text and determining its toxicity. Developed by George Attaloglou under the supervision of Petros Karvelis.

## Getting Started
### Prerequisites
To correctly execute this code you will need to download the following:
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

## Data used for training

The [Offensive Greek Tweet Dataset](https://huggingface.co/datasets/strombergnlp/offenseval_2020/tree/main) Developed by Marcos Zampieri, Zeses Pitenis and Tharindu Ranasinghe for [Offense Eval 2020](https://sites.google.com/site/offensevalsharedtask/offenseval-2020) is the main dataset for evaluation and training.

For the intermediate model we use fasttext and its greek word vector dataset.

## License
Distributed under the MIT license. See `LICENSE` for more information.


