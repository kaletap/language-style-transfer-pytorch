# Language Style Transfer
This repo contains the code and data of the following paper:

<i> "Style Transfer from Non-Parallel Text by Cross-Alignment". Tianxiao Shen, Tao Lei, Regina Barzilay, and Tommi Jaakkola. NIPS 2017. [arXiv](https://arxiv.org/abs/1705.09655)</i>

The method learns to perform style transfer between two non-parallel corpora. For example, given positive and negative reviews as two corpora, the model can learn to reverse the sentiment of a sentence.
<p align="center"><img width=800 src="img/example_sentiment.png"></p>

<br>

## Spotlight Video
[![overview](https://img.youtube.com/vi/OyjXG44j-gs/0.jpg)](https://www.youtube.com/watch?v=OyjXG44j-gs)

<br>

## Data Format
Please name the corpora of two styles by "x.0" and "x.1" respectively, and use "x" to refer to them in options. Each file should consist of one sentence per line with tokens separated by a space.

The <code>data/yelp/</code> directory contains an example Yelp review dataset.

<br>

## Quick Start
- To train a model, first create a <code>tmp/</code> folder (where the model and results will be saved), then go to the <code>code/</code> folder and run the following command:
```bash
python style_transfer.py --train ../data/yelp/sentiment.train --dev ../data/yelp/sentiment.dev --output ../tmp/sentiment.dev --vocab ../tmp/yelp.vocab --model ../tmp/model
```

- To test the model, run the following command:
```bash
python style_transfer.py --test ../data/yelp/sentiment.test --output ../tmp/sentiment.test --vocab ../tmp/yelp.vocab --model ../tmp/model --load_model true --beam 8
```

- To download a trained model, run <code>bash download_model.sh</code>, and then run the testing command with <code>--vocab</code> and <code>--model</code> options specifying <code>../model/yelp.vocab</code> and <code>../model/model</code> respectively.

- Check <code>code/options.py</code> for all running options.

<br>

## Dependencies
Python >= 3.6, TensorFlow 1.3.0

As always it is a pain to manage versions of python and it's libraries.
This repo relies on `tensorflow==1.3.0` which is available only for
python 3.6, so you will have to install it if you don't have it. In
order to run this code before proceeding with instructions above do the
following:

1. `sudo apt update && sudo apt install python3.6` skip if you already
   have python3.6 installed
2. `python3.6 -m venv venv` create virtual environment named
   `venv`
3. `source venv/bin/activate` activate newly created environment
   (do this every time you want to run a code in this repo)
4. `pip install -r requirements.txt`

(tested on Ubuntu 18).
