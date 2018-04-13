# TF_DeepNLP

TensorFlow implementations of several models in DeepNLP. The purpose of this repository is to provice implementations of models with consistent public interface. It is not recommended to use these models in real applications since these codes are not optimized well.

This repository is inspired by [DeepNLP-models-Pytorch](https://github.com/DSKSD/DeepNLP-models-Pytorch).

## Contents
|Model                                    |Links                                  |
|-----------------------------------------|---------------------------------------|
|01. Word2Vec                  |[ [notebook](https://nbviewer.jupyter.org/github/belepi93/TF_DeepNLP/blob/master/01.Word2Vec.ipynb) / [model.py](https://github.com/belepi93/TF_DeepNLP/blob/master/models/Word2Vec.py) / [data](https://www.kaggle.com/snap/amazon-fine-food-reviews) / [paper](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)  ]|
|02. GloVe                     |[ [notebook](https://nbviewer.jupyter.org/github/belepi93/TF_DeepNLP/blob/master/02.GloVe.ipynb) / [model.py](https://github.com/belepi93/TF_DeepNLP/blob/master/models/GloVe.py) / [data](https://www.kaggle.com/snap/amazon-fine-food-reviews) / [paper](https://www.aclweb.org/anthology/D14-1162) ]|
|03. Doc2Vec                   |[ [notebook](https://nbviewer.jupyter.org/github/belepi93/TF_DeepNLP/blob/master/03.Doc2Vec.ipynb) / [model.py](https://github.com/belepi93/TF_DeepNLP/blob/master/models/Doc2Vec.py) / [data](https://www.kaggle.com/snap/amazon-fine-food-reviews) / [paper](https://arxiv.org/pdf/1405.4053.pdf) ]|
|04. NER with Window Classifier        |[ [notebook](https://nbviewer.jupyter.org/github/belepi93/TF_DeepNLP/blob/master/04.WC_NER.ipynb) / [model.py](https://github.com/belepi93/TF_DeepNLP/blob/master/models/WindowsClassifier.py) / data / paper ]|


## Requirements
I can't guarantee that models in this repository work well in different library versions/settings.
- Python 3.6.5
- TensorFlow 1.7.0
- nltk 3.2.5

## Author
Younggyo Seo / [@belepi93](https://github.com/belepi93)
