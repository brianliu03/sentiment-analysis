# imports
from transformers import pipeline
import string
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import chain
from os import listdir
from os.path import isfile, join

def text_stop_sentences():
    nltk.download('stopwords')
    # stop words for filtering
    stopwords = set(nltk.corpus.stopwords.words('english'))

    # load different texts
    # text = open('text_stimuli/01_1.txt').read()[58:]
    # read for all files inside of text_stimuli, no need to [58:] for all
    text = ''
    for f in sorted(listdir('text_stimuli')):
        if isfile(join('text_stimuli', f)):
            text += open(join('text_stimuli', f)).read()
    text = text[58:]
    text = text.replace('\n', ' ')

    text_stop_sentences = [
        ' '.join([word for word in s.split() if word.lower() not in stopwords])
        for s in nltk.sent_tokenize(text)
    ]

    classifier = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", top_k=None)

    predictions_stop_sentences = classifier(text_stop_sentences)

    df_stop_sentences = pd.DataFrame(predictions_stop_sentences).stack().apply(pd.Series).reset_index(names=['sentence', 'emotion'])
    df_stop_sentences = df_stop_sentences.pivot_table(columns='label', index='sentence', values='score')

    return df_stop_sentences
