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

    df_stop_sentences = pd.DataFrame(predictions_stop_sentences).stack().apply(pd.Series).reset_index(names=['index', 'emotion'])
    df_stop_sentences = df_stop_sentences.pivot_table(columns='label', index='index', values='score')

    df_stop_sentences['sentence'] = df_stop_sentences.index.map(lambda x: text_stop_sentences[x])

    return df_stop_sentences
