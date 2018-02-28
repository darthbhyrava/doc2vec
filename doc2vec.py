#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import Counter
from gensim import corpora
from gensim.models import word2vec
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import gensim
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re

path = os.getcwd()
files = os.listdir(path)
files_xlxs = [f for f in files if f[-4:] == 'xlsx']

df = pd.DataFrame()
for f in files_xlxs:
	data = pd.read_excel(f, encoding='utf-8')
	df = df.append(data)

data = df[['text', 'user_name','lang']]
data_en = data[data['lang'] == 'en'].drop_duplicates()

contractions = { 
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he's": "he is",
"how'd": "how did",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'll": "i will",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'll": "it will",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"must've": "must have",
"mustn't": "must not",
"needn't": "need not",
"oughtn't": "ought not",
"shan't": "shall not",
"sha'n't": "shall not",
"she'd": "she would",
"she'll": "she will",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"that'd": "that would",
"that's": "that is",
"there'd": "there had",
"there's": "there is",
"they'd": "they would",
"they'll": "they will",
"they're": "they are",
"they've": "they have",
"wasn't": "was not",
"we'd": "we would",
"we'll": "we will",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"where'd": "where did",
"where's": "where is",
"who'll": "who will",
"who's": "who is",
"won't": "will not",
"wouldn't": "would not",
"you'd": "you would",
"you'll": "you will",
"you're": "you are"
}

def clean_text(text, remove_stopwords = True):
    new_text = []
    text = text.lower().split()
    for word in text:
        if word in contractions:
            new_text.append(contractions[word])
        else:
            new_text.append(word)

    text = " ".join(new_text)
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', '', text)
    text = re.sub(r'&amp;', '', text) 
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', '', text)
    text = re.sub(r'\'""', '', text)
    text = text.replace('‘', '').strip()
    text = text.replace('’', '').strip()
    
    if remove_stopwords:
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)
    return text

data_en['clean_text'] = data_en['text'].apply(lambda x: clean_text(x.encode('utf-8')))
total = list(data_en['clean_text'])

tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
tf = tf_vectorizer.fit_transform(total)
tf_feature_names = tf_vectorizer.get_feature_names()

no_topics = 20
lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))
no_top_words = 6
# display_topics(lda, tf_feature_names, no_top_words)

users = data_en.groupby('user_name')['clean_text'].apply(list).reset_index()
users["users_text"] = users["clean_text"].apply(lambda x: "".join(str(i) for i in x))

pol_list = ['congress', 'inc', 'modi','gandhi','jds','cmofkarnataka','siddaramaiah']
for i in pol_list:
	political = users[users["users_text"].str.contains(i, case = False)]

users["users_text"] = users["users_text"].apply(lambda x: nltk.word_tokenize(x))
users["count"] = users["users_text"].apply(lambda x: Counter(x))

model = gensim.models.Word2Vec(users["users_text"], min_count=5)
congress = model.wv.most_similar('modi', topn =5)
print congress