#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import namedtuple
from gensim import corpora
from gensim.models.doc2vec import Doc2Vec
from gensim.models import word2vec
from gensim.models import doc2vec
import matplotlib.pyplot as plt
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import os
import pandas as pd
import random
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
path = os.getcwd()

# Contractions to help in preprocessing
contractions = { "ain't": "am not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have", "'cause": "because", "could've": "could have", "couldn't": "could not", "couldn't've": "could not have", "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'd've": "he would have", "he'll": "he will", "he's": "he is", "how'd": "how did", "how'll": "how will", "how's": "how is", "i'd": "i would", "i'll": "i will", "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'll": "it will", "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have", "mightn't": "might not", "must've": "must have", "mustn't": "must not", "needn't": "need not", "oughtn't": "ought not", "shan't": "shall not", "sha'n't": "shall not", "she'd": "she would", "she'll": "she will", "she's": "she is", "should've": "should have", "shouldn't": "should not", "that'd": "that would", "that's": "that is", "there'd": "there had", "there's": "there is", "they'd": "they would", "they'll": "they will", "they're": "they are", "they've": "they have", "wasn't": "was not", "we'd": "we would", "we'll": "we will", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what're": "what are", "what's": "what is", "what've": "what have", "where'd": "where did", "where's": "where is", "who'll": "who will", "who's": "who is", "won't": "will not", "wouldn't": "would not", "you'd": "you would", "you'll": "you will", "you're": "you are"}

# A method to ease up the cleaning of data
def clean_text(text, remove_stopwords = True):
    text = text.lower().encode('utf-8').split()
    new_text = []
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

# Get your corpus, dump it into a dataframe.
print "Loading files to build model ..."
df = pd.DataFrame()
files = os.listdir('/home/darthbhyrava/2018/doc2vec/data/')
files_xls = [f for f in files if f[-4:] == 'xlsx']
for f in files_xls:
    data = pd.read_excel(path+'/data/{0}'.format(f), encoding = "utf_8")
    df = df.append(data)
data = df[["text", "user_name", "lang"]]
data_en = data[data["lang"] == "en"]
data_en = data_en.drop_duplicates()
index = data_en.index[data_en["text"] == 0]
data_en = data_en.drop(index = index)

orig_text = data_en["text"]
print len(orig_text)


# Reading the tweets we need to classify.
print "Loading test data ..."
inp = pd.DataFrame()
input_files = os.listdir('/home/darthbhyrava/2018/doc2vec/input/')
input_files_list = [f for f in input_files if f[-4:] == 'xlsx']
for f in input_files_list:
	input_files = pd.read_excel(path+'/input/{0}'.format(f), encoding = "utf_8")
	inp = inp.append(input_files)
inp_data_en = inp[["text", "Party"]]
inp_data_en = inp_data_en.drop_duplicates()
index = inp_data_en.index[inp_data_en["text"] == 0]
inp_data_en = inp_data_en.drop(index = index)

new_text = inp_data_en["text"]
tot = len(new_text)
print tot


indices = []
for ind, i in enumerate(orig_text):
	if i in new_text:
		indices.append(ind)
indices_len = len(indices)
print (float(indices_len)/float(tot))




# # Clean up your data
# print "Cleaning up data ..."
# data_en["clean_text"] = data_en["text"].apply(lambda x: clean_text(x))
# inp_data_en["clean_text"] = inp_data_en["text"].apply(lambda x: clean_text(x))
# data_en["tokenized_text"] = data_en["clean_text"].apply(lambda x: word_tokenize(x))
# inp_data_en["tokenized_text"] = inp_data_en["clean_text"].apply(lambda x: word_tokenize(x))
# doc1 = data_en["tokenized_text"]
# doc2 = inp_data_en["tokenized_text"]
# docs_tokenized = []
# inp_docs_tokenized = []
# analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
# for i, text in enumerate(doc1):
#     words = text
#     tags = [i]
#     docs_tokenized.append(analyzedDocument(words, tags))
# for i, text in enumerate(doc2):
#     words = text
#     tags = [i]
#     inp_docs_tokenized.append(analyzedDocument(words, tags))
# print "We have {0} tweets in our corpus ...".format(len(docs_tokenized))
# print "We have {0} tweets in our input ...".format(len(inp_docs_tokenized))

# # Let's build a doc2vec model on the data. Change the parameters as necessary. Once you save it, you can load the model directly.
# print "Building a model ..."
# model_doc2vec_dm = doc2vec.Doc2Vec(docs_tokenized, vector_size= 750, dm =1, alpha = 0.01, epochs = 5,min_count =50, seed = 123)
# print "Saving model to disk ..."
# model_doc2vec_dm.save('./d2v_dm.d2v')
# model_doc2vec_dm = Doc2Vec.load('./d2v_dm.d2v')

# # These are our list of political and apolitical tweets, by index.
# pol = [76851, 76888, 76998, 87146, 87160, 87246, 89473, 126196, 293433, 293439, 293448, 293462 ,293464, 293465, 293490, 293491, 293510, 293515, 293550, 293579, 293583, 293665, 293704, 293719, 293752, 293755, 293761, 370943, 458920]
# apol = [60445, 77013, 77045, 77070, 86579, 86586, 86593, 87535, 89408, 89409, 117934, 124518, 124577, 125432, 125435, 126329, 127036, 210456, 210457, 214388, 293450, 293454, 293458, 293467, 293494, 293498, 293518, 293600, 293601, 293606, 293681, 293768, 293773, 293832, 293905, 293926, 294183, 320098, 337429, 361904, 387681, 428865, 429104, 429144, 431558, 505937, 569876, 597006, 597007, 617270, 671741, 724829, 736297]


