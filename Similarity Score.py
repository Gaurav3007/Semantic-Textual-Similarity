import nltk
import numpy as np
import pandas as pd

import re
from tqdm import tqdm

import collections

from sklearn.cluster import KMeans

from nltk.stem import WordNetLemmatizer  # For Lemmetization of words
from nltk.corpus import stopwords  # Load list of stopwords

from nltk import word_tokenize # Convert paragraph in tokens

import pickle
import sys

from gensim.models import word2vec       # For represent words in vectors
import gensim

text_data = pd.read_csv("Precily_Text_Similarity.csv")
print("Shape of text_data : ", text_data.shape)
print(text_data.head(3))

print(text_data.isnull().sum())


nltk.download('wordnet')
nltk.download('omw-1.4')



def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


preprocessed_text1 = []

# tqdm is for printing the status bar

for sentance in tqdm(text_data['text1'].values):
    sent = decontracted(sentance)
    sent = sent.replace('\\r', ' ')
    sent = sent.replace('\\"', ' ')
    sent = sent.replace('\\n', ' ')
    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)

    sent = ' '.join(e for e in sent.split() if e not in stopwords.words('english'))
    preprocessed_text1.append(sent.lower().strip())




text_data['text1'] = preprocessed_text1
text_data.head(3)

print(text_data.head(3))

# Combining all the above stundents
from tqdm import tqdm

preprocessed_text2 = []

# tqdm is for printing the status bar
for sentance in tqdm(text_data['text2'].values):
    sent = decontracted(sentance)
    sent = sent.replace('\\r', ' ')
    sent = sent.replace('\\"', ' ')
    sent = sent.replace('\\n', ' ')
    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)

    sent = ' '.join(e for e in sent.split() if e not in stopwords.words('english'))
    preprocessed_text2.append(sent.lower().strip())


text_data['text2'] = preprocessed_text2

text_data.head(3)

print(text_data.head(3))


def word_tokenizer(text):
    # tokenizes and stems the text
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens


similarity = []  # List for store similarity score


s1 = text_data['text1']
s2 = text_data['text2']


combine_text = s1 +s2


print(combine_text)


model_name = "bert-base-nli-mean-tokens"







