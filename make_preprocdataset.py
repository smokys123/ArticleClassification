!#/usr/bin/env python
# coding: utf-8
import os,sys
import re
import nltk
import csv
import pandas as pd
import numpy as np
from textblob import Word
from nltk import word_tokenize
from nltk.corpus import stopwords


nltk.download("stopwords")
nltk.download("wordnet")
"""
data-preprocessing
upper -> lower
special word -> <spc>
cut Article into sentence
hading quotes


skip-grams

"""
#def clean_quotes(string):


def clean_str(string):
    """
    Tokenization/string cleaning for datasets.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"\'s", "", string)
    string = re.sub(r"\'ve", "", string)
    string = re.sub(r"n\'t", "", string)
    string = re.sub(r"\'re", "", string)
    string = re.sub(r"\'d", "", string)
    string = re.sub(r"\'ll", "", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"!", "", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"'", "", string)
    string = re.sub(r"\...","",string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`.]", " ", string)
    string = re.sub(r"[0-9]\w+|[0-9]","", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


news_data = pd.read_csv('dataset/dataset.csv',encoding='ISO-8859-1')
news = news_data['news'].tolist()
types = news_data['type'].tolist()
stop = set(stopwords.words('english'))

#upper -> lower, special word -> <spc>
for idx, val in enumerate(news):
    #print ("processing data :",idx)
    #news[idx] = ' '.join([Word(word).lemmatize() for word in clean_str(val).split()])
	news[idx] = ' '.join([word for word in clean_str(val).split()])

#remove stopwords
for idx, val in enumerate(news):
	news[idx] = ' '.join(word for word in val.split() if word not in stop)
	print ("processing data :",idx)
	print (news[idx])

#make csv file
"""
with open('preproc_dataset.csv','wb') as dataset:
    filewriter = csv.writer(dataset,delimiter=',')
    for i in range(len(news)):
        filewriter.writerow([news[i],types[i]])
"""
data = {'news': news, 'type': types}
df = pd.DataFrame(data)
print ('writing csv flie ...')
df.to_csv('/home/smoky/Article_classification/dataset/preproc_dataset.csv',sep=',', index=False)

