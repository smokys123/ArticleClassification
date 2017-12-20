#!/usr/bin/env python
import os
import numpy as np
import pandas as pd

import gensim
from gensim.models import Word2Vec

fpath=os.path.dirname(os.path.abspath(__file__))
model = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(fpath,'word2vec','model','GoogleNews-vectors-negative300.bin'), binary=True)

voc_path=os.path.join(fpath,'dataset/bbc/vocabulary.txt')
fp=open(voc_path,'r')

nvoc=int(fp.readline())
print("vocabulary size : ",nvoc)

vocabulary=fp.read().split()
fp.close()

wordvecs=[]
unk_wordlist=[]
for word in vocabulary:
	if word in model.vocab:
		wordvecs.append(model[word])
	else:
		unk_wordlist.append(word)
		wordvecs.append(np.zeros((300)))

df=pd.DataFrame(wordvecs, index=vocabulary)
df.to_csv(os.path.join(fpath,'dataset/bbc/word2vec.csv'), header=None)

fp=open(os.path.join(fpath, 'dataset/bbc/unk_wordlist.txt'),'w')
for word in unk_wordlist:
	fp.write(word)
	fp.write('\n')
fp.close()
