#!/usr/bin/env python
# coding: utf-8
import os
import pandas as pd

import re

fpath=os.path.dirname(os.path.abspath(__file__))
file_path=os.path.join(fpath, 'dataset/bbc/preproc_dataset.csv')
vocabulary=[]
df=pd.read_csv(file_path)

#extract word from preprocdataset.csv(dataset/bbc/preprocdataset.csv) 
for i, document in enumerate(df['news']):
	document=re.sub(r'\.','',document)
	words=document.split()
	words=[x.lstrip().rstrip() for x in words]
	for word in words:
		if not word in vocabulary: vocabulary.append(word)
	if i%50==0: print(i)

nvoc=len(vocabulary)
fp=open(os.path.join(fpath, 'dataset/bbc/vocabulary.txt'),'w')
fp.write(str(nvoc))
fp.write('\n')
for i, word in enumerate(vocabulary):
	fp.write(word+'\t')
	if i%10==0: fp.write('\n')
fp.close()
