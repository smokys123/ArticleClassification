#!/usr/bin/env python3
import random
import os
import pandas as pd


class document_DataSet(object):
    def __init__(self, data, classes, valid_idxs=None):
        self.data = data
        # data =(class, number of sentences, [number of words], [w1, w2, ... ])
        total_num_examples = self.get_data_size()
        self.valid_idxs = range(total_num_examples) if valid_idxs is None else valid_idxs
        self.valid_data = data if valid_idxs is None else self.get_valid_data()
        self.num_examples = len(self.valid_idxs)
        self.classes = classes
        self.nclass = len(classes)
        self.vocabulary = self.get_vocabulary()
        self.nvoc = len(self.vocabulary)+1
        # vocabulary size, +1 is for special symbol <pad>
        self.word_to_idx = self.get_word_to_idx()
        self.idx_to_word = self.get_idx_to_word()
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_class = {i: c for i, c in enumerate(self.classes)}

    def get_valid_data():
        valid_data = []
        for idx in self.valid_idxs:
            valid_data.append(self.data[idx])
        return valid_data

    def get_data_size(self):
        return len(self.data)

    def get_vocabulary(self):
        # for entire set
        current_path = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_path, 'dataset', 'BBC')
        voc_path = os.path.join(data_path, 'vocabulary.txt')
        fp = open(voc_path, 'r')
        # nvoc=int(fp.readline())
        vocabulary = fp.read().split()
        fp.close()

        # for tesing
        """
        voc=[]
        # (class, number of sentence, number of words, news)
        for _, num_sen_in_doc, num_word_in_sen, news in self.valid_data:
        # (target_class, length of sentence, [sentence])
        for i in range(num_sen_in_doc):
        for j in range(num_word_in_sen[i]):
        word = news[i][j]
        if word not in voc: voc.append(word)
        return voc
        """
        return vocabulary

    def get_word_to_idx(self):
        # no special symbols <s>, </s>
        word2idx = {word: i+1 for i, word in enumerate(self.vocabulary)}
        word2idx["<pad>"] = 0
        return word2idx

    def _word_to_idx(self):
        for idx in range(self.num_examples):
            doc = self.valid_data[idx][3]
            num_sen_in_doc = self.valid_data[idx][1]
            num_word_in_sen = self.valid_data[idx][2]
            for i in range(num_sen_in_doc):
                for j in range(num_word_in_sen[i]):
                    doc[i][j] = self.word_to_idx[doc[i][j]]

    def padding_doc(self):
        for idx in range(self.num_examples):
            doc = self.valid_data[idx][3]
            num_sen_in_doc = self.valid_data[idx][1]
            num_word_in_sen = self.valid_data[idx][2]
            max_len_sen = 0
            for num in num_word_in_sen:
                if max_len_sen < num:
                    max_len_sen = num
            for i in range(num_sen_in_doc):
                for j in range(max_len_sen):
                    if j >= (num_word_in_sen[i]):
                        doc[i].append(0)

    def get_idx_to_word(self):
        # no special symbols <s>, </s>
        idx2word = {i+1: word for i, word in enumerate(self.vocabulary)}
        idx2word[0] = "<pad>"
        return idx2word

    def _sort_by_len(self): raise NotImplementedError()

    def _sort_by_senlen(self): raise NotImplementedError()

    def split_set(self): raise NotImplementedError()

    def get_by_idxs(self, idxs): raise NotImplementedError()

    def get_one(self, idx): raise NotImplementedError()

    def get_batches(self, batch_size, num_batches=None, shuffle=False):
        return self.valid_data

    def get_max_len(self):
        max_len = 0
        for i in self.valid_idxs:
            _, length, _ = self.data[i]
            if length > max_len:
                max_len = length
        return max_len

    def get_mean_len(self):
        tot_len = 0
        for i in self.valid_idxs:
            _, length, _ = self.data[i]
            tot_len += length
        return tot_len/self.num_examples

    def show_length_distribution(self):
        raise NotImplementedError()


def read_bbc(config):
    def preprocess(document):  # if sentence is long cut sentence
        doc = [x.split() for x in document.split('.') if len(x.strip()) > 0]
        truncate_length = 50
        fdoc = []
        for sen in doc:
            if len(sen) <= truncate_length: fdoc.append(sen)
            else:
                trunc_sen = [x for x in sen]
                while True:
                    if len(trunc_sen) <= truncate_length:
                        fdoc.append(trunc_sen)
                        break
                    else:
                        fdoc.append(trunc_sen[:truncate_length])
                        trunc_sen = trunc_sen[truncate_length:]

        num_sentence = len(fdoc)
        num_word_in_sen = [len(x) for x in fdoc]
        return fdoc, num_sentence, num_word_in_sen

    def make_data(df):
        data = []
        for rowIdx in df.index:
            target_class = df.ix[rowIdx]['type']
            news, num_sen_in_doc, num_word_in_sen = preprocess(df.ix[rowIdx]['news'])
            if num_sen_in_doc != 94:
                data.append((target_class, num_sen_in_doc,
                             num_word_in_sen, news))
        return data

    current_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_path,
                             config.data_dir,
                             config.data_type)
    train_path = os.path.join(data_path, 'train_data.csv')
    test_path = os.path.join(data_path, 'test_data.csv')

    train_df = pd.read_csv(train_path, encoding='ISO-8859-1')
    test_df = pd.read_csv(test_path, encoding='ISO-8859-1')

    train_data = make_data(train_df)
    test_data = make_data(test_df)

    train_data = train_data[:1000]
    test_data = test_data[:200]

    """
    data -- list of tuple
    (class, num of sen, num words in each sen,
    [[w1, w2, w3], [w1, w2], ... ])
    """

    random.shuffle(train_data)  # 2225 instances
    random.shuffle(test_data)

    return (document_DataSet(train_data, classes=['business', 'entertainment', 'politics', 'sport', 'tech']),
            document_DataSet(test_data, classes=['business', 'entertainment', 'politics', 'sport', 'tech']))

