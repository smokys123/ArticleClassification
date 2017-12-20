#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class GRU_CNN(nn.Module):

    def __init__(self, config, wordvector_dim, word_hdim, sen_hdim):
        super(GRU_CNN, self).__init__()
        self.worddim = wordvector_dim
        self.word_hdim = word_hdim
        self.sen_hdim = sen_hdim
        self.cnn_outchannel = 10
        self.embedding = nn.Embedding(21582, wordvector_dim)
        self.batchNormWord = nn.BatchNorm1d(wordvector_dim)
        self.word_bi_GRU = nn.GRU(wordvector_dim, word_hdim, num_layers=1,
                                  bias=True, dropout=0.8, bidirectional=True)
        self.bi_conv = nn.Conv2d(1, 10, (2, 2*self.word_hdim))
        self.tri_conv = nn.Conv2d(1, 10, (3, 2*self.word_hdim))
        self.quad_conv = nn.Conv2d(1, 10, (4, 2*self.word_hdim))
        # self.batchNormal = nn.BatchNorm1d(30)
        self.fc = nn.Linear(30, 5)

    def forward(self, x, num_seq, len_seq, num_word):
        x = Variable(torch.LongTensor(x))
        x = self.embedding(x)

        """
        # embedding batch normalization
        normalized_x = None
        for i in range(num_seq):
        x_ = x[i][:num_word[i]]
        x_ = self.batchNormWord(x_)
        if num_word[i] < len_seq:
            y_ = torch.cat([x_, Variable(torch.zeros(len_seq-num_word[i],
                            self.worddim))], dim=0).unsqueeze(dim=0)
        else:
            y_ = x_.unsqueeze(dim=0)
        if normalized_x is None: normalized_x = y_
        else: normalized_x = torch.cat([normalized_x,y_],dim=0)
        x = F.relu(normalized_x)
        """

        # word level bidirectional GRU -
        # x.shape = (num_seq,len_seq,wordvector_dim)
        lens = torch.LongTensor(num_word)
        lens, ordered_idx = lens.sort(dim=0, descending=True)
        lens = lens.tolist()
        x = x[ordered_idx]

        x = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True)
        h0 = Variable(torch.randn(2, num_seq, self.word_hdim),
                      requires_grad=False)
        x, _ = self.word_bi_GRU(x, h0)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = x[ordered_idx]
        x = F.relu(x)

        # word CNN
        x = x.unsqueeze(1)
        x1 = torch.squeeze(F.relu(self.bi_conv(x)), 3)
        x2 = torch.squeeze(F.relu(self.tri_conv(x)), 3)
        x3 = torch.squeeze(F.relu(self.quad_conv(x)), 3)
        x1 = F.max_pool1d(x1, x1.size(2)).squeeze(2)
        x2 = F.max_pool1d(x2, x2.size(2)).squeeze(2)
        x3 = F.max_pool1d(x3, x3.size(2)).squeeze(2)
        x = torch.cat([x1, x2, x3], dim=1)
        x = x.unsqueeze(0)
        x = torch.transpose(x, 1, 2)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        # x = self.batchNormal(x)
        x = self.fc(x)

        return x
