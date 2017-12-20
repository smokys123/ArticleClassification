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
        # layer 1 = word_bi_gru & conv(bi,tri,quad)
        self.word_bi_GRU = nn.GRU(wordvector_dim, word_hdim,
                                  num_layers=1, bias=True, dropout=0.8,
                                  bidirectional=True)
        self.quad_conv = nn.Conv2d(1, 10, (4, 2*self.word_hdim))
        # layer 2 = sen_bi_gru & conv
        self.sen_bi_GRU = nn.GRU(10, sen_hdim, num_layers=1, bias=True,
                                 dropout=0.8, bidirectional=True)
        self.senCNN = nn.Conv2d(1, 10, (4, 2*self.sen_hdim))  # for test
        # self.batchNormal = nn.BatchNorm1d(10)
        self.fc1 = nn.Linear(10, 5)

    def forward(self, x, num_seq, len_seq, num_word):
        x = Variable(torch.LongTensor(x))
        x = self.embedding(x)
        # print("Article embedding is done!")
        # embedding batch normalization
        normalized_x = None
        for i in range(num_seq):
            x_ = x[i][:num_word[i]]
            x_ = self.batchNormWord(x_)
            if num_word[i] < len_seq:
                y_ = torch.cat([x_, Variable(torch.zeros(len_seq-num_word[i],
                                                         self.worddim))],
                               dim=0).unsqueeze(dim=0)
            else:
                y_ = x_.unsqueeze(dim=0)
            if normalized_x is None:
                normalized_x = y_
            else:
                normalized_x = torch.cat([normalized_x, y_], dim=0)
        x = F.relu(normalized_x)

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
        x3 = F.relu(self.quad_conv(x)).squeeze(3)
        x3 = F.max_pool1d(x3, x3.size(2)).squeeze(2)

        # sen level bidirectional GRU  x = (1,(num_seq-2)*3, 10)
        lens1 = []
        for i in range(num_seq):
            lens1.append(1)
        x = x3.unsqueeze(0)
        x = torch.transpose(x, 0, 1)
        # x = x3
        x = nn.utils.rnn.pack_padded_sequence(x, lens1, batch_first=True)
        h1 = Variable(torch.randn(2, num_seq, self.sen_hdim))
        x, _ = self.sen_bi_GRU(x, h1)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = F.relu(x)

        # senCNN(1, num_seq-2, 20)
        x = x.unsqueeze(0)  # x = (1,1,num_seq-2,2*sen_hdim) #sen_hdim = 10
        x = torch.transpose(x, 1, 2)
        x = F.relu(self.senCNN(x))  # x = (1, 10, num_seq-1, 1
        x = torch.squeeze(x, 0)  # x = (10 ,num_seq-1,1)
        x = torch.transpose(x, 1, 2)  # x = (10, 1, num_se1-1)
        x = F.max_pool1d(x, x.size(2))  # x = (10, 1, 1)
        x = x.contiguous()
        x = x.view(-1, 10)
        # x = self.batchNormal(x)
        x = self.fc1(x)

        return x
