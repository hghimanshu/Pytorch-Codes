#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 15:43:30 2019

@author: techject
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
#import gym

sentences = ['How may I help you?',
             'Can I be of assistance?',
             'May I help you with something?',
             'May I assist you?','Do you need any help?','Can I assist you with something?','How can I help you?']

#tokenize the data
words = dict()
reverse = dict()
i = 0
for s in sentences:
    s = s.replace('?',' <unk>')
    for w in s.split():
        if w.lower() not in words:
            words[w.lower()] = i
            reverse[i] = w.lower()
            i = i + 1
            
#tokenize the data

#embedding layer is treated as any other, getting parameter updates via backpropagation
    
class DataGenerator():
    def __init__(self, dset):
        self.dset = dset
        self.len = len(self.dset)
        self.idx = 0
    def __len__(self):
        return self.len
    def __iter__(self):
        return self
    def __next__(self):
        x = Variable(torch.LongTensor([[self.dset[self.idx]]]), requires_grad=False)
        if self.idx == self.len - 1:
            raise StopIteration
        y = Variable(torch.LongTensor([self.dset[self.idx+1]]), requires_grad=False)
        self.idx = self.idx + 1
        return (x, y)
    


#class simpleBot(nn.Module):
#    def __init__(self):
#        super().__init__()
#        self.embedding = nn.Embedding(len(words), 10)
#        self.rnn = nn.LSTM(10, 20, 2, dropout=0.5)
#        self.h = (Variable(torch.zeros(2, 1, 20)), Variable(torch.zeros(2, 1, 20)))
#        self.l_out = nn.Linear(20, len(words))
#        
#    def forward(self, cs):
#        inp = self.embedding(cs)
#        outp,h = self.rnn(inp, self.h)
#        out = F.log_softmax(self.l_out(outp), dim=-1).view(-1, len(words))
#        return out


class simpleBot(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(words), 10)
        self.rnn = nn.LSTM(10, 20, 2, dropout=0.5,bidirectional=True)
        self.rnn2=nn.LSTM(40,20,2,bidirectional=True)
        self.h = (Variable(torch.zeros(4, 1, 20)), Variable(torch.zeros(4, 1, 20)))
        self.l_out = nn.Linear(40, len(words))
        
    def forward(self, cs):
        inp = self.embedding(cs)
        outp,h = self.rnn(inp, self.h)
        outp2,h2=self.rnn2(outp,h)
        out = F.log_softmax(self.l_out(outp2), dim=-1).view(-1, len(words))
        return out




    
m= simpleBot()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(m.parameters(), lr=0.01)
for epoch in range(0,600):
    gen = DataGenerator([words[word.lower()] for word in ' '.join(sentences).replace('?',' <unk>').split(' ')])
    for x1, y1 in gen:
        m.zero_grad()
        output = m(x1)
        loss = criterion(output, y1)
        loss.backward()
        optimizer.step()
print(loss)

def get_next(word_):
    word = word_.lower()
    out = m(Variable(torch.LongTensor([words[word_]])).unsqueeze(0))
    return reverse[int(out.max(dim=1)[1].data)]

def get_next_n(word_, n=3):
    print(word_)
    for i in range(0, n):
        word_ = get_next(word_)
        print(word_)

get_next_n('can', n=3)
    
