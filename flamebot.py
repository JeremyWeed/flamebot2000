import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

class RNN(nn.Module):
    def __init__(self, n_chars, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        
        self.n_chars = n_chars
        self.hidden_size = hidden_size
        self.input_size = input_size
        
        #char embedding layer, needs one hot input
        self.embedding = nn.Embedding(self.n_chars, 100)
        #1st lstm
        self.lstm1 = nn.LSTM(input_size = self.n_chars, hidden_size = self.hidden_size)
        #2nd lstm layer
        self.lstm2 = nn.LSTM(input_size = self.hidden_size, hidden_size = self.hidden_size)
        #attention layer
        self.attn = nn.Linear(self.n_chars+2*self.hidden_size, self.n_chars+2*self.hidden_size)
        #dense output layer
        self.output = nn.Linear(self.n_chars+2*self.hidden_size, self.n_chars)
        #softmax output for probabilities
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        #x is input of length 40? of character indices
        embeddings = self.embedding(x)
        l1 = self.lstm1(embeddings)
        l2 = self.lstm1(l1)
        attn_weights = F.softmax(self.attn(torch.cat((embeddings, l1, l2), 1)), dim=0)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), )
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)