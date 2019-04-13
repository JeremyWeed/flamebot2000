import os
import argparse
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from torch.autograd import Variable


class RNN(nn.Module):
    def __init__(self, n_chars, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.n_chars = n_chars
        self.hidden_size = hidden_size
        self.input_size = input_size

        # char embedding layer, needs one hot input
        self.embedding = nn.Embedding(self.n_chars, 100)
        # 1st lstm
        self.lstm1 = nn.LSTM(input_size=self.n_chars,
                             hidden_size=self.hidden_size)
        # 2nd lstm layer
        self.lstm2 = nn.LSTM(input_size=self.hidden_size,
                             hidden_size=self.hidden_size)
        # attention layer
        self.attn = nn.Linear(self.n_chars+2*self.hidden_size,
                              self.n_chars+2*self.hidden_size)
        # dense output layer
        self.output = nn.Linear(self.n_chars+2*self.hidden_size, self.n_chars)
        # softmax output for probabilities
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x is input of length 40? of character indices
        embeddings = self.embedding(x)
        l1 = self.lstm1(embeddings)
        l2 = self.lstm1(l1)
        attn_weights = F.softmax(self.attn(torch.cat((embeddings, l1, l2), 1)),
                                 dim=1)
        output = self.softmax(self.output(attn_weights))
        return output

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


def train(model, loader, optimizer, criterion, epoch, device):
    model.train()
    for i, (input, target) in enumerate(loader):
        input = Variable(input)
        target = Variable(target)
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print('Train: epoch {}\t'.format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='code')
    parser.add_argument('--data_path', type=str, default=None,
                        help='path to dataset')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.01,
                        help='momentum')
    parser.add_argument('--epochs', type=int, default=10,
                        help='epochs')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = RNN(40, 40, 40, 40)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        train(model, train_loader, optimizer, criterion, epoch, device)
        test(model, test_loader, criterion, epoch, device)
