import os
import argparse
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from torch.autograd import Variable
from torch.utils import data


class RNN(nn.Module):
    def __init__(self, n_chars, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.n_chars = n_chars
        self.hidden_size = hidden_size
        self.input_size = input_size

        # char embedding layer, needs one hot input
        self.embedding = nn.Embedding(self.n_chars, 100)
        # 1st lstm
        self.lstm1 = nn.LSTM(input_size=100,
                             hidden_size=self.hidden_size)
        # 2nd lstm layer
        self.lstm2 = nn.LSTM(input_size=self.hidden_size,
                             hidden_size=self.hidden_size)
        # attention layer
        self.attn = nn.Linear(100+2*self.hidden_size,
                              self.n_chars+2*self.hidden_size)
        # dense output layer
        self.output = nn.Linear(self.n_chars+2*self.hidden_size, self.n_chars)
        # softmax output for probabilities
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x is input of length 40? of character indices
        embeddings = self.embedding(x)
        l1, _ = self.lstm1(embeddings)
        l2, _ = self.lstm2(l1)
        attn_weights = F.softmax(self.attn(torch.cat((embeddings, l1, l2), dim=2)),
                                 dim=0)
        output = self.softmax(self.output(attn_weights))
        return output.squeeze()

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, path, n_samples, n_read):
        'Initialization'
        self.path = path
        self.n_samples = n_samples
        self.n_read = n_read
        self.f = open(self.path, 'rb')

    def __len__(self):
        'Denotes the total number of samples'
        return self.n_samples

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        self.f.seek(index)
        X = self.f.read(self.n_read+1)
        X = [int(c) for c in X]
        # Load data and get label
        X = torch.tensor(X[:-1], dtype=torch.long)
        y = torch.tensor(int(X[-1]), dtype=torch.long)
        return X, y


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
    parser.add_argument('--n_chars', type=int, default=128,
                        help='number of chars to feed in at a time')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    file_size = os.path.getsize(args.data_path)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = RNN(args.n_chars, 40, 128, 128)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum)
    criterion = nn.CrossEntropyLoss()
    train_loader = torch.utils.data.DataLoader(
        Dataset(args.data_path, file_size-args.n_chars, args.n_chars),
        batch_size=64, shuffle=True, num_workers=4)
    for epoch in range(args.epochs):
        train(model, train_loader, optimizer, criterion, epoch, device)
