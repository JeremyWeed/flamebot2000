import argparse
import torch
from flamebot import RNN
from torch.autograd import Variable


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="model tester")
    parser.add_argument('starting_char', type=str, help='Seed character')
    parser.add_argument('model_file', type=str, help='PyTorch model to load',
                        default='flamebot.pt')
    parser.add_argument('string_length', type=int, help='Length of string to'
                        + 'generate')

    args = parser.parse_args()
    model = RNN()
    model.load_state_dict(torch.load(args.model_file))
    seed = args.starting_char
    out = str(model(ord(seed)))
    for i in range(args.string_length):
        out += str(model(out[-1]))
