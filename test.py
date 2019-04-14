import argparse
import torch
from flamebot import RNN
from torch.autograd import Variable
from collections import deque


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="model tester")
    parser.add_argument('starting_char', type=str, help='Seed character')
    parser.add_argument('model_file', type=str, help='PyTorch model to load',
                        default='flamebot2000.pt')

    args = parser.parse_args()
    model = RNN(128, 40, 128, 128)
    model.load_state_dict(torch.load(args.model_file))
    seed = args.starting_char * 40

    def string_to_byte(string):
        return [ord(s) for s in string]

    def byte_to_string(byte):
        return [str(byte) for b in byte]

    out = deque(string_to_byte(seed), 40)
    for i in range(40):
        d = model(torch.tensor(list(out), dtype=torch.long)
                  .unsqueeze(0))
        out.append(d.tolist())

    print("generated:" + byte_to_string(out.tolist()))
