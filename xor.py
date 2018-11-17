"""
Toy script to get used to LSTM
and GRU on PyTorch.

Trains a single-cell recurrent
neural network to act as a XOR
gate on bit sequences of arbitrary
lengths.

Author: Arnaldo E. Pereira
"""

import numpy as np
import torch
#import torch.nn as nn
from toy_nets import OneCell
from datagen import datagen_xor


if __name__ == "__main__":
    epochs = 100
    batch_size = 256
    batches_per_epoch = 100
    # Maximum sequence length during training
    max_seq_len = 50
    cell = 'GRU'

    x_test, y_test = next(datagen_xor(batch_size=1000,
                                  max_len=100))

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    model = OneCell(device, cell_type=cell,
                    mini_batch=batch_size).to(device)

    model.train(x_test,
                y_test,
                max_seq_len,
                epochs=epochs,
                batch_size=batch_size,
                batches_per_epoch = batches_per_epoch)
