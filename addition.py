"""
Toy script to get used to LSTM
and GRU on PyTorch.

Trains a single-layer recurrent
neural network to add sequences
of arbitrary lengths of binary
numbers.

Author: Arnaldo E. Pereira
"""

import torch
from toy_nets import AdditionCell
from datagen import datagen_add


if __name__ == "__main__":
    epochs = 10
    batch_size = 256
    batches_per_epoch = 100
    # Maximum sequence length during training
    max_seq_len = 10
    cell = 'LSTM'

    x_test, y_test = next(datagen_add(batch_size=1000,
                                      max_len=50))

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    model = AdditionCell(device, cell_type=cell,
                         mini_batch=batch_size).to(device)

    model.train(x_test,
                y_test,
                max_seq_len,
                epochs=epochs,
                batch_size=batch_size,
                batches_per_epoch=batches_per_epoch)
