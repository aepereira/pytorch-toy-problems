"""
Modular single-cell recurrent
neural network classes for
the XOR and addition problems.

Author: Arnaldo E. Pereira
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import datagen


class OneCell(nn.Module):

    def __init__(self, device, cell_type='LSTM', mini_batch=64):
        super(OneCell, self).__init__()
        self.device = device
        self.mini_batch = mini_batch
        self.cell = None
        self.hidden = None
        self.loss_function = nn.MSELoss()
        self.datagen = datagen.datagen_xor
        cell_params = {'input_size': 1,
                       'hidden_size': 1,
                       'num_layers': 1,
                       'bias': True,
                       'batch_first': True,
                       'dropout': 0.0,
                       'bidirectional': False}
        if cell_type == 'LSTM':
            self.cell = nn.LSTM(**cell_params)
        elif cell_type == 'GRU':
            self.cell = nn.GRU(**cell_params)
        else:
            raise ValueError("Valid cell types are LSTM and GRU.")

        # Initialize hidden state
        self.reset_hidden(self.mini_batch)

        # Initialize optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)

    def reset_hidden(self, batch_size):
        if isinstance(self.cell, nn.LSTM):
            self.hidden = (torch.zeros(1, batch_size, 1).to(self.device),
                           torch.zeros(1, batch_size, 1).to(self.device))
        elif isinstance(self.cell, nn.GRU):
            self.hidden = torch.zeros(1, batch_size, 1).to(self.device)

    def forward(self, x, batch_size=None):
        if batch_size is None:
            batch_size = self.mini_batch
        self.reset_hidden(batch_size)
        out, self.hidden = self.cell(x, self.hidden)
        return out

    def predict(self, x):
        eval_batch = x.shape[0]
        with torch.no_grad():
            return self.forward(x, eval_batch)

    def train(self,
              x_test,
              y_test,
              max_seq_len,
              epochs=100,
              batch_size=256,
              batches_per_epoch = 100):

        if self.device == torch.device('cuda'):
            print("Training on: {}".format(torch.cuda.get_device_name(0)))
        else:
            print("Training on CPU.")

        for e in range(epochs):
            print("TRAINING EPOCH {} of {}".format(e + 1, epochs))

            for b in range(batches_per_epoch):
                #print("Mini-batch {} of {}".format(b + 1, batches_per_epoch))
                # Generate training data for this epoch
                x_train, y_train = next(self.datagen(batch_size=batch_size,
                                                    max_len=max_seq_len))

                x_train = x_train.to(self.device)
                y_train = y_train.to(self.device)
                # Clear gradient and hidden state
                self.optimizer.zero_grad()

                # Get predictions
                y_pred = self(x_train)

                # Calculate accuracy
                # train_acc = (y_pred.round() == y_train.round()).float().mean() * 100.

                # Calculate loss, calculate gradient, and update.
                loss = self.loss_function(y_pred, y_train)
                loss.backward()
                self.optimizer.step()

                # print("Training Loss: {:.8f} \t\tAccuracy: {:.4f}%".format(loss, train_acc))
            # Report accuracy on test set at end of each epoch
            with torch.no_grad():
                x_test = x_test.to(self.device)
                y_test = y_test.to(self.device)
                y_pred = self.predict(x_test)
                test_loss = self.loss_function(y_pred, y_test)
            test_acc = (y_pred.round() == y_test.round()).float().mean() * 100.
            print("Test Loss: {:.8f} \t\tAccuracy: {:.4f}%".format(
                test_loss, test_acc))


class AdditionCell(OneCell):

    def __init__(self, device, cell_type='LSTM', mini_batch=64):
        super(AdditionCell, self).__init__(device, cell_type, mini_batch)

        self.datagen = datagen.datagen_add
        cell_params = {'input_size': datagen.BIT_SIZE,
                       'hidden_size': datagen.BIT_SIZE,
                       'num_layers': 1,
                       'bias': True,
                       'batch_first': True,
                       'dropout': 0.0,
                       'bidirectional': False}
        if cell_type == 'LSTM':
            self.cell = nn.LSTM(**cell_params)
        elif cell_type == 'GRU':
            self.cell = nn.GRU(**cell_params)
        else:
            raise ValueError("Valid cell types are LSTM and GRU.")

        # Initialize hidden state
        self.reset_hidden(self.mini_batch)

        # Initialize optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)

    def reset_hidden(self, batch_size):
        if isinstance(self.cell, nn.LSTM):
            self.hidden = (torch.zeros(
                                1,
                                batch_size,
                                datagen.BIT_SIZE
                                ).to(self.device),
                           torch.zeros(
                               1,
                               batch_size,
                               datagen.BIT_SIZE
                               ).to(self.device))
        elif isinstance(self.cell, nn.GRU):
            self.hidden = torch.zeros(
                            1,
                            batch_size,
                            datagen.BIT_SIZE
                            ).to(self.device)
