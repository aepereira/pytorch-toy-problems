# pytorch-toy-problems

I have been using Keras a lot for work, but I decided to switch to PyTorch now that the stable release is out.
This repo is basically a self-tutorial I made for myself to train LSTM and GRU recurrent neural nets to learn
some very easy toy problems in PyTorch.

## XOR

This network is the easiest problem for a recurrent neural network. It uses a single hidden unit. The network
trains itself to act as a XOR gate (checking bit parity) on bit strings of different lengths. It can then
generalize to arbitrary sequence lengths. While this is a simple problem, it showcases the power of recurrent
networks, because a feed-forward neural network cannot adapt to arbitrary sequence lengths.

## Addition

This network is similar to a larger version of the XOR problem. But instead of learning to count parity,
the network learns to add an arbitrary sequence of integers (represented as 32-bit strings). This network
has one hidden state per bit.
