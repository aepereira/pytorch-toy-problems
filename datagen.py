"""
Generators for XOR and
addition data.

Batches of data are genrated randomly
and their labels computed automatically.

Author: Arnaldo E. Pereira
"""

import numpy as np
import torch.tensor


BIT_SIZE = 32


def datagen_xor(batch_size=64, max_len=100):
    lengths = np.linspace(1, max_len, num=max_len, dtype=int)
    sequences = []
    for _ in range(batch_size):
        length = np.random.choice(lengths)
        padding = np.zeros(max_len - length)
        binary_num = np.random.choice([0, 1], size=length, replace=True)
        binary_num = np.append(binary_num, padding)
        sequences.append(binary_num)
    x = torch.tensor(sequences).view([batch_size, max_len, 1]).float()
    y = torch.tensor([np.cumsum(binary_num) % 2
                      for binary_num in sequences]).float().view([batch_size, max_len, 1])
    yield x, y


def to_decimal(b):
    if not isinstance(b, np.ndarray):
        b = np.array(b)
    d = b[:-1].dot(1 << np.arange(BIT_SIZE - 1)[::-1])
    if b[-1] == 1:
        d *= -1
    return d


def to_binary(d):
    sign = [0]
    if d < 0:
        sign = [1]
        d *= -1
    b = np.array(list(np.binary_repr(int(d))), dtype=int)
    b = np.concatenate([b, sign])
    if b.size < BIT_SIZE:
        padding = [0] * (BIT_SIZE - b.size)
        b = np.concatenate([padding, b])
    return b.tolist()


def datagen_add(batch_size=64, max_len=10):
    """Handles 32 bit numbers. Rightmost digit is the sign (0 for
    positive, 1 for negative). Zero-padded to the left.
    """
    # Possible lengths of sequence
    lengths = np.linspace(1, max_len, num=max_len, dtype=int)
    bit_size = BIT_SIZE
    # Always choose at least the sign bit and one exponent bit.
    bit_lens = np.linspace(2, bit_size, num=bit_size - 1, dtype=int)

    sequences = []
    labels = []
    for _ in range(batch_size):
        seq_len = np.random.choice(lengths)
        binaries = []
        decimals = []
        for _ in range(seq_len):
            bit_len = np.random.choice(bit_lens)
            padding = [0] * (bit_size - bit_len)
            b = np.random.choice([0, 1], size=bit_len, replace=True).tolist()
            # Handle special case of -0
            if np.sum(b[:-1]) == 0:
                b[-1] = 0
            b = padding + b
            binaries.append(b)
            decimals.append(to_decimal(b))
        # Labels (i.e., binary form) for the partial sums within this sequence
        sub_labels = [to_binary(np.cumsum(d))
                      for d in decimals]

        # Handle sequence padding
        pad_seqs = [[0] * bit_size] * (max_len - seq_len)
        pad_labels = pad_seqs
        binaries += pad_seqs
        sub_labels += pad_labels

        # Append to the master lists
        labels.append(sub_labels)
        sequences.append(binaries)

    x = torch.tensor(sequences).float()
    y = torch.tensor(labels).float()

    yield x, y
