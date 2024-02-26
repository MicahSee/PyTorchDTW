import numpy as np
import torch

def gen_mask_seq(row, col):
  masks = []

  i=0
  j=0

  for line in range(1, (row + col)):
    start_col = max(0, line - row)
    count = min(min(line, (col - start_col)), row)

    mask = np.zeros((row, col))

    for i in range(count):
      mask[min(row, line) - i - 1][start_col + i] = 1

    masks.append(mask)

  return np.array(masks)

def gen_start_indices(num_patterns, rows_per_pattern):
  rpr = rows_per_pattern

  x_base = torch.Tensor([[0,1,0,1]]).repeat(num_patterns, 1).type(torch.int32)

  y_base = []

  for i in range(num_patterns):
    entry = [rpr*i, rpr*i, rpr*i+1, rpr*i+1]
    y_base.append(entry)

  y_base = torch.Tensor(y_base).type(torch.int32)

  return x_base, y_base