import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dtw_utilities import gen_mask_seq, gen_start_indices

class DTW(nn.Module):
    def __init__(self, device='cpu'):
        super(DTW, self).__init__()
        self.kernel_size = 2
        self.pool1 = nn.MaxPool2d(kernel_size=self.kernel_size, stride=1, padding=0)
        self.device = device

    def min_pool(self, x):
        x_fmt = x.unsqueeze(0)
        res = -self.pool1(-x_fmt)
        return res.view(x.shape[0], x.shape[1]-1, x.shape[2]-1)

    def forward(self, x, y):
        cost_mats = self.__calc_cost_mats(x, y)
        acc_costs = self.__acc_costs(cost_mats)

        return acc_costs

    def __acc_costs(self, cost_mats):
        pad_cost_mats = F.pad(cost_mats, (0, 1, 0, 1, 0, 0), 'constant', torch.inf)
        cmr = pad_cost_mats.reshape(-1, pad_cost_mats.shape[2])

        start_indices = gen_start_indices(pad_cost_mats.shape[0], pad_cost_mats.shape[1])

        max_iter = cost_mats.shape[1] + cost_mats.shape[2] - 1
        cost_vec = cost_mats[:, 0, 0]

        x_indices = start_indices[0].to(self.device)
        y_indices = start_indices[1].to(self.device)

        for i in range(max_iter):
            sel = cmr[y_indices, x_indices]
            sel[:, 0] = torch.inf

            # enforce priority of operations: diagonal, down, right (for edge case when multiple adjacent entries have the same value)
            sel_rev = torch.flip(sel, dims=[1])
            values, indices = torch.min(sel_rev, dim=1)
            last_indices = 3 - indices
            ind_mod_loc = torch.where(values==torch.inf)[0]
            last_indices[ind_mod_loc] = 0

            values[values==torch.inf] = 0

            x_adj = last_indices.reshape(-1, 1).repeat(1, 4) % 2
            y_adj = last_indices.reshape(-1, 1).repeat(1, 4) // 2
            x_indices = x_indices + x_adj
            y_indices = y_indices + y_adj

            cost_vec += values

        return cost_vec

    def __calc_cost_mats(self, x, y):
        # input x should be a single sequence
        # input y can be a 2D tensor, a list of sequences
        x_len = x.shape[-1]
        y_len = y.shape[-1]

        n_patterns = y.shape[0]

        # swap to put the longer sequence on the x-axis of the matrix
        if (y_len < x_len):
            y_shorter = True
            mask_dim1, mask_dim2 = y_len, x_len
        else:
            y_shorter = False
            mask_dim1, mask_dim2 = x_len, y_len

        masks = gen_mask_seq(mask_dim1, mask_dim2)
        masks = torch.tensor(masks).to(self.device)

        inf_masks = masks.clone()
        inf_masks[inf_masks==1] = torch.inf
        inf_masks = F.pad(inf_masks, (1,0,1,0), 'constant', 0)

        # x is mapped along x-axis of cost matrix
        x_mat_ = torch.repeat_interleave(x, y_len, dim=0)
        x_mat = torch.repeat_interleave(x_mat_.unsqueeze(0), n_patterns, dim=0)

        if not y_shorter:
            x_mat = x_mat.permute(0, 2, 1)

        # print(x_mat)

        # y is mapped along y-axis of cost matrix (starting in upper-left corner)
        y_mat = torch.repeat_interleave(y.unsqueeze(1), x_len, dim=1)

        if y_shorter:
            y_mat = y_mat.permute(0, 2, 1)

        # fill in the cost matrix
        cost_mat_ = (x_mat - y_mat)**2

        cost_mat = F.pad(cost_mat_, (1,0,1,0,0,0), 'constant', 0)

        # pad with inf
        cost_mat[:, 0, 1:] = torch.inf
        cost_mat[:, 1:, 0] = torch.inf

        inner_cost_mat = cost_mat[:, 1:, 1:]

        for i in range(masks.shape[0]):
            mask_mat_ = masks[i]
            mask_mat = torch.repeat_interleave(mask_mat_.unsqueeze(0), n_patterns, dim=0)

            inf_mask_mat_ = inf_masks[i]
            inf_mask_mat = torch.repeat_interleave(inf_mask_mat_.unsqueeze(0), n_patterns, dim=0)

            inf_mask_cost_mat = cost_mat + inf_mask_mat

            pooled_res = self.min_pool(inf_mask_cost_mat)

            inner_cost_mat += (mask_mat * pooled_res)

        return inner_cost_mat