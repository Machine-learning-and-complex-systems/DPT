# GatedPixelCNN (fix first spin up)

import argparse
import copy
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import log
from tqdm import tqdm


class StackedPixelCNNLayer(nn.Module):
    def __init__(self, mask_type, in_channels, out_channels, k, residual):
        super().__init__()
        self.vertical = nn.Conv2d(in_channels, out_channels, kernel_size=k, padding='same')
        self.horizontal = nn.Conv2d(in_channels, out_channels, kernel_size=(1, k), padding='same')
        self.vtohori = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.htohori = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        self.register_buffer('vmask', self.vertical.weight.data.clone())
        self.register_buffer('hmask', self.horizontal.weight.data.clone())

        self.vmask.fill_(1)
        self.hmask.fill_(1)

        # zero the bottom half rows of the vmask
        self.vmask[:, :, k // 2 + 1 :, :] = 0

        # zero the right half of the hmask
        self.hmask[:, :, :, k // 2 + 1 :] = 0
        if mask_type == 'A':
            self.hmask[:, :, :, k // 2] = 0

        self.residual = residual

    def down_shift(self, x):
        x = x[:, :, :-1, :]
        pad = nn.ZeroPad2d((0, 0, 1, 0))
        return pad(x)

    def forward(self, x):
        # main difference: gated activation -> ReLU
        vi, hi = x.chunk(2, dim=1)

        self.vertical.weight.data *= self.vmask
        self.horizontal.weight.data *= self.hmask

        vo = self.vertical(vi)
        ho = self.horizontal(hi)

        # Allow horizontal stack to see information from vertical stack
        ho = ho + self.vtohori(self.down_shift(vo))

        # apply activation
        vo = F.relu(vo)
        ho = F.relu(ho)
        ho = self.htohori(F.relu(ho))

        if self.residual:
            ho = ho + hi

        return torch.cat((vo, ho), dim=1)


class StackedPixelCNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.Model = kwargs['Model']
        self.L = kwargs['L']
        self.net_depth = kwargs['net_depth']
        self.net_width = kwargs['net_width']
        self.kernel_size = 2 * kwargs['half_kernel_size'] + 1
        self.epsilon = kwargs['epsilon']
        self.device = kwargs['device']
        self.symmetry = kwargs['z2']

        assert self.Model in ['2DFA', 'SE']

        net = [StackedPixelCNNLayer('A', 1, self.net_width, self.kernel_size, residual=False)]
        for _ in range(self.net_depth - 1):
            net.extend(
                [StackedPixelCNNLayer('B', self.net_width, self.net_width, self.kernel_size, residual=True)]
            )
        self.net = nn.Sequential(*net)
        self.final_conv = nn.Conv2d(self.net_width, 1, 1)

        if self.Model == 'SE':
            # For South-or-East Model, force the first x_hat to be 1
            self.register_buffer('x_hat_mask', torch.ones([self.L] * 2))
            self.x_hat_mask[0, 0] = 0
            self.register_buffer('x_hat_bias', torch.zeros([self.L] * 2))
            self.x_hat_bias[0, 0] = 1
        else:
            # For FA Model, we remove [0,...,0] state
            self.register_buffer('x_hat_mask', torch.ones([self.L] * 2))
            self.register_buffer('x_hat_bias', torch.zeros([self.L] * 2))

    def forward(self, x):
        temp = self.net(torch.cat((x, x), dim=1)).chunk(2, dim=1)[1]
        logits = self.final_conv(temp)
        x_hat = torch.sigmoid(logits)
        if self.Model == 'SE':
            # force the first x_hat to be 1
            x_hat = x_hat * self.x_hat_mask + self.x_hat_bias
        else:
            # if all N-1 sites are in down state, we force the last x_hat to be 1
            site_sum = torch.sum(x[:, 0, :-1, :], dim=(1, 2)) + torch.sum(x[:, 0, -1, :-1], dim=-1)
            site_sum = site_sum + self.L**2 - 1
            if torch.count_nonzero(site_sum) != x.shape[0]:
                x_hat_mask = self.x_hat_mask.repeat(x.shape[0], 1, 1, 1)
                x_hat_bias = self.x_hat_bias.repeat(x.shape[0], 1, 1, 1)
                x_hat_mask[(site_sum == 0.0).nonzero(), 0, -1, -1] = 0
                x_hat_bias[(site_sum == 0.0).nonzero(), 0, -1, -1] = 1
                x_hat = x_hat * x_hat_mask + x_hat_bias
        return x_hat

    def sample(self, batch_size):
        """Sample method

        Args:
            batch_size (int): batch size

        Returns:
            Samples from the Gated PixelCNN model
        """
        sample = torch.zeros(batch_size, 1, self.L, self.L, dtype=torch.float, device=self.device)
        for i in range(self.L):
            for j in range(self.L):
                x_hat = self.forward(sample)
                sample[:, :, i, j] = torch.bernoulli(x_hat[:, :, i, j]).to(torch.float) * 2 - 1
        if self.Model == 'SE' and self.symmetry:
            reflection_idx = torch.nonzero(torch.randint(2, size=(batch_size,)), as_tuple=True)
            sample[reflection_idx] = sample[reflection_idx].permute(0, 1, 3, 2)
        elif self.Model == '2DFA' and self.symmetry:
            # https://proofwiki.org/wiki/Definition:Dihedral_Group_D4
            symmetry_group = torch.randint(8, size=(batch_size,))
            sample[symmetry_group == 1] = sample[symmetry_group == 1].rot90(dims=[2, 3])  # a
            sample[symmetry_group == 2] = (
                sample[symmetry_group == 2].rot90(dims=[2, 3]).rot90(dims=[2, 3])
            )  # a^2
            sample[symmetry_group == 3] = (
                sample[symmetry_group == 3].rot90(dims=[2, 3]).rot90(dims=[2, 3]).rot90(dims=[2, 3])
            )  # a^3
            sample[symmetry_group == 4] = sample[symmetry_group == 4].flip(dims=[3])  # b
            sample[symmetry_group == 5] = sample[symmetry_group == 5].rot90(dims=[2, 3]).flip(dims=[3])  # ba
            sample[symmetry_group == 6] = (
                sample[symmetry_group == 6].rot90(dims=[2, 3]).rot90(dims=[2, 3]).flip(dims=[3])
            )  # ba^2
            sample[symmetry_group == 7] = (
                sample[symmetry_group == 7]
                .rot90(dims=[2, 3])
                .rot90(dims=[2, 3])
                .rot90(dims=[2, 3])
                .flip(dims=[3])
            )  # ba^3

        return sample, x_hat

    def _log_prob(self, sample, x_hat):
        mask = (sample + 1) / 2
        log_prob = torch.log(x_hat + self.epsilon) * mask + torch.log(1 - x_hat + self.epsilon) * (1 - mask)
        log_prob = log_prob.view(log_prob.shape[0], -1).sum(dim=1)
        return log_prob

    def log_prob(self, sample):
        """Calculate log probability of configurations

        Args:
            sample (Tensor): shape (batch_size, n)

        Returns:
            log probability of each sample
        """
        x_hat = self.forward(sample)
        log_prob = self._log_prob(sample, x_hat)

        if self.Model == 'SE' and self.symmetry:
            # Invariance under reflection along the diagonal for South-or-East Model
            sample_reflect = sample.permute(0, 1, 3, 2)
            x_hat_reflect = self.forward(sample_reflect)
            log_prob_reflect = self._log_prob(sample_reflect, x_hat_reflect)
            log_prob = torch.logsumexp(torch.stack([log_prob, log_prob_reflect]), dim=0)
            log_prob = log_prob - log(2)
        elif self.Model == '2DFA' and self.symmetry:
            # D4 symmetry for 2D FA Model
            sample_a = sample.rot90(dims=[2, 3])
            x_hat_a = self.forward(sample_a)
            log_prob_a = self._log_prob(sample_a, x_hat_a)

            sample_a2 = sample.rot90(dims=[2, 3]).rot90(dims=[2, 3])
            x_hat_a2 = self.forward(sample_a2)
            log_prob_a2 = self._log_prob(sample_a2, x_hat_a2)

            sample_a3 = sample.rot90(dims=[2, 3]).rot90(dims=[2, 3]).rot90(dims=[2, 3])
            x_hat_a3 = self.forward(sample_a3)
            log_prob_a3 = self._log_prob(sample_a3, x_hat_a3)

            sample_b = sample.flip(dims=[3])
            x_hat_b = self.forward(sample_b)
            log_prob_b = self._log_prob(sample_b, x_hat_b)

            sample_ba = sample.rot90(dims=[2, 3]).flip(dims=[3])
            x_hat_ba = self.forward(sample_ba)
            log_prob_ba = self._log_prob(sample_ba, x_hat_ba)

            sample_ba2 = sample.rot90(dims=[2, 3]).rot90(dims=[2, 3]).flip(dims=[3])
            x_hat_ba2 = self.forward(sample_ba2)
            log_prob_ba2 = self._log_prob(sample_ba2, x_hat_ba2)

            sample_ba3 = sample.rot90(dims=[2, 3]).rot90(dims=[2, 3]).rot90(dims=[2, 3]).flip(dims=[3])
            x_hat_ba3 = self.forward(sample_ba3)
            log_prob_ba3 = self._log_prob(sample_ba3, x_hat_ba3)

            log_prob = torch.logsumexp(
                torch.stack(
                    [
                        log_prob,
                        log_prob_a,
                        log_prob_a2,
                        log_prob_a3,
                        log_prob_b,
                        log_prob_ba,
                        log_prob_ba2,
                        log_prob_ba3,
                    ]
                ),
                dim=0,
            )
            log_prob = log_prob - log(8)

        return log_prob

    def log_psi(self, sample):
        return self.log_prob(sample) / 2
