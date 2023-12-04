# GRU for VAN

import numpy as np
import torch
import torch.nn.functional as F
from numpy import log
from torch import nn

from utils import default_dtype_torch


class GRU(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.L = kwargs['L']
        self.n = self.L
        self.net_depth = kwargs['net_depth']
        self.net_width = kwargs['net_width']
        self.bias = kwargs['bias']
        self.z2 = kwargs['z2']
        self.res_block = kwargs['res_block']
        self.x_hat_clip = kwargs['x_hat_clip']
        self.epsilon = kwargs['epsilon']
        self.device = kwargs['device']

        self.order = list(range(self.n))

        # model parameters
        self.rnn = nn.GRUCell(2, self.net_width)
        self.fc1 = nn.Linear(self.net_width, 2)

    def _forward(self, x, h):
        embedded_x = torch.stack([(x + 1) / 2, 1.0 - (x + 1) / 2], dim=1)  # (batch_size, 2)
        h_next = self.rnn(embedded_x, h)
        y = F.log_softmax(self.fc1(h_next), dim=1)
        return h_next, y

    def log_prob(self, x):
        """Calculate log probability of configurations

        Args:
            x (Tensor): shape (batch_size, n)

        Returns:
            log probability of each sample
        """
        batch_size = x.shape[0]
        log_prob = torch.zeros_like(x)
        mask = (1 + x) / 2  # mask is to check up or down spin-by-spin for each sample
        x_init = torch.zeros(batch_size, dtype=default_dtype_torch, device=self.device)
        h_init = torch.zeros(batch_size, self.net_width, dtype=default_dtype_torch, device=self.device)
        h, y = self._forward(x_init, h_init)

        log_prob[:, 0] = y[:, 0] * mask[:, 0] + y[:, 1] * (1.0 - mask[:, 0])

        for i in range(1, self.n):
            h, y = self._forward(x[:, i - 1], h)
            log_prob[:, i] = y[:, 0] * mask[:, i] + y[:, 1] * (1.0 - mask[:, i])

        return log_prob.sum(dim=1)

    def sample(self, batch_size):
        """Sample method

        Args:
            batch_size (int): batch size

        Returns:
            Samples from the GRU model

        """
        samples = torch.zeros([batch_size, self.n], dtype=default_dtype_torch, device=self.device)

        x_init = torch.zeros(batch_size, dtype=default_dtype_torch, device=self.device)
        h_init = torch.zeros(batch_size, self.net_width, dtype=default_dtype_torch, device=self.device)
        h, y = self._forward(x_init, h_init)
        p = torch.exp(y)[:, 0]
        samples[:, 0] = torch.bernoulli(p).to(default_dtype_torch) * 2 - 1

        for i in range(1, self.n):
            h, y = self._forward(samples[:, i - 1], h)
            p = torch.exp(y)[:, 0]
            samples[:, i] = torch.bernoulli(p).to(default_dtype_torch) * 2 - 1

        x_hat = torch.zeros_like(samples)

        return samples, x_hat
