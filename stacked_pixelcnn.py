# Gated PixenCNN

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import log

from utils import default_dtype_torch


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
        self.L = kwargs['L']
        self.net_depth = kwargs['net_depth']
        self.net_width = kwargs['net_width']
        self.kernel_size = 2 * kwargs['half_kernel_size'] + 1
        self.epsilon = kwargs['epsilon']
        self.device = kwargs['device']

        # For South-or-East model, force the first x_hat to be 1
        self.register_buffer('x_hat_mask', torch.ones([self.L] * 2))
        self.x_hat_mask[0, 0] = 0
        self.register_buffer('x_hat_bias', torch.zeros([self.L] * 2))
        self.x_hat_bias[0, 0] = 1

        model = [StackedPixelCNNLayer('A', 1, self.net_width, self.kernel_size, residual=False)]
        for _ in range(self.net_depth - 1):
            model.extend(
                [StackedPixelCNNLayer('B', self.net_width, self.net_width, self.kernel_size, residual=True)]
            )
        self.net = nn.Sequential(*model)
        self.final_conv = nn.Conv2d(self.net_width, 1, 1)

    def forward(self, x):
        temp = self.net(torch.cat((x, x), dim=1)).chunk(2, dim=1)[1]
        logits = self.final_conv(temp)
        x_hat = torch.sigmoid(logits)
        x_hat = x_hat * self.x_hat_mask + self.x_hat_bias
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
        reflection_idx = torch.nonzero(torch.randint(2, size=(batch_size,)), as_tuple=True)
        sample[reflection_idx] = sample[reflection_idx].permute(0, 1, 3, 2)

        # all-0 state: begin
        mask = (sample + 1) / 2
        aa = torch.sum(torch.sum(torch.sum(mask, 3), 2), 1)
        Record = aa == 0  # find all-0 configuration
        # For the all-0 state, change the last spin to up: its probabilty has been changed accordingly in _log_prob
        sample[Record, 0, self.L - 1, self.L - 1] = 1
        # all-0 state: end

        return sample, x_hat

    def _log_prob(self, sample, x_hat):
        mask = (sample + 1) / 2
        log_prob = torch.log(x_hat + self.epsilon) * mask + torch.log(1 - x_hat + self.epsilon) * (1 - mask)

        # all-0 state: begin
        aa = torch.sum(torch.sum(torch.sum(mask, 3), 2), 1)
        Record = aa == 1  # find only-one-spin-up configuration
        LastUp = mask[Record, 0, self.L - 1, self.L - 1]  # last spin of the only-one-spin-up configuration
        log_prob[Record, 0, self.L - 1, self.L - 1] = log_prob[Record, 0, self.L - 1, self.L - 1] * (
            1 - LastUp
        )  # If last spin is up, change the log_prob to 0
        # all-0 state: end

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

        # Invariance under reflection along the diagonal for South-or-East model
        sample_reflect = sample.permute(0, 1, 3, 2)
        x_hat_reflect = self.forward(sample_reflect)
        log_prob_reflect = self._log_prob(sample_reflect, x_hat_reflect)
        log_prob = torch.logsumexp(torch.stack([log_prob, log_prob_reflect]), dim=0)
        log_prob = log_prob - log(2)

        return log_prob

    def log_psi(self, sample):
        return self.log_prob(sample) / 2
