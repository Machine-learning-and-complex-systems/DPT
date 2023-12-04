# MADE: Masked Autoencoder for Distribution Estimation (for 1D systems)

import torch
from numpy import log
from torch import nn

from pixelcnn import ResBlock
from utils import default_dtype_torch


class MaskedLinear(nn.Linear):
    def __init__(self, in_channels, out_channels, n, bias, exclusive):
        super(MaskedLinear, self).__init__(in_channels * n, out_channels * n, bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n = n
        self.exclusive = exclusive
        self.register_buffer('mask', torch.ones([self.n] * 2))  # My code

        if self.exclusive:
            self.mask = 1 - torch.triu(self.mask)
            # lower triangle without diagonal
        else:
            self.mask = torch.tril(self.mask)
            # lower triangle with diagonal

        self.mask = torch.cat([self.mask] * in_channels, dim=1)
        self.mask = torch.cat([self.mask] * out_channels, dim=0)
        self.weight.data *= self.mask

        # Correction to Xavier initialization
        self.weight.data *= torch.sqrt(self.mask.numel() / self.mask.sum())

    def forward(self, x):
        return nn.functional.linear(x, self.mask * self.weight, self.bias)

    def extra_repr(self):
        return super(MaskedLinear, self).extra_repr() + ', exclusive={exclusive}'.format(**self.__dict__)


# TODO: reduce unused weights, maybe when torch.sparse is stable
class ChannelLinear(nn.Linear):
    def __init__(self, in_channels, out_channels, n, bias):
        super(ChannelLinear, self).__init__(in_channels * n, out_channels * n, bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n = n

        self.register_buffer('mask', torch.eye(self.n))
        self.mask = torch.cat([self.mask] * in_channels, dim=1)
        self.mask = torch.cat([self.mask] * out_channels, dim=0)
        self.weight.data *= self.mask

        # Correction to Xavier initialization
        self.weight.data *= torch.sqrt(self.mask.numel() / self.mask.sum())

    def forward(self, x):
        return nn.functional.linear(x, self.mask * self.weight, self.bias)


class MADE1D(nn.Module):
    def __init__(self, **kwargs):
        super(MADE1D, self).__init__()
        self.L = kwargs['L']
        self.n = self.L  # **2  # My code: Number of sites
        self.net_depth = kwargs['net_depth']
        self.net_width = kwargs['net_width']
        self.bias = kwargs['bias']
        self.z2 = kwargs['z2']
        self.res_block = kwargs['res_block']
        self.x_hat_clip = kwargs['x_hat_clip']
        self.epsilon = kwargs['epsilon']
        self.device = kwargs['device']

        # Force the first x_hat to be 0.5
        if self.bias and not self.z2:
            self.register_buffer('x_hat_mask', torch.ones([self.L]))  # * 2)) # My code:
            self.x_hat_mask[0] = 0  # My code
            self.register_buffer('x_hat_bias', torch.zeros([self.L]))  # * 2))#My code
            self.x_hat_bias[0] = 0.5  # My code

        layers = []
        layers.append(
            MaskedLinear(1, 1 if self.net_depth == 1 else self.net_width, self.n, self.bias, exclusive=True)
        )

        for count in range(self.net_depth - 2):
            if self.res_block:  # We didn't use res_block so far
                layers.append(self._build_res_block(self.net_width, self.net_width))
            else:
                layers.append(self._build_simple_block(self.net_width, self.net_width))
        if self.net_depth >= 2:
            layers.append(self._build_simple_block(self.net_width, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def _build_simple_block(self, in_channels, out_channels):
        layers = []
        layers.append(nn.PReLU(in_channels * self.n, init=0.5))
        layers.append(MaskedLinear(in_channels, out_channels, self.n, self.bias, exclusive=False))
        block = nn.Sequential(*layers)
        return block

    def _build_res_block(self, in_channels, out_channels):
        layers = []
        layers.append(ChannelLinear(in_channels, out_channels, self.n, self.bias))
        layers.append(nn.PReLU(in_channels * self.n, init=0.5))
        layers.append(MaskedLinear(in_channels, out_channels, self.n, self.bias, exclusive=False))
        block = ResBlock(nn.Sequential(*layers))
        return block

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        x_hat = self.net(x)  # Use NN to get a probability of the configuration: [Batchsize,SpinNum]
        x_hat = x_hat.view(x_hat.shape[0], 1, self.L)  # My code, self.L)
        if self.x_hat_clip:
            with torch.no_grad():
                delta_x_hat = torch.clamp(x_hat, self.x_hat_clip, 1 - self.x_hat_clip) - x_hat
            assert not delta_x_hat.requires_grad
            x_hat = x_hat + delta_x_hat

        return x_hat

    def sample(self, batch_size):
        """Sample method

        Args:
            batch_size (int): batch size

        Returns:
            Samples from the MADE model
        """
        sample = torch.zeros(
            [batch_size, 1, self.L], dtype=default_dtype_torch, device=self.device  # My code, self.L],
        )
        for i in range(self.L):
            x_hat = self.forward(sample)

            sample[:, :, i] = torch.bernoulli(x_hat[:, :, i]).to(default_dtype_torch) * 2 - 1  # My code
            # sample is to randomly generate 1, -1 values to get the configuration for the probability x_hat
        if self.z2:
            flip = (
                torch.randint(2, [batch_size, 1, 1], dtype=sample.dtype, device=sample.device) * 2 - 1
            )  # My code
            sample *= flip

        return sample, x_hat

    def _log_prob(self, sample, x_hat):
        mask = (sample + 1) / 2  # Spin state of the samples in 1, 0: [Batchsize,1,SpinNum]
        # x_hat is the conditional probability for the current spin up: [Batchsize,1,SpinNum]
        log_prob = torch.log(x_hat + self.epsilon) * mask + torch.log(1 - x_hat + self.epsilon) * (1 - mask)
        # Check the probability data dimension: [Batchsize,1,SpinNum]
        log_prob = log_prob.view(log_prob.shape[0], -1).sum(
            dim=1
        )  # It seems to sum up all the log(conditional P)
        # Check the probability data dimension: [Batchsize]
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

        if self.z2:
            # Density estimation on inverted sample
            sample_inv = -sample
            x_hat_inv = self.forward(sample_inv)
            log_prob_inv = self._log_prob(sample_inv, x_hat_inv)
            log_prob = torch.logsumexp(torch.stack([log_prob, log_prob_inv]), dim=0)
            log_prob = log_prob - log(2)

        return log_prob

    def _log_prob2(self, sample, x_hat):
        mask = (sample + 1) / 2
        log_prob = torch.log(x_hat + self.epsilon) * mask + torch.log(1 - x_hat + self.epsilon) * (1 - mask)
        log_prob = log_prob.view(log_prob.shape[0], log_prob.shape[1], -1).sum(dim=2)
        return log_prob

    def log_prob2(self, sample):
        x_hat = self.forward2(sample)
        log_prob = self._log_prob2(sample, x_hat)
        return log_prob

    def forward2(self, x):
        xx = torch.reshape(x, (x.shape[0] * x.shape[1], self.L))  # x.view(x.shape[0]*x.shape[1], -1)
        x_hat = self.net(xx)
        x_hat = x_hat.view(x.shape[0], x.shape[1], self.L)  # My code
        return x_hat
