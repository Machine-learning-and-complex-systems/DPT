# MADE: Masked Autoencoder for Distribution Estimation (for 3D systems)

import torch
from numpy import log
from torch import nn

from pixelcnn import ResBlock
from utils import default_dtype_torch

torch.set_printoptions(precision=20)


class MaskedLinear(nn.Linear):
    def __init__(self, in_channels, out_channels, n, bias, exclusive):
        super(MaskedLinear, self).__init__(in_channels * n, out_channels * n, bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n = n
        self.exclusive = exclusive

        self.register_buffer('mask', torch.ones([self.n] * 2))
        if self.exclusive:
            self.mask = 1 - torch.triu(self.mask)
        else:
            self.mask = torch.tril(self.mask)
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


class MADE3D(nn.Module):
    def __init__(self, **kwargs):
        super(MADE3D, self).__init__()
        self.L = kwargs['L']
        self.size = kwargs['size']
        self.n = self.L**3  # Number of sites
        self.net_depth = kwargs['net_depth']
        self.net_width = kwargs['net_width']
        self.bias = kwargs['bias']
        self.z2 = kwargs['z2']
        self.res_block = kwargs['res_block']
        self.x_hat_clip = kwargs['x_hat_clip']
        self.epsilon = kwargs['epsilon']
        self.device = kwargs['device']
        self.binomialP = torch.tensor(kwargs['binomialP'], dtype=torch.float64).to(self.device)

        # Force the first x_hat to be 0.5
        if self.bias and not self.z2:
            self.register_buffer('x_hat_mask', torch.ones([self.L] * 2))
            self.x_hat_mask[0, 0] = 0
            self.register_buffer('x_hat_bias', torch.zeros([self.L] * 2))
            self.x_hat_bias[0, 0] = 0.5

        layers = []
        layers.append(
            MaskedLinear(1, 1 if self.net_depth == 1 else self.net_width, self.n, self.bias, exclusive=True)
        )
        for count in range(self.net_depth - 2):
            if self.res_block:
                layers.append(self._build_res_block(self.net_width, self.net_width))
            else:
                layers.append(self._build_simple_block(self.net_width, self.net_width))
        if self.net_depth >= 2:
            layers.append(self._build_simple_block(self.net_width, 1))
        layers.append(nn.Sigmoid())  # The previous
        self.net = nn.Sequential(*layers)

    def _build_simple_block(self, in_channels, out_channels):
        layers = []
        layers.append(nn.PReLU(in_channels * self.n, init=0.5))  # The previous
        # layers.append(nn.ReLU()) # My code
        layers.append(MaskedLinear(in_channels, out_channels, self.n, self.bias, exclusive=False))
        block = nn.Sequential(*layers)
        return block

    def _build_res_block(self, in_channels, out_channels):
        layers = []
        layers.append(ChannelLinear(in_channels, out_channels, self.n, self.bias))
        layers.append(nn.PReLU(in_channels * self.n, init=0.5))  # The previous
        # layers.append(nn.ReLU()) # My code
        layers.append(MaskedLinear(in_channels, out_channels, self.n, self.bias, exclusive=False))
        block = ResBlock(nn.Sequential(*layers))
        return block

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        x_hat = self.net(x)
        x_hat = x_hat.view(x_hat.shape[0], 1, self.L, self.L, self.L)

        if self.x_hat_clip:
            # Clip value and preserve gradient
            with torch.no_grad():
                delta_x_hat = torch.clamp(x_hat, self.x_hat_clip, 1 - self.x_hat_clip) - x_hat
            assert not delta_x_hat.requires_grad
            x_hat = x_hat + delta_x_hat

        return x_hat

    def sampleIS(self, batch_size):
        """Sample method using importance sampling

        Args:
            batch_size (int): batch size

        Returns:
            Samples from the MADE model
        """
        sample = torch.zeros(
            [batch_size, 1, self.L, self.L, self.L], dtype=default_dtype_torch, device=self.device
        )
        # binomialP=self.binomialP*(1+torch.rand(sample.shape, dtype=torch.float64).to(self.device))
        binomialP = self.binomialP * torch.ones(sample.shape, dtype=torch.float64).to(self.device)
        x_hat = []
        for i in range(self.L):
            for j in range(self.L):
                for k in range(self.L):
                    if i == 0 and j == 0 and k == 0:
                        sample[:, :, i, j, k] = torch.ones(
                            [batch_size, 1], dtype=default_dtype_torch, device=self.device
                        )  # Manual set the first up-spin
                    else:
                        sample[:, :, i, j, k] = (
                            torch.bernoulli(binomialP[:, :, i, j, k]).to(default_dtype_torch) * 2 - 1
                        )

        return sample, x_hat

    def sample(self, batch_size):
        """Sample method

        Args:
            batch_size (int): batch size

        Returns:
            Samples from the MADE model
        """
        sample = torch.zeros(
            [batch_size, 1, self.L, self.L, self.L], dtype=default_dtype_torch, device=self.device
        )

        for i in range(self.L):
            for j in range(self.L):
                for k in range(self.L):
                    x_hat = self.forward(sample)
                    if i == 0 and j == 0 and k == 0:
                        sample[:, :, i, j, k] = torch.ones(
                            [batch_size, 1], dtype=default_dtype_torch, device=self.device
                        )  # Manual set the first up-spin
                    else:
                        sample[:, :, i, j, k] = (
                            torch.bernoulli(x_hat[:, :, i, j, k]).to(default_dtype_torch) * 2 - 1
                        )

        if self.z2:
            flip = torch.randint(2, [batch_size, 1, 1, 1], dtype=sample.dtype, device=sample.device) * 2 - 1
            sample *= flip
            rotate = torch.randint(
                2, [batch_size, 1, 1, 1], dtype=sample.dtype, device=sample.device
            )  # 1 is rotate, 0 is not
            sample = torch.rot90(sample, 2, [2, 3]) * rotate + sample * (1 - rotate)

        return sample, x_hat

    def _log_prob(self, sample, x_hat):
        mask = (sample + 1) / 2
        log_prob = torch.log(x_hat + self.epsilon) * mask + torch.log(1 - x_hat + self.epsilon) * (1 - mask)

        log_prob[:, 0, 0, 0, 0] = 1 - mask[:, 0, 0, 0, 0]  # Manual set the first up-spin has logp=0

        log_prob = log_prob.view(log_prob.shape[0], -1).sum(dim=1)
        return log_prob

    def log_prob(self, sample):
        """Calculate log probability of configurations

        Args:
            sample (Tensor): shape (batch_size, L, L, L)

        Returns:
            log probability of each sample
        """
        x_hat = self.forward(sample)
        log_prob = self._log_prob(sample, x_hat)

        if self.z2:
            x_flip = -sample
            x_rotate = torch.rot90(sample, 2, [2, 3])
            x_flip_rotate = torch.rot90(x_flip, 2, [2, 3])
            x_hat_inv2 = self.forward(x_flip)
            log_prob2 = self._log_prob(x_flip, x_hat_inv2)
            x_hat_inv3 = self.forward(x_rotate)
            log_prob3 = self._log_prob(x_rotate, x_hat_inv3)
            x_hat_inv4 = self.forward(x_flip_rotate)
            log_prob4 = self._log_prob(x_flip_rotate, x_hat_inv4)
            log_prob = torch.logsumexp(torch.stack([log_prob, log_prob2, log_prob3, log_prob4]), dim=0)
            log_prob = log_prob - log(4)

        return log_prob

    def log_probIS(self, sample):
        """Calculate log probability of configurations by importance sampling

        Args:
            sample (Tensor): shape (batch_size, L, L, L)

        Returns:
            log probability of each sample
        """
        mask = (sample + 1) / 2
        aa = torch.sum(torch.sum(torch.sum(torch.sum(mask, 4), 3), 2), 1) - 1  # The first spin is fixed up
        log_probIS = (
            aa * torch.log(self.binomialP) + (self.size - 1 - aa) * torch.log(1 - self.binomialP)
        ).to(default_dtype_torch)

        return log_probIS

    def _log_prob2(self, sample, x_hat):
        mask = (sample + 1) / 2
        mask = mask.view(mask.shape[0], mask.shape[1], self.L, self.L)  # My code
        log_prob = torch.log(x_hat + self.epsilon) * mask + torch.log(1 - x_hat + self.epsilon) * (1 - mask)
        log_prob = log_prob.view(log_prob.shape[0], log_prob.shape[1], -1).sum(dim=2)
        return log_prob

    def log_prob2(self, sample):
        x_hat = self.forward2(sample)
        log_prob = self._log_prob2(sample, x_hat)
        return log_prob

    def forward2(self, x):
        xx = torch.reshape(x, (x.shape[0] * x.shape[1], self.L * self.L))  # x.view(x.shape[0]*x.shape[1], -1)
        x_hat = self.net(xx)
        x_hat = x_hat.view(x.shape[0], x.shape[1], self.L, self.L)  # My code
        return x_hat
