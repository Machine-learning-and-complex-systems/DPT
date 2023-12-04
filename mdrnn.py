# 2D Tensorized RNN Cell described in Variational Neural Annealing
# for TensorizedRNNCell, y = f(x^T W h + b)
# for 2DTensorizedRNNCell, y = f([x1, x2]^T W [h1, h2] + b)

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import default_dtype_torch


class MDTensorizedRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.bilinear = nn.Bilinear(
            self.input_size * 2, self.hidden_size * 2, self.hidden_size, bias=self.bias
        )

    def forward(self, input_h, input_v, hx_h=None, hx_v=None):
        if hx_h is None:
            hx_h = torch.zeros(
                input_h.size(0), self.hidden_size, dtype=default_dtype_torch, device=self.device
            )
        if hx_v is None:
            hx_v = torch.zeros(
                input_v.size(0), self.hidden_size, dtype=default_dtype_torch, device=self.device
            )
        output = torch.tanh(self.bilinear(torch.cat((input_h, input_v), 1), torch.cat((hx_h, hx_v), 1)))
        return output


# 2D Vanilla RNN Cell
# for RNNCell, y = f(W_x x + W_h h + b)
# for 2DRNNCell, y = f(W_1 [x1 h1] + W_2 [x2, h2] + b)
class MDRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, device, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.bias = bias

        self.linear1 = nn.Linear(self.input_size + self.hidden_size, self.hidden_size, bias=self.bias)
        self.linear2 = nn.Linear(self.input_size + self.hidden_size, self.hidden_size, bias=self.bias)

    def forward(self, input_h, input_v, hx_h=None, hx_v=None):
        if hx_h is None:
            hx_h = torch.zeros(
                input_h.size(0), self.hidden_size, dtype=default_dtype_torch, device=self.device
            )
        if hx_v is None:
            hx_v = torch.zeros(
                input_v.size(0), self.hidden_size, dtype=default_dtype_torch, device=self.device
            )
        output = self.linear1(torch.cat((input_h, hx_h), 1)) + self.linear1(torch.cat((input_v, hx_v), 1))
        output = torch.tanh(output)
        return output


# 2D Tensorized RNN without weight sharing for stat mech problems
# Zig-zag order
class RNN2D(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.input_size = 2  # only consider binary input [0, 1]
        self.epsilon = 1e-8
        self.hidden_size = kwargs['net_width']
        self.L = kwargs['L']
        self.device = kwargs['device']
        self.batch_size = kwargs['batch_size']

        # Since we forgo the weight sharing scheme, we use L * L different RNNCells and FC layers
        # self.cell_list = nn.ModuleList([MDTensorizedRNNCell(self.input_size, self.hidden_size) for _ in range(self.L ** 2)])  # use 2D Tensorized RNN
        self.cell_list = nn.ModuleList(
            [MDRNNCell(self.input_size, self.hidden_size, self.device) for _ in range(self.L**2)]
        )  # use 2D vanilla RNN
        self.fc_list = nn.ModuleList([nn.Linear(self.hidden_size, 1) for _ in range(self.L**2)])

    def _forward(self, i, j, x_h, x_v, h_h=None, h_v=None):
        x_h = torch.stack([x_h, 1 - x_h], dim=1)  # 1 -> (1, 0), 0 -> (0, 1)
        x_v = torch.stack([x_v, 1 - x_v], dim=1)  # 1 -> (1, 0), 0 -> (0, 1)
        h_p = self.cell_list[i * self.L + j](x_h, x_v, h_h, h_v)  # h_{i,j}
        x_hat = torch.sigmoid(self.fc_list[i * self.L + j](h_p)).squeeze(1)  # x_hat = p(s_{i+1,j} = 1 | ...)

        return h_p, x_hat

    def log_prob(self, samples):
        """Calculate log probability of configurations

        Args:
            sample (Tensor): shape (batch_size, L, L)

        Returns:
            log probability of each sample
        """
        mask = samples
        log_prob = torch.zeros_like(samples)
        x_0 = torch.zeros(samples.size(0), dtype=default_dtype_torch, device=self.device)
        h_v_list = []  # stores all the vertical hidden state above

        # top boundary, i=0, from left to right
        for j in range(self.L):
            if j == 0:
                h, x_hat = self._forward(0, j, x_0, x_0, h_h=None, h_v=None)
            else:
                h, x_hat = self._forward(0, j, samples[:, 0, j - 1], x_0, h_h=h, h_v=None)
            h_v_list.append(h)
            log_prob[:, 0, j] = torch.log(x_hat + self.epsilon) * mask[:, 0, j] + torch.log(
                1 - x_hat + self.epsilon
            ) * (1.0 - mask[:, 0, j])

        for i in range(1, self.L):
            # from left to right
            if i % 2 == 0:
                for j in range(self.L):
                    if j == 0:
                        h, x_hat = self._forward(i, j, x_0, samples[:, i - 1, j], h_h=None, h_v=h_v_list[j])
                    else:
                        h, x_hat = self._forward(
                            i, j, samples[:, i, j - 1], samples[:, i - 1, j], h_h=h, h_v=h_v_list[j]
                        )
                    h_v_list[j] = h
                    log_prob[:, i, j] = torch.log(x_hat + self.epsilon) * mask[:, i, j] + torch.log(
                        1 - x_hat + self.epsilon
                    ) * (1.0 - mask[:, i, j])
            # from right to left
            elif i % 2 == 1:
                for j in range(self.L)[::-1]:
                    if j == self.L - 1:
                        h, x_hat = self._forward(i, j, x_0, samples[:, i - 1, j], h_h=None, h_v=h_v_list[j])
                    else:
                        h, x_hat = self._forward(
                            i, j, samples[:, i, j + 1], samples[:, i - 1, j], h_h=h, h_v=h_v_list[j]
                        )
                    h_v_list[j] = h
                    log_prob[:, i, j] = torch.log(x_hat + self.epsilon) * mask[:, i, j] + torch.log(
                        1 - x_hat + self.epsilon
                    ) * (1.0 - mask[:, i, j])

        return log_prob.sum(dim=(1, 2))

    def sample(self, batch_size):
        """Sample method

        Args:
            batch_size (int): batch size

        Returns:
            Samples from the MDRNN model
        """
        samples = torch.zeros(batch_size, self.L, self.L, dtype=default_dtype_torch, device=self.device)
        x_0 = torch.zeros(batch_size, dtype=default_dtype_torch, device=self.device)
        h_v_list = []  # stores all the vertical hidden state above

        # top boundary, i=0
        for j in range(self.L):
            if j == 0:
                h, x_hat = self._forward(0, j, x_0, x_0, h_h=None, h_v=None)
            else:
                h, x_hat = self._forward(0, j, samples[:, 0, j - 1], x_0, h_h=h, h_v=None)
            h_v_list.append(h)
            samples[:, 0, j] = torch.bernoulli(x_hat)

        for i in range(1, self.L):
            # from left to right
            if i % 2 == 0:
                for j in range(self.L):
                    if j == 0:
                        h, x_hat = self._forward(i, j, x_0, samples[:, i - 1, j], h_h=None, h_v=h_v_list[j])
                    else:
                        h, x_hat = self._forward(
                            i, j, samples[:, i, j - 1], samples[:, i - 1, j], h_h=h, h_v=h_v_list[j]
                        )
                    h_v_list[j] = h
                    samples[:, i, j] = torch.bernoulli(x_hat)
            # from right to left
            elif i % 2 == 1:
                for j in range(self.L)[::-1]:
                    if j == self.L - 1:
                        h, x_hat = self._forward(i, j, x_0, samples[:, i - 1, j], h_h=None, h_v=h_v_list[j])
                    else:
                        h, x_hat = self._forward(
                            i, j, samples[:, i, j + 1], samples[:, i - 1, j], h_h=h, h_v=h_v_list[j]
                        )
                    h_v_list[j] = h
                    samples[:, i, j] = torch.bernoulli(x_hat)

        x_hat = torch.zeros_like(samples)

        return samples, x_hat
