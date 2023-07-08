"""Models from "Neural Lyapunov Control"

https://papers.nips.cc/paper/2019/file/2647c1dba23bc0e0f9cdf75339e120d2-Paper.pdf
"""

import os
import torch
from torch import nn


# class TanhGrad(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x):
#         return 1 - torch.tanh(x)**2


class InvertedPendulum(nn.Module):
    def __init__(self):
        super().__init__()
        self.V = nn.Sequential(
            nn.Linear(2, 6),
            nn.Tanh(),
            nn.Linear(6, 1),
            nn.Tanh(),
        )
        self.u = nn.Linear(2, 1, bias=False)

    def forward(self, x):
        """Compute the forward output and the Jacobian matrix.

        Must be a feedforward network for now!
        """
        inputs = [x]
        assert isinstance(self.V, nn.Sequential)
        layers = list(self.V._modules.values())
        for layer in layers:
            inputs.append(layer(inputs[-1]))
        V = inputs[-1]
        u = self.u(x)
        grad = torch.ones_like(inputs[-1]).to(x)
        for i in range(len(layers) - 1, -1, -1):
            layer = layers[i]
            if isinstance(layer, nn.Tanh):
                grad = grad * (1 - torch.tanh(inputs[i]) * torch.tanh(inputs[i]))
            elif isinstance(layer, nn.Linear):
                grad = grad.matmul(layer.weight)
            else:
                raise NotImplementedError(layer)
        theta = x[:, 0:1]
        dtheta = x[:, 1:2]
        g = 0.81
        m = 0.15
        l = 0.5
        dynamics = torch.concat([
            dtheta,
            (m * g * l * torch.sin(theta) + u
             - 0.1 * dtheta) / (m * l * l)
        ], dim=-1)
        # Verify: V > 0 and lie_derivative < 0
        tolerance = 0.01 # A tolerance value is required as used in dreal
        return torch.concat([
            V,
            -(grad * dynamics).sum(dim=-1, keepdim=True) + tolerance # -lie_derivative
        ], dim=-1)


class CaltechFan(nn.Module):
    """Do not use.

    "Neural Lyapunov Control" for the Caltech fan example is not reproducible.
    """
    def __init__(self):
        super().__init__()
        self.V = nn.Sequential(
            nn.Linear(6, 6),
            nn.Tanh(),
            nn.Linear(6, 1),
            nn.Tanh(),
        )
        self.u1 = nn.Linear(6, 1, bias=False)
        self.u2 = nn.Linear(6, 1, bias=False)

    def forward(self, x):
        v = self.V(x)
        return v


def inverted_pendulum_data(spec):
    return {
        'X': torch.zeros(1, 2),
        'norm': spec['norm'],
        'eps': spec['epsilon'], 'eps_min': spec['epsilon_min'],
    }


def caltech_fan_data(spec):
    return {
        'X': torch.zeros(1, 6),
        'norm': spec['norm'],
        'eps': spec['epsilon'], 'eps_min': spec['epsilon_min'],
    }


if __name__ == '__main__':
    model = InvertedPendulum()
    print(model(torch.randn(1, 2)))
