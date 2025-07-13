import torch
import torch.nn as nn
from torch.distributions import Transform, transforms

class RealNVPBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.dim = dim
        self.net_s = nn.Sequential(
            nn.Linear(dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim // 2),
            nn.Tanh()
        )
        self.net_t = nn.Sequential(
            nn.Linear(dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim // 2)
        )

    def forward(self, x, reverse=False):
        x1, x2 = x.chunk(2, dim=1)
        if not reverse:
            s = self.net_s(x1)
            t = self.net_t(x1)
            y2 = x2 * torch.exp(s) + t
            log_det = s.sum(dim=1)
            return torch.cat([x1, y2], dim=1), log_det
        else:
            s = self.net_s(x1)
            t = self.net_t(x1)
            y2 = (x2 - t) * torch.exp(-s)
            log_det = -s.sum(dim=1)
            return torch.cat([x1, y2], dim=1), log_det

class NormalizingFlow(nn.Module):
    def __init__(self, dim, hidden_dim, n_flows):
        super().__init__()
        self.blocks = nn.ModuleList([RealNVPBlock(dim, hidden_dim) for _ in range(n_flows)])
        self.base_dist = torch.distributions.MultivariateNormal(
            loc=torch.zeros(dim),
            covariance_matrix=torch.eye(dim)
        )

    def forward(self, x):
        log_det_sum = 0
        z = x
        for block in self.blocks:
            z, log_det = block(z)
            log_det_sum += log_det
        log_prob = self.base_dist.log_prob(z) + log_det_sum
        return z, log_prob

    def sample(self, num_samples):
        z = self.base_dist.sample((num_samples,))
        x = z
        for block in reversed(self.blocks):
            x, _ = block(x, reverse=True)
        return x


def build_flow(cfg):
    """
    Build a normalizing flow model based on configuration.
    cfg.flow.dim: dimensionality of input
    cfg.flow.hidden_dim: hidden layer size
    cfg.flow.n_flows: number of RealNVP blocks
    """
    dim = cfg.flow.dim
    hidden_dim = cfg.flow.hidden_dim
    n_flows = cfg.flow.n_flows
    return NormalizingFlow(dim, hidden_dim, n_flows)
