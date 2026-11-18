"""Cubic B-spline edge-function KAN policy for the exact HeteroMRTA observation API.
Original work: Liu et al. KAN, arxiv:2404.19756; this is an independent,
compact, fixed-grid Cox-de Boor implementation, not the full pykan package.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid=5, degree=3):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.grid, self.degree = grid, degree
        step = 2.0 / grid
        knots = torch.linspace(-1.0 - degree * step, 1.0 + degree * step,
                               grid + 1 + 2 * degree)
        self.register_buffer('knots', knots)
        n_basis = knots.numel() - degree - 1
        self.base_weight = nn.Parameter(torch.empty(in_features, out_features))
        self.spline_weight = nn.Parameter(torch.empty(in_features, out_features, n_basis))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.015)

    def basis(self, x):
        # x: (..., input_dim); knots are extended beyond [-1, 1]
        x = x.clamp(-1.0 + 1e-6, 1.0 - 1e-6).unsqueeze(-1)
        t = self.knots
        b = ((x >= t[:-1]) & (x < t[1:])).to(x.dtype)
        for k in range(1, self.degree + 1):
            d1 = t[k:-1] - t[:-(k + 1)]
            d2 = t[(k + 1):] - t[1:-k]
            left = (x - t[:-(k + 1)]) / d1.clamp(min=1e-9) * b[..., :-1]
            right = (t[(k + 1):] - x) / d2.clamp(min=1e-9) * b[..., 1:]
            b = left + right
        return b

    def forward(self, x):
        leading = x.shape[:-1]
        x = x.reshape(-1, self.in_features)
        spline = torch.einsum('nib,iob->no', self.basis(x), self.spline_weight)
        base = F.silu(x) @ self.base_weight
        return (base + spline + self.bias).reshape(*leading, self.out_features)


class KANPolicy(nn.Module):
    """(tasks[B,T,15], agents[B,A,11], mask[B,T], index[B,1,1]) -> (prob, logprob).

    Same exact state/action/mask as original AttentionNet. Only the network differs.
    """
    def __init__(self, task_dim=15, agent_dim=11, hidden=24, grid=5):
        super().__init__()
        self.task_dim, self.agent_dim = task_dim, agent_dim
        # Normalize bounded/positive coordinates internally, not environment state.
        self.register_buffer('task_scale', torch.tensor([2.]*5 + [2.]*5 + [5., 10.] + [1.]*2 + [1.]))
        self.register_buffer('agent_scale', torch.tensor([1.]*5 + [10., 10., 10., 1., 1., 1.]))
        self.kan1 = KANLinear(task_dim * 2 + agent_dim * 2, hidden, grid=grid)
        self.kan2 = KANLinear(hidden, 1, grid=grid)

    def forward(self, tasks, agents, global_mask, index):
        b, t, _ = tasks.shape
        if global_mask.ndim == 3:
            global_mask = global_mask.squeeze(1)
        mask = global_mask.bool()
        agent_pad = agents.abs().sum(-1) > 0
        task_pad = tasks.abs().sum(-1) > 0
        ts = torch.tanh(tasks / self.task_scale)
        ag = torch.tanh(agents / self.agent_scale)
        mean_task = (ts * task_pad.unsqueeze(-1)).sum(1, keepdim=True) / task_pad.sum(1, keepdim=True).clamp(min=1).unsqueeze(-1)
        mean_agent = (ag * agent_pad.unsqueeze(-1)).sum(1, keepdim=True) / agent_pad.sum(1, keepdim=True).clamp(min=1).unsqueeze(-1)
        idx = index.reshape(b, 1, 1).expand(-1, -1, self.agent_dim)
        current_agent = torch.gather(ag, 1, idx)
        z = torch.cat([
            ts, mean_task.expand(-1, t, -1),
            current_agent.expand(-1, t, -1), mean_agent.expand(-1, t, -1)
        ], dim=-1)
        logits = self.kan2(torch.tanh(self.kan1(z))).squeeze(-1)
        # The original environment always leaves depot action (0) unmasked.
        logits = logits.masked_fill(mask, -1e9)
        logp = F.log_softmax(logits, dim=-1)
        return logp.exp(), logp
