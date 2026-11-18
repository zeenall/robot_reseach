"""Warm-started relational Attention -> original 52-input KAN head.

The original 11,473-parameter KAN V7.1 is preserved under self.kan.  A new
relational adapter produces a 52-d residual added to the EXACT original 52-d
feature vector. Its final projection starts at all zeros, so warm-start
predictions initially equal the baseline KAN exactly (up to FP roundoff).
All parameters, including original KAN, remain trainable throughout RL.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from kan_policy import KANPolicy


class WarmRelationalKAN(nn.Module):
    def __init__(self, d=32, heads=4):
        super().__init__()
        if d % heads:
            raise ValueError('Attention width must be divisible by heads')
        self.d = d
        self.kan = KANPolicy(task_dim=15, agent_dim=11, hidden=24, grid=5)
        self.task_proj = nn.Sequential(nn.Linear(15, d), nn.LayerNorm(d), nn.Tanh())
        self.agent_proj = nn.Sequential(nn.Linear(11, d), nn.LayerNorm(d), nn.Tanh())
        self.task_attention = nn.MultiheadAttention(d, heads, batch_first=True, dropout=0.)
        self.task_ln = nn.LayerNorm(d)
        self.query = nn.Linear(d, d, bias=False)
        self.key = nn.Linear(d, d, bias=False)
        self.value = nn.Linear(d, d, bias=False)
        # Edge-aware bias from [useful skill ratio, shortage ratio, assigned indicator].
        self.edge_bias = nn.Sequential(nn.Linear(3, 16), nn.Tanh(), nn.Linear(16, 1))
        # Task context, task-to-all-agents context, current agent, 3 explicit edges.
        self.adapter = nn.Linear(3 * d + 3, 52)
        nn.init.zeros_(self.adapter.weight)
        nn.init.zeros_(self.adapter.bias)
        self._initial_kan_params = sum(p.numel() for p in self.kan.parameters())

    def _features(self, tasks, agents, index):
        """Bit-for-bit counterpart of KANPolicy.forward input assembly."""
        b, t, _ = tasks.shape
        ts = torch.tanh(tasks / self.kan.task_scale)
        ag = torch.tanh(agents / self.kan.agent_scale)
        task_pad = tasks.abs().sum(-1) > 0
        agent_pad = agents.abs().sum(-1) > 0
        mean_task = ((ts * task_pad.unsqueeze(-1)).sum(1, keepdim=True)
                     / task_pad.sum(1, keepdim=True).clamp(min=1).unsqueeze(-1))
        mean_agent = ((ag * agent_pad.unsqueeze(-1)).sum(1, keepdim=True)
                      / agent_pad.sum(1, keepdim=True).clamp(min=1).unsqueeze(-1))
        idx = index.reshape(b, 1, 1).long().expand(-1, -1, 11)
        current_agent = torch.gather(ag, 1, idx)
        z = torch.cat((ts, mean_task.expand(-1, t, -1),
                       current_agent.expand(-1, t, -1),
                       mean_agent.expand(-1, t, -1)), dim=-1)
        return z, ts, ag, task_pad, agent_pad, current_agent

    def forward(self, tasks, agents, global_mask, index):
        base_z, ts, ag, task_pad, agent_pad, cur = self._features(tasks, agents, index)
        b, t, _ = tasks.shape
        # Depot is a REAL action, not padding. Always include its token.
        tp = task_pad.clone()
        ap = agent_pad.clone()
        tp[:, 0] = True
        ap[:, 0] = True
        q = self.task_proj(ts)
        a = self.agent_proj(ag)
        self_ctx, _ = self.task_attention(q, q, q,
                                          key_padding_mask=~tp, need_weights=False)
        self_ctx = self.task_ln(q + self_ctx)
        remaining = tasks[..., 0:5].clamp(min=0)
        abilities = agents[..., 0:5].clamp(min=0)
        need = remaining.sum(-1).clamp_min(1e-6)
        overlap = torch.minimum(remaining.unsqueeze(2), abilities.unsqueeze(1)).sum(-1) / need.unsqueeze(-1)
        shortfall = (remaining.unsqueeze(2) - abilities.unsqueeze(1)).clamp_min(0).sum(-1) / need.unsqueeze(-1)
        assigned = agents[..., 10].unsqueeze(1).expand(-1, t, -1)
        edges = torch.stack((overlap, shortfall, assigned), dim=-1)
        bias = self.edge_bias(edges).squeeze(-1)
        scores = torch.matmul(self.query(self_ctx), self.key(a).transpose(1,2)) / math.sqrt(self.d)
        scores = scores + bias
        scores = scores.masked_fill(~ap.unsqueeze(1), -1e4)
        attn = scores.softmax(-1)
        agent_ctx = torch.matmul(attn, self.value(a))
        curidx = index.reshape(b).long()
        current = a[torch.arange(b, device=a.device), curidx].unsqueeze(1).expand(-1, t, -1)
        current_ability = abilities[torch.arange(b, device=a.device), curidx].unsqueeze(1)
        useful = torch.minimum(remaining, current_ability).sum(-1, keepdim=True) / need.unsqueeze(-1)
        shortage = (remaining - current_ability).clamp(min=0).sum(-1, keepdim=True) / need.unsqueeze(-1)
        travel = ts[..., 11:12]   # Time-to-task, as recorded in ORIGINAL TaskEnv.
        context = torch.cat((self_ctx, agent_ctx, current, useful, shortage, travel), dim=-1)
        z = base_z + self.adapter(context)
        logits = self.kan.kan2(torch.tanh(self.kan.kan1(z))).squeeze(-1)
        mask = global_mask.squeeze(1) if global_mask.ndim == 3 else global_mask
        logits = logits.masked_fill(mask.bool(), -1e9)
        logp = F.log_softmax(logits, dim=-1)
        return logp.exp(), logp


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def verify_exact_warmstart(kan, hybrid, tasks, agents, mask, index):
    kan.eval(); hybrid.eval()
    with torch.no_grad():
        p1, _ = kan(tasks, agents, mask, index)
        p2, _ = hybrid(tasks, agents, mask, index)
    error = (p1 - p2).abs().max().item()
    if error > 2e-6:
        raise AssertionError(f'WARM START NOT IDENTICAL: max diff={error}')
    return error
