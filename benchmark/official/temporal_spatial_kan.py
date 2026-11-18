"""Time-/space-aware relational KAN, initialized to EXACTLY mimic trained hybrid.

The original TaskEnv observer provides:
 task[:, 11]: current-robot travel time to task
 task[:, 12:14]: current-robot position minus task position
 agent[:, 5]: time until other robot's current arrival
 agent[:, 6]: remaining working time of other robot
 agent[:, 8:10]: current-robot position minus other-robot position
 agent[:, 10]: assigned flag
Thus (task[12:14] - agent[8:10]) is other-robot position minus task position.
All robot velocities are 0.2 in the PINNED authors' TaskEnv.

New edge dimensions are initialized with ZERO weights; the first three old
edge-bias features and original residual adapter are copied strictly. The
new model therefore reproduces the pretrained hybrid until optimizer updates.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from warm_relational_kan import WarmRelationalKAN, count_parameters


class TemporalSpatialKAN(WarmRelationalKAN):
    def __init__(self, d=32, heads=4, robot_velocity=0.2):
        super().__init__(d=d, heads=heads)
        self.velocity = float(robot_velocity)
        if self.velocity <= 0:
            raise ValueError('velocity must be positive')
        self.edge_bias = nn.Sequential(nn.Linear(7, 16), nn.Tanh(), nn.Linear(16, 1))
        self.adapter = nn.Linear(3 * d + 5, 52)
        # Do not use new terms before weights are transferred from old hybrid.
        nn.init.zeros_(self.edge_bias[0].weight[:, 3:])
        nn.init.zeros_(self.adapter.weight[:, 3*d+3:])

    def transfer_pretrained(self, old: WarmRelationalKAN):
        """Copy ALL original weights exactly, growing 2 layers with zero new inputs."""
        if old.d != self.d:
            raise ValueError('Attention dimensions differ')
        incoming = old.state_dict()
        target = self.state_dict()
        resized = {'edge_bias.0.weight', 'adapter.weight'}
        with torch.no_grad():
            for k, value in incoming.items():
                if k in resized:
                    continue
                if k not in target or target[k].shape != value.shape:
                    raise RuntimeError(f'Unmatched pretrained key {k}: {value.shape}')
                target[k].copy_(value)
            target['edge_bias.0.weight'].zero_()
            target['edge_bias.0.weight'][:, :3].copy_(incoming['edge_bias.0.weight'])
            target['adapter.weight'].zero_()
            target['adapter.weight'][:, :3*self.d+3].copy_(incoming['adapter.weight'])
            self.load_state_dict(target, strict=True)

    def forward(self, tasks, agents, global_mask, index):
        z_base, ts, ag, task_valid, agent_valid, _ = self._features(tasks, agents, index)
        b, t, _ = tasks.shape
        tv = task_valid.clone()
        av = agent_valid.clone()
        tv[:, 0] = True  # depot is an actual action
        av[:, 0] = True
        q = self.task_proj(ts)
        a = self.agent_proj(ag)
        task_ctx, _ = self.task_attention(q, q, q,
                                          key_padding_mask=~tv, need_weights=False)
        task_ctx = self.task_ln(q + task_ctx)

        remaining = tasks[..., :5].clamp_min(0)
        abilities = agents[..., :5].clamp_min(0)
        need = remaining.sum(-1).clamp_min(1e-6)
        overlap = torch.minimum(remaining.unsqueeze(2), abilities.unsqueeze(1)).sum(-1) / need.unsqueeze(-1)
        shortfall = (remaining.unsqueeze(2) - abilities.unsqueeze(1)).clamp_min(0).sum(-1) / need.unsqueeze(-1)
        assigned = agents[..., 10].unsqueeze(1).expand(-1,t,-1)
        # Spatial relative distance from *each* robot to *each* task, visible in original observation.
        other_minus_task = tasks[..., 12:14].unsqueeze(2) - agents[..., 8:10].unsqueeze(1)
        other_travel = torch.linalg.vector_norm(other_minus_task, dim=-1) / self.velocity
        # Estimate availability from current travel/work commitments in visible state.
        busy = (agents[..., 5].clamp_min(0) + agents[..., 6].clamp_min(0)).unsqueeze(1).expand(-1,t,-1)
        eta = busy + other_travel
        current_eta = tasks[..., 11].clamp_min(0).unsqueeze(-1)
        travel_scaled = torch.tanh(other_travel / 10.)
        busy_scaled = torch.tanh(busy / 10.)
        eta_gap_scaled = torch.tanh((eta - current_eta).abs() / 10.)
        # First 3 edge dimensions match original model, extra 4 starts with zero effect.
        edges = torch.stack((overlap, shortfall, assigned, travel_scaled,
                             busy_scaled, eta_gap_scaled, torch.tanh(eta / 10.)), dim=-1)
        bias = self.edge_bias(edges).squeeze(-1)
        scores = torch.matmul(self.query(task_ctx), self.key(a).transpose(1,2)) / math.sqrt(self.d)
        scores = (scores + bias).masked_fill(~av.unsqueeze(1), -1e4)
        weights = scores.softmax(-1)
        agent_ctx = weights @ self.value(a)
        ix = index.reshape(b).long()
        cur = a[torch.arange(b, device=a.device), ix].unsqueeze(1).expand(-1,t,-1)
        current_ability = abilities[torch.arange(b, device=a.device),ix].unsqueeze(1)
        useful = torch.minimum(remaining,current_ability).sum(-1,keepdim=True)/need.unsqueeze(-1)
        shortage = (remaining-current_ability).clamp_min(0).sum(-1,keepdim=True)/need.unsqueeze(-1)
        # EXACT same first 3 scalar adapter features as old hybrid.
        travel = ts[...,11:12]
        coalition_eta = (weights * torch.tanh(eta / 10.)).sum(-1,keepdim=True)
        coalition_busy = (weights * busy_scaled).sum(-1,keepdim=True)
        context = torch.cat((task_ctx, agent_ctx,cur,useful,shortage,travel,
                             coalition_eta,coalition_busy),dim=-1)
        z = z_base + self.adapter(context)
        logits = self.kan.kan2(torch.tanh(self.kan.kan1(z))).squeeze(-1)
        mask = global_mask.squeeze(1) if global_mask.ndim == 3 else global_mask
        logits = logits.masked_fill(mask.bool(), -1e9)
        logp = F.log_softmax(logits,dim=-1)
        return logp.exp(),logp


def verify_transfer(old, new, tasks, agents, mask, index, tolerance=2e-6):
    old.eval(); new.eval()
    with torch.no_grad():
        po,_ = old(tasks, agents,mask,index)
        pn,_ = new(tasks,agents,mask,index)
    diff = float((po-pn).abs().max().item())
    if diff > tolerance:
        raise AssertionError(f'Temporal-spatial transfer changed pretrained predictions: delta={diff}')
    return diff
