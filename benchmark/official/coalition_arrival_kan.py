"""Coalition-arrival logit residual for Stage-A WarmRelationalKAN.

Uses *only* the original TaskEnv observation (no future outcomes).  Each task
has a currently missing 5-dimensional skill vector.  We estimate when the
k-th distinct other robot carrying a required skill could become available,
using relative coordinates, the pinned environment's robot speed (0.2), and
observed remaining travel/work times.  Taking the maximum over the required
skill dimensions gives an optimistic coalition-ready time, not a claim of an
exact optimal coalition (one robot can meet multiple skill requirements).

The zero-initialized final layer ensures predictions equal the Stage-A model
before additional updates. The entire pretrained network remains trainable.
"""
import torch
from torch import nn
from torch.nn import functional as F
from warm_relational_kan import WarmRelationalKAN

class CoalitionArrivalKAN(WarmRelationalKAN):
    def __init__(self, d=32, heads=4, velocity=0.2):
        super().__init__(d=d, heads=heads)
        if velocity <= 0:raise ValueError('robot velocity must be positive')
        self.velocity=float(velocity)
        self.coalition_head=nn.Sequential(nn.Linear(9,32),nn.Tanh(),nn.Linear(32,1))
        nn.init.zeros_(self.coalition_head[-1].weight)
        nn.init.zeros_(self.coalition_head[-1].bias)

    def transfer_stage_a(self, stage_a):
        source=stage_a.state_dict()
        target=self.state_dict()
        # No old tensor is skipped or reshaped.
        for key,value in source.items():
            if key not in target or target[key].shape != value.shape:
                raise RuntimeError(f'Cannot transplant Stage-A tensor {key}')
            target[key]=value.detach().clone()
        self.load_state_dict(target,strict=True)

    def coalition_features(self,tasks,agents,index):
        b,t,_=tasks.shape
        a=agents.shape[1]
        assert tasks.shape[-1]==15 and agents.shape[-1]==11
        deficit=tasks[...,:5].clamp_min(0.)                         # B,T,5
        skill=agents[...,:5].clamp_min(0.)                          # B,A,5
        current_idx=index.reshape(b).long()
        current_ability=skill[torch.arange(b,device=skill.device),current_idx]  # B,5
        missing=(deficit-current_ability.unsqueeze(1)).clamp_min(0.)  # B,T,5
        required=missing.ceil().long()                              # robot units per trait

        # (current - task) - (current - other robot) = other robot - task.
        offset=tasks[...,12:14].unsqueeze(2)-agents[...,8:10].unsqueeze(1)
        dist=torch.linalg.vector_norm(offset,dim=-1)                # B,T,A
        available_after=(agents[...,5].clamp_min(0)+agents[...,6].clamp_min(0))
        eta=(dist/self.velocity)+available_after.unsqueeze(1)       # B,T,A
        current_eta=tasks[...,11].clamp_min(0)                      # B,T
        valid_agent=(agents.abs().sum(-1)>0)
        valid_agent[torch.arange(b,device=skill.device),current_idx]=False

        # Candidate kth arrivals for each required trait; invalid robots excluded.
        qualified=(skill>0).permute(0,2,1).unsqueeze(1)             # B,1,5,A
        feasible=qualified & valid_agent.unsqueeze(1).unsqueeze(1)
        eta_trait=eta.unsqueeze(2).expand(-1,-1,5,-1)
        ranked=eta_trait.masked_fill(~feasible,1.e5).sort(dim=-1).values
        needed=required.clamp(min=1,max=a)
        selected=ranked.gather(-1,(needed-1).unsqueeze(-1)).squeeze(-1)
        required_trait=required>0
        unavailable=required_trait & (selected>=1.e4)
        selected=selected.clamp(max=60.)
        partner_eta=selected.masked_fill(~required_trait,0.).amax(dim=-1)
        earliest=selected.masked_fill(~required_trait,60.).amin(dim=-1)
        earliest=torch.where(required_trait.any(-1),earliest,current_eta)
        own_wait=(partner_eta-current_eta).clamp_min(0)
        partner_wait=(current_eta-earliest).clamp_min(0)
        eta_spread=(partner_eta-earliest).clamp_min(0)
        demand=required.sum(-1).float()
        pool=(feasible.sum(-1)).float().expand(-1,t,-1)
        scarcity=((required.float()/(pool+1.e-3)) * required_trait).sum(-1)/required_trait.sum(-1).clamp_min(1)
        shortage=unavailable.float().sum(-1)/required_trait.sum(-1).clamp_min(1)
        duration=tasks[...,10].clamp_min(0.)
        # B,T,9 normalized observations. All zero for depot/no deficit.
        features=torch.stack((torch.tanh(current_eta/10),torch.tanh(partner_eta/10),
               torch.tanh(own_wait/10),torch.tanh(partner_wait/10),
               torch.tanh(eta_spread/10),torch.tanh(demand/5),
               scarcity.clamp(0,2)/2,shortage,torch.tanh(duration/10)),dim=-1)
        is_task=(deficit.sum(-1)>0).unsqueeze(-1)
        return features*is_task.to(features.dtype)

    def forward(self,tasks,agents,global_mask,index):
        _,base_logp=super().forward(tasks,agents,global_mask,index)
        edge=self.coalition_head(self.coalition_features(tasks,agents,index)).squeeze(-1)
        mask=global_mask.squeeze(1) if global_mask.ndim==3 else global_mask
        logits=(base_logp+edge).masked_fill(mask.bool(),-1e9)
        logp=F.log_softmax(logits,dim=-1)
        return logp.exp(),logp
