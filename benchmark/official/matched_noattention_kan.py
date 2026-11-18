"""Parameter-matched feed-forward control for WarmRelationalKAN.

This is a deliberately NO-ATTENTION control: static masked mean pooling over
agents, a per-task MLP, and the exact same original KAN head and adapter.
No query/key weighting, multi-head attention or attention softmax is used.
The residual adapter starts at zero, ensuring exact KAN V7.1 predictions at
initialization when both models load the same pretrained KAN checkpoint.
"""
import torch
from torch import nn
from torch.nn import functional as F
from warm_relational_kan import WarmRelationalKAN

class MatchedNoAttentionKAN(WarmRelationalKAN):
    def __init__(self):
        super().__init__(d=32,heads=4)
        # Drop all trainable attention-specific tensors; retain the same task
        # and agent projection layers, LayerNorm, adapter, and full KAN.
        del self.task_attention
        del self.query
        del self.key
        del self.value
        del self.edge_bias
        # 99->56->32 = 7440 parameters replaces 7,393 attention parameters.
        # This makes the control 47 parameters larger than WarmRelationalKAN.
        self.relation_ffn = nn.Sequential(
            nn.Linear(99,56), nn.Tanh(), nn.Linear(56,32)
        )

    def forward(self,tasks,agents,global_mask,index):
        z,ts,ag,task_pad,agent_pad,cur=self._features(tasks,agents,index)
        b,t,_=tasks.shape
        tp=task_pad.clone();ap=agent_pad.clone()
        tp[:,0]=True;ap[:,0]=True
        q=self.task_proj(ts)
        agent_tokens=self.agent_proj(ag)
        global_task=(q*tp.unsqueeze(-1)).sum(dim=1,keepdim=True)/tp.sum(dim=1,keepdim=True).clamp(min=1).unsqueeze(-1)
        global_agents=(agent_tokens*ap.unsqueeze(-1)).sum(dim=1,keepdim=True)/ap.sum(dim=1,keepdim=True).clamp(min=1).unsqueeze(-1)
        task_context=self.task_ln(q+global_task)
        idx=index.reshape(b).long()
        current=agent_tokens[torch.arange(b,device=tasks.device),idx].unsqueeze(1).expand(-1,t,-1)
        remaining=tasks[...,0:5].clamp(min=0)
        abilities=agents[...,0:5].clamp(min=0)
        curr_ab=abilities[torch.arange(b,device=tasks.device),idx].unsqueeze(1)
        need=remaining.sum(-1).clamp_min(1e-6)
        useful=torch.minimum(remaining,curr_ab).sum(-1,keepdim=True)/need.unsqueeze(-1)
        shortage=(remaining-curr_ab).clamp(min=0).sum(-1,keepdim=True)/need.unsqueeze(-1)
        travel=ts[...,11:12]
        pooled_agents=global_agents.expand(-1,t,-1)
        pooled_input=torch.cat((task_context,pooled_agents,current,useful,shortage,travel),dim=-1)
        agent_context=self.relation_ffn(pooled_input)
        context=torch.cat((task_context,agent_context,current,useful,shortage,travel),dim=-1)
        residual=self.adapter(context)
        logits=self.kan.kan2(torch.tanh(self.kan.kan1(z+residual))).squeeze(-1)
        mask=global_mask.squeeze(1) if global_mask.ndim==3 else global_mask
        logits=logits.masked_fill(mask.bool(),-1e9)
        lp=F.log_softmax(logits,dim=-1)
        return lp.exp(),lp
