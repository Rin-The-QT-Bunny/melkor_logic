import torch
import torch.nn as nn

from aluneth.rinlearn.nn.functional_net import FCBlock

def I(x):return x

class NNNet(nn.Module):
    def __init__(self,hidden,layers,inputs,outputs):
        super().__init__()
        self.backbone = FCBlock(hidden,layers,inputs,outputs)
    def forward(self,x):return torch.sigmoid(self.backbone(x) * 5)

class TemporalInitial(nn.Module):
    def __init__(self,in_dims,out_dims):
        super().__init__()
        self.NN1 = NNNet(132,2,in_dims,out_dims)
        #self.NN1 = I
    def forward(self,P):
        # input shape: [b,T,D]
        b,T,D = P.shape;outputs = []
        for t in range(T):
            outputs.append(self.NN1(torch.max(P[:,t:,:],1).values.unsqueeze(1)))
        outputs = torch.cat(outputs,1)
        # the predicate represents for all t,p(t) or exist t,q(t)
        # any(not p(t)) = False -> for all p(t)=True
        # any(p(t)) = True -> exist t, p(t)=True (primitive candidates)
        return outputs

class TemporalLogicLayer(nn.Module):
    def __init__(self,in_dims,out_dims):
        super().__init__()
        self.in_dims = in_dims
        self.NN1 = NNNet(132,2,in_dims+in_dims,out_dims)

    def forward(self,P):
        b,T,D = P.shape;outputs = []
        assert D==self.in_dims,print("input dims doesn't match")
        for t in range(T):
            stage_outputs = []
            for t_end in range(t,T):                
                features = torch.max(P[:,t:t_end+1,:],1).values
                features = torch.cat([features,P[:,t_end,:]],-1)
                stage_outputs.append(self.NN1(features).unsqueeze(1))
            stage_outputs = torch.cat(stage_outputs,1)
            outputs.append(torch.max(stage_outputs,1).values.unsqueeze(1))
        return torch.cat(outputs,1)

class TemporalLogicNet(nn.Module):
    def __init__(self,configs):
        super().__init__()
        self.configs = configs
        self.temporal_nets = nn.ModuleList([])
        for i in range(len(configs)):
            self.temporal_nets.append(TemporalLogicLayer(configs[i][0],configs[i][1]))
    def forward(self,x):
        for net in self.temporal_nets:
            x = net(x)
        return x
      