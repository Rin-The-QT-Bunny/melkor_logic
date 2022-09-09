import torch
import torch.nn as nn

from moic.mklearn.nn.functional_net import FCBlock

class NNNet(nn.Module):
    def __init__(self,hidden,layers,inputs,outputs):
        super().__init__()
        self.backbone = FCBlock(hidden,layers,inputs,outputs)
    def forward(self,x):return torch.sigmoid(self.backbone(x) * 5)

class SpatialRelationLayer(nn.Module):
    def __init__(self,D,O):
        super().__init__()
        D0,D1,D2 = D;O0,O1,O2 = O
        self.D0 = D0;self.D1 = D1;self.D2 = D2
        self.NNP = NNNet(132,2,D0+D1,O0)
        self.NNQ = NNNet(132,2,D1+D0+2*D2,O1)
        self.NNR = NNNet(132,2,D2+D1*2,O2)

    def forward(self,P,Q,R):
        """Inputs:P:[b,T,D0];Q:[b,T,N,D1];R:[b,T,N,N,D2]"""
        bP,T,D0 = P.shape;bQ,T,N,D1 = Q.shape;bR,T,N,N,D2 = R.shape
        assert bP==bQ and bQ==bR,print("input batches doesn't match,expect P,Q,R has same input shape")
        assert D0==self.D0,print("input dim in P doesn't match, expect {},got {} instead").format(self.D0,D0)
        assert D1==self.D1,print("input dim in P doesn't match, expect {},got {} instead").format(self.D1,D1)
        assert D2==self.D2,print("input dim in P doesn't match, expect {},got {} instead").format(self.D2,D2)
        # create the reduce form of 1 order predicates to create the P update
        P_inputs = torch.cat([torch.max(Q,2).values,P],2)
        P_outputs = self.NNP(P_inputs)
        # create the expand form of P and the reduce form of R to create the Q feature
        P_expands = P.unsqueeze(2).repeat(1,1,N,1)
        R_reduce0 = torch.max(R,2).values
        R_reduce1 = torch.max(R,3).values
        Q_inputs = torch.cat([P_expands,Q,R_reduce0,R_reduce1],3)
        Q_outputs = self.NNQ(Q_inputs)
        # create the expand form of Q to creat the R features
        Q_expand0 = Q.unsqueeze(2).repeat(1,1,N,1,1)
        Q_expand1 = Q.unsqueeze(3).repeat(1,1,1,N,1)
        #print(R.shape,Q_expand0.shape,Q_expand1.shape)
        R_inputs = torch.cat([R,Q_expand0,Q_expand1],-1)
        R_outputs = self.NNR(R_inputs) # outputs features to generate next level predicates
        return P_outputs,Q_outputs,R_outputs

class SpatialLogicNet(nn.Module):
    def __init__(self,configs):
        super().__init__()
        self.configs = configs # config of how network layers got stack together
        self.spatial_nets = nn.ModuleList([])
        for i in range(len(configs)-1):
            self.spatial_nets.append(SpatialRelationLayer(configs[i],configs[i+1]))
    def forward(self,P,Q,R):
        # the network draw conclusions from the input predicates
        for net in self.spatial_nets:
            P,Q,R = net(P,Q,R)
        # each predicate represents the soft probablity of each term being correct
        return P,Q,R
