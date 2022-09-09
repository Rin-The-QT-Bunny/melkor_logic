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
        """Inputs:P:[b,D0];Q:[b,N,D1];R:[b,N,N,D2]"""
        bP,D0 = P.shape;bQ,N,D1 = Q.shape;bR,N,N,D2 = R.shape
        assert bP==bQ and bQ==bR,print("input batches doesn't match,expect P,Q,R has same input shape")
        assert D0==self.D0,print("input dim in P doesn't match, expect {},got {} instead").format(self.D0,D0)
        assert D1==self.D1,print("input dim in P doesn't match, expect {},got {} instead").format(self.D1,D1)
        assert D2==self.D2,print("input dim in P doesn't match, expect {},got {} instead").format(self.D2,D2)
        # create the reduce form of 1 order predicates to create the P update
        P_inputs = torch.cat([torch.max(Q,1).values,P],1)
        P_outputs = self.NNP(P_inputs)
        # create the expand form of P and the reduce form of R to create the Q feature
        P_expands = P.unsqueeze(1).repeat(1,N,1)
        R_reduce0 = torch.max(R,1).values
        R_reduce1 = torch.max(R,2).values
        Q_inputs = torch.cat([P_expands,Q,R_reduce0,R_reduce1],2)
        Q_outputs = self.NNQ(Q_inputs)
        # create the expand form of Q to creat the R features
        Q_expand0 = Q.unsqueeze(1).repeat(1,N,1,1)
        Q_expand1 = Q.unsqueeze(2).repeat(1,1,N,1)
        #print(R.shape,Q_expand0.shape,Q_expand1.shape)
        R_inputs = torch.cat([R,Q_expand0,Q_expand1],-1)
        R_outputs = self.NNR(R_inputs) # outputs features to generate next level predicates
        return P_outputs,Q_outputs,R_outputs

class NeuroLogicMachine(nn.Module):
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

if __name__=="__main__":
    config = (
        [3,5,2],
        [10,20,10],
        [10,25,20],
        [20,10,10],
    )
    net = NeuroLogicMachine(config)
    
    N = 10;b = 4
    global_conditions = torch.randn([b,3])
    object_conditions = torch.randn([b,N,5])
    relation_conditions = torch.randn([b,N,N,2])

    P,Q,R = net(global_conditions,object_conditions,relation_conditions)
    print(P.shape,Q.shape,R.shape)