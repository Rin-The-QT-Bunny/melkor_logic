from .spatial_layer import *
from .temporal_layer import *

class TOQNet(nn.Module):
    def __init__(self,spatial_config,temporal_config):
        super().__init__()
        self.spatial_net = SpatialLogicNet(spatial_config)
        self.temporal_net = TemporalLogicNet(temporal_config)
        in_dim = spatial_config[-1][0]
        self.temporal_init = TemporalInitial(in_dim,temporal_config[0])
    
    def forward(self,P,Q,R):
        P,Q,R = self.spatial_net(P,Q,R)
        intermediate = self.temporal_init(P)
        toq_out = self.temporal_net(intermediate)
        return toq_out

if __name__ == "__main__":
    T=25;d0=8;d1=10;d2=12;N = 4;b=3
    config1 = (
    (8,10,12),
    (10,12,14),
    (6,8,4),
    )

    config2 = (
    24,28,30,32
    )

    P = torch.randn([b,T,d0])
    Q = torch.randn([b,T,N,d1])
    R = torch.randn([b,T,N,N,d2])

    net = TOQNet(config1,config2)

    outputs = net(P,Q,R)
    print(outputs.shape)
