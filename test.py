from melkor_logic.TOQNet import *

T=25;d0=8;d1=10;d2=12;N = 4;b=3
config1 = (
    (8,10,12),
    (10,12,14),
    (6,8,4),
    )

config2 = (
    [24,28],
    [28,30],
    [30,32]
)

P = torch.randn([b,T,d0])
Q = torch.randn([b,T,N,d1])
R = torch.randn([b,T,N,N,d2])

net = TOQNet(config1,config2)

outputs = net(P,Q,R)
print(outputs.shape)
