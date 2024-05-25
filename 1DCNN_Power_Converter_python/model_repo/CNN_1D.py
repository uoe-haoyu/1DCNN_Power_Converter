import torch
from torch import nn
import os
from einops import rearrange
import torch.nn.functional as F
from torchsummary import summary
from fvcore.nn import FlopCountAnalysis, parameter_count
class Net(nn.Module):
    def __init__(self, pretrain=None):
        super().__init__()
        self.name = os.path.basename(__file__).split('.')[0]

        self.fc_in = nn.Sequential(
            nn.Linear(15, 36),
            nn.ReLU(),
        )

        self.con1=nn.Sequential(
            nn.Conv1d(1,4,3,padding=1),
        )
        self.BN = nn.BatchNorm1d(4)
        self.ReLU = nn.ReLU()
        self.MaxPool = nn.MaxPool1d(2)


        self.fc_out = nn.Sequential(
            nn.Linear(72, 32),
            nn.ReLU(),
            # nn.Dropout(0.2),
            # nn.Linear(64, 32),
            # nn.ReLU(),
            nn.Linear(32, 6),
            nn.Sigmoid()
        )

    def forward(self, input):
        input=input.view(input.size(0),-1)
        output = self.fc_in(input)
        output = output.view(-1, 1, 36)

        output = self.con1(output)
        output = self.BN(output)
        output = self.ReLU(output)
        output = self.MaxPool(output)

        output=output.view(output.size(0),-1)
        output = self.fc_out(output)



        return output

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net().to(device)    # print(net)
    x = torch.randn(1, 15).to(device)
    y = net(x)
    print(y.shape)
    # Using torchsummary to print the summary of the model
    summary(net, (15,))

    flops = FlopCountAnalysis(net, x)
    params = parameter_count(net)
    print(f"FLOPs: {flops.total():.0f}")
