import torch
from torch import nn
import os
from torchsummary import summary
from fvcore.nn import FlopCountAnalysis, parameter_count


class Net(nn.Module):
    def __init__(self, pretrain=None):
        super().__init__()
        self.name = os.path.basename(__file__).split('.')[0]

        self.fc = nn.Sequential(
            nn.Linear(15, 36),
            nn.ReLU(),
            nn.Linear(36, 72),
            nn.ReLU(),
            nn.Linear(72, 32),
            nn.ReLU(),
            nn.Linear(32, 6),
            nn.Sigmoid()
        )
        self.apply(self.weights_init)


    def forward(self, input):
        input=input.view(input.size(0),-1)
        output = self.fc(input)
        return output
    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net().to(device)    # print(net)
    x = torch.randn(1, 15).to(device)
    y = net(x)
    print(y.shape)
    summary(net, (15,))

    # Using fvcore to calculate FLOPs and parameters
    flops = FlopCountAnalysis(net, x)
    params = parameter_count(net)
    print(f"FLOPs: {flops.total():.0f}")
