import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_list=list, out_dim=2, drop_out=0.5):
        super().__init__()
        hidden_list.insert(0, in_dim)
        self.hidden_layers = nn.Sequential()
        for i in range(len(hidden_list)-1):
            self.hidden_layers.append(
                nn.Sequential(
                nn.Linear(hidden_list[i], hidden_list[i+1]),
                nn.LeakyReLU(),
                nn.Dropout(drop_out)
            ))
        self.output_layer = nn.Linear(hidden_list[-1], out_dim)

    def forward(self, X):
        X_hid = self.hidden_layers(X)
        return self.output_layer(X_hid)

if __name__ == '__main__':
    X = torch.rand(10, 1000)
    net = MLP(X.size()[1], [512, 256, 128, 64, 32], 2)
    print(net)
    print(net(X))