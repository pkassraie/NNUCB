import torch.nn as nn

class NN(nn.Module):
    def __init__(self, input_dim, depth, width):
        super(NN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim,width))
        self.layers.append(nn.ReLU())
        for i in range(depth-1):
            self.layers.append(nn.Linear(width, width))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(width,1))

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x


class CNN(nn.Module):
    def __init__(self, input_dim, in_channels=1, depth=1, width=100):
        super(CNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append( nn.Conv1d(in_channels, width, kernel_size=3, padding=1, stride=1))
        self.layers.append(nn.ReLU(inplace=True))
        for i in range(depth-1):
            self.layers.append(nn.Conv1d(width, width, kernel_size=3, padding=1, stride=1))
            self.layers.append(nn.ReLU(inplace=True))
        self.fclast = nn.Linear(width * input_dim, 1)

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        x = x.view(x.size(0), -1)
        x = self.fclast(x)
        return x
