### LeNet

from functools import reduce
import operator as op

import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):

    
    def __init__(self):
        super(LeNet, self).__init__()

        # 1 input channel
        # 6 output channel
        # 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


    def forward(self, x):
        # max pooling over (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))

        x = x.view(-1, LeNet.num_flat_features(x))

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:] # all dimensions except the batch dimension

        return reduce(op.mul, size)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
