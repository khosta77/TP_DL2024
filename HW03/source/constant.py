import torch.nn as nn
import torch.optim as optim

SHAPE = 256
EPOCH = 40
LEARNING_RATE = 0.01
CRITERION = nn.BCELoss()
OPTIMIZER = lambda m: optim.SGD(m.parameters(), lr=LEARNING_RATE)
