import torch
from torch import nn
from torch.utils.data import TensorDataset

from Trainer import Trainer
from network import SimpleCNN


def test_answer():
    net = SimpleCNN()
    trainer = Trainer(net, nn.CrossEntropyLoss(), torch.optim.Adam(net.parameters(), lr=0.001))

    dataset = TensorDataset(torch.rand(1, 1, 28, 28), torch.tensor([1]))
    trainer.train(dataset, batch_size=100, epochs_num=50)
