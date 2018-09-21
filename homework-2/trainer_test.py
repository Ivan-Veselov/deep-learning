import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import TensorDataset

from Trainer import Trainer
from ResNeXt import ResNeXt


def test_answer():
    net = ResNeXt([3, 4, 6, 3], 32)
    trainer = Trainer(net, nn.CrossEntropyLoss(), torch.optim.Adam(net.parameters(), lr=0.001), None)

    dataset = TensorDataset(
        torch.rand(2, ResNeXt.in_channels, ResNeXt.in_map_size, ResNeXt.in_map_size),
        torch.tensor([1, 2])
    )

    trainer.train(dataset, batch_size=256, epochs_num=1)
