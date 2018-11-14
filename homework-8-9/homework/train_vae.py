import logging
import os

import torch
from torch.optim import Adam, SGD
from torchvision import datasets
from torchvision.transforms import transforms

from homework.vae.trainer import Trainer
from homework.vae.vae import VAE, loss_function


def main():
    logging.basicConfig(
        format='%(asctime)s | %(message)s',
        handlers=[
            logging.FileHandler(os.path.join('../logs', 'train_dcgan.log')),
            logging.StreamHandler()
        ],
        level=logging.INFO
    )

    batch_size = 100

    dataset_folder = 'data'
    transformer = transforms.ToTensor()
    train_dataset = datasets.CIFAR10(dataset_folder, train=True, download=True, transform=transformer)
    test_dataset = datasets.CIFAR10(dataset_folder, train=False, download=True, transform=transformer)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    image_size = 32
    channels = 3
    model = VAE()

    trainer = Trainer(
        model,
        train_loader,
        test_loader,
        Adam(model.parameters(), lr=0.00005, betas=(0.5, 0.999)),
        loss_function,
        channels,
        image_size,
        'cpu'
    )

    epoch = 100
    log_interval = 10
    trainer.train(epoch, log_interval)


if __name__ == '__main__':
    main()
