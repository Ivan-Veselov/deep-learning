import argparse
import logging
import os

import torch
import torchvision.datasets as datasets
from torch.optim import Adam
from torchvision import transforms

from homework.dcgan.dcgan import DCGenerator, DCDiscriminator
from homework.dcgan.trainer import DCGANTrainer


def main():
    log_root = '../logs'
    log_name = 'train_dcgan.log'
    image_size = 32
    data_root = 'data'
    batch_size = 64
    epochs = 100
    n_show_samples = 8
    show_img_every = 10
    log_metrics_every = 100

    logging.basicConfig(
        format='%(asctime)s | %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_root,
                                             log_name)),
            logging.StreamHandler()],
        level=logging.INFO)

    transform = transforms.Compose([transforms.Scale(image_size), transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = datasets.CIFAR10(root=data_root, download=True,
                               transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                             num_workers=4, pin_memory=True)

    image_channels = 3
    latent_size = 100

    discriminator, generator =\
        DCDiscriminator(image_size, image_channels), DCGenerator(image_size, image_channels, latent_size)

    trainer = DCGANTrainer(generator=generator, discriminator=discriminator,
                           optimizer_d=Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)),
                           optimizer_g=Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999)),
                           metrics_dir='metrics', latent_size=latent_size)

    trainer.train(dataloader, epochs, n_show_samples, show_img_every, log_metrics_every)


if __name__ == '__main__':
    main()
