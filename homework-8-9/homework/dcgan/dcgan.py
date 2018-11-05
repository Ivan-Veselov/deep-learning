import torch.nn as nn


class DCGenerator(nn.Module):
    def __init__(self, image_size, image_channels, latent_size):
        super(DCGenerator, self).__init__()

        self.init_size = image_size // 8
        self.l1 = nn.Sequential(nn.Linear(latent_size, 512 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, image_channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, data):
        data = data.view(data.shape[0], -1)
        out = self.l1(data)
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class DCDiscriminator(nn.Module):
    def __init__(self, image_size, image_channels):
        super(DCDiscriminator, self).__init__()

        self.model = nn.Sequential(
            *self.__discriminator_block(image_channels, 64, bn=False),
            *self.__discriminator_block(64, 128),
            *self.__discriminator_block(128, 256),
            *self.__discriminator_block(256, 512),
        )

        ds_size = image_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(512 * ds_size ** 2, 1),
                                       nn.Sigmoid())

    def forward(self, data):
        out = self.model(data)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity

    @staticmethod
    def __discriminator_block(in_filters, out_filters, bn=True):
        block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                 nn.LeakyReLU(0.2, inplace=True),
                 nn.Dropout2d(0.25)]
        if bn:
            block.append(nn.BatchNorm2d(out_filters, 0.8))

        return block
