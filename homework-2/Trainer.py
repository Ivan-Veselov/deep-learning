import torch
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, net, loss, optimizer):
        self.__net = net
        self.__loss = loss
        self.__optimizer = optimizer

    def train(self, dataset, batch_size, epochs_num):
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.__net.train(True)

        for epoch in range(epochs_num):
            for inputs, labels in data_loader:
                self.__optimizer.zero_grad()
                outputs = self.__net(inputs)

                loss_value = self.__loss(outputs, labels)
                loss_value.backward()
                self.__optimizer.step()

            training_loss_after_epoch = self.__average_loss(data_loader)
            print('Training loss after epoch #{} = {:.3f}'.format(epoch + 1, training_loss_after_epoch))

        self.__net.train(False)

    def __average_loss(self, data):
        with torch.no_grad():
            loss_values = [self.__loss(self.__net(inputs), labels).unsqueeze(0) for inputs, labels in data]
            return torch.cat(loss_values).mean().item()
