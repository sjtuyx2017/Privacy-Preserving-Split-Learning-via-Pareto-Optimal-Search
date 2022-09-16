# Encoder only outputs means, variances are set manually
import functools
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math

class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, bn=False, stride=1):
        super(ResBlock, self).__init__()
        self.bn = bn
        if bn:
            self.bn0 = nn.BatchNorm2d(in_planes)

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1)
        if bn:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1)
        self.shortcut = nn.Sequential()

        if stride > 1:
            self.shortcut = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)

    def forward(self, x):
        if self.bn:
            out = F.relu(self.bn0(x))
        else:
            out = F.relu(x)

        if self.bn:
            out = F.relu(self.bn1(self.conv1(out)))
        else:
            out = F.relu(self.conv1(out))

        out = self.conv2(out)
        out += self.shortcut(x)
        return out

class ResBlock_L(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes,bn=False):
        super(ResBlock_L, self).__init__()

        self.net1 = nn.Linear(in_planes, planes)
        self.net2 = nn.Linear(planes, planes)
        self.shortcut = nn.Sequential()

        self.shortcut = nn.Linear(in_planes, planes)

    def forward(self, x):

        out = F.relu(self.net1(x))
        out = self.net2(out)
        out += self.shortcut(x)
        return out


# local encoder with 2 CNN (without compression)
class Encoder1(nn.Module):
    def __init__(self):
        super(Encoder1,self).__init__()
        self.cnnModel = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 24
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 12
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )

    def forward(self, x):
        output = self.cnnModel(x)

        return output

class InfocensorEncoder1(nn.Module):
    def __init__(self):
        super(InfocensorEncoder1,self).__init__()
        self.cnnModel = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 24
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        self.mu = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 12
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        self.logvar = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 12
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )

    def forward(self, x):
        output = self.cnnModel(x)
        mu = self.mu(output)
        logvar = self.logvar(output)
        sigma = torch.exp(0.5 * logvar)
        return mu, sigma

# local encoder with 3 CNN
class Encoder2(nn.Module):
    def __init__(self):
        super(Encoder2,self).__init__()
        self.cnnModel = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 24
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 12
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 6
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )

    def forward(self, x):
        output = self.cnnModel(x)
        return output

class InfocensorEncoder2(nn.Module):
    def __init__(self):
        super(InfocensorEncoder2,self).__init__()
        self.cnnModel = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 24
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 12
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        self.mu = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 12
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.logvar = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 12
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )

    def forward(self, x):
        output = self.cnnModel(x)
        mu = self.mu(output)
        logvar = self.logvar(output)
        sigma = torch.exp(0.5 * logvar)
        return mu, sigma


# local encoder with all the 4 CNN
class Encoder3(nn.Module):
    def __init__(self):
        super(Encoder3,self).__init__()
        self.cnnModel = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 24
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 12
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 6
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 3
            # nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # flatten
        )

    def forward(self, x):
        output = self.cnnModel(x)
        output = output.squeeze()
        return output

class InfocensorEncoder3(nn.Module):
    def __init__(self):
        super(InfocensorEncoder3,self).__init__()
        self.cnnModel = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 24
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 12
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 6
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        self.mu = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 3
            # nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # flatten
        )
        self.logvar = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 3
            # nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # flatten
        )

    def forward(self, x):
        output = self.cnnModel(x)
        mu = self.mu(output)
        logvar = self.logvar(output)
        sigma = torch.exp(0.5 * logvar)
        return mu, sigma


class Classifier1(nn.Module):
    def __init__(self, out_dim):
        super(Classifier1,self).__init__()
        self.cnnModel = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 6
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 3
            # nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # flatten
        )

        self.dnnModel = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, out_dim)
        )

    def forward(self,x):
        output = self.cnnModel(x)
        output = output.squeeze()
        output = self.dnnModel(output)
        return output

class Classifier2(nn.Module):
    def __init__(self, out_dim):
        super(Classifier2,self).__init__()
        self.cnnModel = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 3
            # nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # flatten
        )

        self.dnnModel = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, out_dim)
        )

    def forward(self,x):
        output = self.cnnModel(x)
        output = output.squeeze()
        output = self.dnnModel(output)
        return output


class Classifier3(nn.Module):
    def __init__(self, out_dim):
        super(Classifier3,self).__init__()

        self.dnnModel = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, out_dim)
        )

    def forward(self,x):
        output = self.dnnModel(x)
        return output



class Discriminator1(nn.Module):
    def __init__(self):
        super(Discriminator1, self).__init__()
        net = []
        bn = False
        net += [nn.Conv2d(64, 128, 3, 2, 1)]
        net += [nn.ReLU()]
        net += [nn.Conv2d(128, 256, 3, 2, 1)]
        net += [ResBlock(256, 256, bn=bn)]
        #net += [ResBlock(256, 256, bn=bn)]
        #net += [ResBlock(256, 256, bn=bn)]

        net += [nn.Conv2d(256, 256, 3, 2, 1)]
        net += [nn.Flatten()]
        net += [nn.Linear(1024, 1)]

        self.Model = nn.Sequential(*net)

    def forward(self,x):
        output = self.Model(x)
        return output

class Discriminator2(nn.Module):
    def __init__(self):
        super(Discriminator2, self).__init__()
        net = []
        bn = False
        net += [nn.Conv2d(128, 256, 3, 2, 1)]
        net += [ResBlock(256, 256, bn=bn)]
        #net += [ResBlock(256, 256, bn=bn)]
        #net += [ResBlock(256, 256, bn=bn)]

        net += [nn.Conv2d(256, 256, 3, 2, 1)]
        net += [nn.Flatten()]
        net += [nn.Linear(1024, 1)]

        self.Model = nn.Sequential(*net)

    def forward(self,x):
        output = self.Model(x)
        return output


class Discriminator3(nn.Module):
    def __init__(self):
        super(Discriminator3, self).__init__()
        net = []
        bn = False
        #net += [nn.ConvTranspose2d(256, 128, 3)]
        #net += [nn.ConvTranspose2d(128, 128, 3)]
        #net += [nn.ConvTranspose2d(128, 128, 5)]
        net += [nn.ReLU()]
        net += [ResBlock_L(256, 256, bn=bn)]
        net += [ResBlock_L(256, 256, bn=bn)]
        net += [ResBlock_L(256, 256, bn=bn)]
        #net += [nn.Conv2d(128, 128, 3, 2, 1)]
        #net += [nn.Flatten()]

        net += [nn.Linear(256, 1)]

        self.Model = nn.Sequential(*net)

    def forward(self,x):
        #x = x.reshape((200,256,1,1))
        output = self.Model(x)
        return output



class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Decoder, self).__init__()
        print("input shape: ",in_dim)
        net = []
        net += [nn.Flatten()]
        #net += [nn.BatchNorm1d(in_dim)]
        #net += [nn.Linear(in_dim, 128)]
        #net += [nn.LeakyReLU()]
        net += [nn.Linear(in_dim, out_dim)]

        self.Model = nn.Sequential(*net)

    def forward(self,x):
        #print("x: ", x.shape)
        output = self.Model(x)
        return output