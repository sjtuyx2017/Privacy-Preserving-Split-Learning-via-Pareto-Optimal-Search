import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Adult_Encoder(nn.Module) :
    def __init__(self,in_dim) :
        super(Adult_Encoder, self).__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 32)
                                 )

    def forward(self, x):
        out = self.net(x)
        return out


class Info_Encoder_a(nn.Module) :
    def __init__(self,in_dim) :
        super(Info_Encoder_a, self).__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 64),
                                 nn.ReLU()
                                 )

        self.mu = nn.Linear(64, 32)
        self.logvar = nn.Linear(64, 32)

    def forward(self, input):
        encoding = self.net(input)
        mu = self.mu(encoding)

        logvar = self.logvar(encoding)
        sigma = torch.exp(0.5 * logvar)

        return mu, sigma



class Classifier(nn.Module) :
    def __init__(self,num_classes=2) :
        super(Classifier, self).__init__()
        self.net = nn.Sequential(
                                 nn.Linear(32, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, num_classes)
                                )

    def forward(self, x):
        out = self.net(x)
        return out


class Decoder(nn.Module) :
    def __init__(self,num_classes):
        super(Decoder, self).__init__()
        self.net = nn.Sequential(
                                 nn.Linear(32, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 32),
                                 nn.ReLU(),
                                 nn.Linear(32, num_classes)
                                 )

    def forward(self, x):
        out = self.net(x)
        return out


class discriminator(nn.Module) :
    def __init__(self) :
        super(discriminator, self).__init__()
        self.net = nn.Sequential(nn.Linear(32, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, 64),
                                 nn.LeakyReLU(),
                                 # nn.Linear(64, 64),
                                 # nn.LeakyReLU(),
                                 )
        self.logits = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.net(x)
        return self.sigmoid(self.logits(out))


class f_tilde(nn.Module) :
    def __init__(self) :
        super(f_tilde, self).__init__()
        self.net = nn.Sequential(nn.Linear(100, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, 32),
                                 nn.LeakyReLU()
                                 )

    def forward(self, x):
        out = self.net(x)
        return out


class Decoder2(nn.Module) :
    def __init__(self):
        super(Decoder2, self).__init__()
        self.net = nn.Sequential(
                                 nn.Linear(32, 64),
                                 nn.LeakyReLU(),
                                 # nn.Linear(64, 64),
                                 # nn.ReLU(),
                                 nn.Linear(64, 2)
                                 )

    def forward(self, x):
        out = self.net(x)
        return out
