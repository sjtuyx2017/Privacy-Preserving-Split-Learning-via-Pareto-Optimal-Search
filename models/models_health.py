import torch
from torch import nn
from torch.nn import functional as F

## health:mlp_encoder
class Encoder1(nn.Module):
    def __init__(self, in_dim, out_dim, drop_rate=0.0):
        super(Encoder1, self).__init__()
        n_h = 128
        self.net = nn.Sequential(               # [71, 256, 128, 128]
            nn.Linear(in_dim, n_h*2),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(n_h*2, n_h),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(n_h, out_dim)
        )

    def forward(self, x):
        output = self.net(x)
        return output

class Encoder_Info1(nn.Module):
    def __init__(self, in_dim, out_dim, drop_rate=0.0):
        super(Encoder_Info1, self).__init__()
        n_h = 128
        self.net = nn.Sequential(               # [71, 256, 128, 128]
            nn.Linear(in_dim, n_h*2),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(n_h*2, n_h),
            nn.ReLU(),
            nn.Dropout(drop_rate)
        )
        self.mu = nn.Linear(n_h, out_dim)
        self.logvar = nn.Linear(n_h, out_dim)

    def forward(self, x):
        encoding = self.net(x)
        mu = self.mu(encoding)
        logvar = self.logvar(encoding)
        sigma = torch.exp(0.5 * logvar)

        return mu, sigma



class Classifier1(nn.Module):
    def __init__(self, in_dim, drop_rate=0.0, num_classes=2):
        super(Classifier1, self).__init__()

        self.drop_rate = drop_rate

        self.net = nn.Sequential(           # [128,2]
            nn.Linear(in_dim,num_classes)
        )

    def forward(self, x):
        x = F.dropout(x, p=self.drop_rate, training=self.training)
        output = self.net(x)
        return output   # 返回logits


class Discriminator1(nn.Module):
    def __init__(self, in_dim, hidden_dims=None, bn=True, drop_rate=0.0):
        super(Discriminator1, self).__init__()

        self.drop_rate = drop_rate
        modules = []
        n_h = 128
        if hidden_dims is None:
            hidden_dims = [n_h*2, n_h*2]     #[128,256,256,1]

        hidden_dims = [in_dim] + hidden_dims

        for layer_idx in range(len(hidden_dims)-1):
            if bn:
                modules.append(
                    nn.Sequential(
                        nn.Linear(hidden_dims[layer_idx], hidden_dims[layer_idx+1]),
                        nn.BatchNorm1d(hidden_dims[layer_idx+1]),
                        nn.LeakyReLU(0.2),
                        nn.Dropout(drop_rate))
                )
            else:
                modules.append(
                    nn.Sequential(
                        nn.Linear(hidden_dims[layer_idx], hidden_dims[layer_idx+1]),
                        nn.LeakyReLU(),
                        nn.Dropout(drop_rate))
                )



        self.features = None if len(modules) == 0 else nn.Sequential(*modules)
        self.logits = nn.Linear(hidden_dims[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        features = F.dropout(input, p=self.drop_rate, training=self.training)
        if self.features is not None: features = self.features(features)
        return self.sigmoid(self.logits(features))


class Encoder2(nn.Module):
    def __init__(self, in_dim, out_dim, drop_rate=0.0):
        super(Encoder2, self).__init__()
        n_h = 128
        self.net = nn.Sequential(               # [71, 256, 128]
            nn.Linear(in_dim, n_h*2),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(n_h*2, out_dim)

        )

    def forward(self, x):
        output = self.net(x)
        return output

class Encoder_Info2(nn.Module):
    def __init__(self, in_dim, out_dim, drop_rate=0.0):
        super(Encoder_Info2, self).__init__()
        n_h = 128
        self.net = nn.Sequential(               # [71, 256, 128, 128]
            nn.Linear(in_dim, n_h*2),
            nn.ReLU(),
            nn.Dropout(drop_rate),
        )
        self.mu = nn.Linear(n_h*2, out_dim)
        self.logvar = nn.Linear(n_h*2, out_dim)

    def forward(self, x):
        encoding = self.net(x)
        mu = self.mu(encoding)
        logvar = self.logvar(encoding)
        sigma = torch.exp(0.5 * logvar)

        return mu, sigma


class Classifier2(nn.Module):
    def __init__(self, in_dim, drop_rate=0.0, num_classes=2):
        super(Classifier2, self).__init__()

        self.drop_rate = drop_rate

        self.net = nn.Sequential(           # [128,128,2]
            nn.Linear(in_dim,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = F.dropout(x, p=self.drop_rate, training=self.training)
        output = self.net(x)
        return output   # 返回logits


class Discriminator2(nn.Module):
    def __init__(self, in_dim, drop_rate=0.0):
        super(Discriminator2, self).__init__()

        self.drop_rate = drop_rate

        self.net = nn.Sequential(       # [128,256,256,128,1]
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(128, 1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.net(x)
        return self.sigmoid(x)


class Encoder3(nn.Module):
    def __init__(self, in_dim, out_dim, drop_rate=0.0):
        super(Encoder3, self).__init__()
        n_h = 128
        self.net = nn.Sequential(               # [71, 256]
            nn.Linear(in_dim, n_h*2),
        )

    def forward(self, x):
        output = self.net(x)
        return output

class Classifier3(nn.Module):
    def __init__(self, in_dim, drop_rate=0.0, num_classes=2):
        super(Classifier3, self).__init__()

        self.drop_rate = drop_rate

        self.net = nn.Sequential(           # [256,256,128,2]
            nn.Linear(in_dim,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = F.dropout(x, p=self.drop_rate, training=self.training)
        output = self.net(x)
        return output


class Decoder1(nn.Module):
    def __init__(self, in_dim, drop_rate=0.0, num_classes=2):
        super(Decoder1, self).__init__()

        self.drop_rate = drop_rate
        self.net = nn.Sequential(            # [feature,256,128,sens]
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = F.dropout(x, p=self.drop_rate, training=self.training)
        output = self.net(x)
        return output   # 返回logits

class Decoder2(nn.Module):
    def __init__(self, in_dim, drop_rate=0.0, num_classes=2):
        super(Decoder2, self).__init__()

        self.drop_rate = drop_rate
        self.net = nn.Sequential(            # [feature,256,128,sens]
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = F.dropout(x, p=self.drop_rate, training=self.training)
        output = self.net(x)
        return output   # 返回logits