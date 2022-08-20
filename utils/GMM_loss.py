import torch
import math
from torch import nn
from utils.tools import kldiv

class GMMLoss(nn.Module):
    def __init__(self, sigma, device):
        super(GMMLoss, self).__init__()
        self.device = device
        self.sigma = sigma

    def forward(self, mu, private_attribute):
        #print("mu shape: ",mu.shape)
        mus = torch.split(mu, 1, dim=0)
        batch_size = private_attribute.shape[0]
        k = mus[0].shape[1]
        cov = (torch.eye(k) * self.sigma).to(self.device)
        result = torch.tensor(0.0).to(device=self.device)

        keys = []
        mu_Fs = {}
        count_Fs = {}
        Sigma_Fs = {}
        #Sigma_a = torch.eye(k).float().to(device=self.device) * self.sigma

        for i in range(batch_size):
            label = float(private_attribute[i])
            if label in mu_Fs:
                mu_Fs[label] = mu_Fs[label] + mus[i]
                count_Fs[label] += 1
            else:
                mu_Fs[label] = mus[i]
                count_Fs[label] = 1

        for i in range(batch_size):
            label = float(private_attribute[i])
            mu_f = mu_Fs[label] / count_Fs[label]
            if label in Sigma_Fs:
                Sigma_Fs[label] = Sigma_Fs[label] + torch.mm((mus[i] - mu_f).t(), (mus[i] - mu_f))
            else:
                Sigma_Fs[label] = torch.mm((mus[i] - mu_f).t(), (mus[i] - mu_f))

        for key in mu_Fs:
            mu_Fs[key] = mu_Fs[key] / count_Fs[key]
            Sigma_Fs[key] = cov + Sigma_Fs[key] / count_Fs[key]
            keys.append(key)

        # 0.02
        # print(len(keys))
        # kltime = datetime.datetime.now()
        for i in range(len(keys)):
            s1, u1 = Sigma_Fs[keys[i]], mu_Fs[keys[i]]
            for j in range(len(keys)):
                if j != i:
                    s2, u2 = Sigma_Fs[keys[j]], mu_Fs[keys[j]]
                    result = result + (kldiv(u1, s1, u2, s2)[0] * count_Fs[keys[i]] * count_Fs[keys[j]])

        # kltime = datetime.datetime.now() - kltime
        # print(kltime)
        # 0.4-0.8

        return result / (batch_size ** 2)

def PCA(X, k):
    X_mean = torch.mean(X, 0)
    X = X - X_mean.expand_as(X)
    U, S, V = torch.svd(torch.t(X))
    #print("U shape: ",U.shape)
    result = torch.mm(X, U[:,:k])
    return result

# GMM loss with dimension reduction
class GMMLoss_pca(nn.Module):
    def __init__(self, sigma, device):
        super(GMMLoss_pca, self).__init__()
        self.dim = 200
        self.device = device
        self.cov = (torch.eye(self.dim)*sigma).to(device)

    def forward(self, mu, private_attribute):
        mu = mu.flatten(start_dim=1)
        #print(mu.shape)
        mu = PCA(mu, self.dim)
        #print(mu.shape)
        #mu = torch.pca_lowrank(mu, q=self.dim)[1]
        mus = torch.split(mu, 1, dim=0)
        batchsize = private_attribute.shape[0]
        k = mus[0].shape[1]

        keys = []
        mu_Fs = {}
        count_Fs = {}
        Sigma_Fs = {}
        #Sigma_a = torch.eye(k).float().to(device=self.device) * self.sigma

        for i in range(batchsize):
            label = float(private_attribute[i])
            if label in mu_Fs:
                mu_Fs[label] = mu_Fs[label] + mus[i]
                count_Fs[label] += 1
            else:
                mu_Fs[label] = mus[i]
                count_Fs[label] = 1

        for i in range(batchsize):
            label = float(private_attribute[i])
            mu_f = mu_Fs[label] / count_Fs[label]
            if label in Sigma_Fs:
                Sigma_Fs[label] = Sigma_Fs[label] + torch.mm((mus[i] - mu_f).t(), (mus[i] - mu_f))
            else:
                Sigma_Fs[label] = torch.mm((mus[i] - mu_f).t(), (mus[i] - mu_f))

        result = torch.tensor(0.0).to(device=self.device)

        for key in mu_Fs:
            mu_Fs[key] = mu_Fs[key] / count_Fs[key]
            Sigma_Fs[key] = self.cov + Sigma_Fs[key] / count_Fs[key]
            keys.append(key)

        # 0.02
        # print(len(keys))
        # kltime = datetime.datetime.now()
        for i in range(len(keys)):
            s1, u1 = Sigma_Fs[keys[i]], mu_Fs[keys[i]]
            for j in range(len(keys)):
                if j != i:
                    s2, u2 = Sigma_Fs[keys[j]], mu_Fs[keys[j]]
                    result = result + (kldiv(u1, s1, u2, s2)[0] * count_Fs[keys[i]] * count_Fs[keys[j]])

        # kltime = datetime.datetime.now() - kltime
        # print(kltime)
        # 0.4-0.8

        return result / (batchsize ** 2)


# each Gaussian distribution approaches to the total distribution
class GMMLoss2(nn.Module):
    def __init__(self, cov, device):
        super(GMMLoss2, self).__init__()
        self.device = device
        self.cov = cov.to(self.device)

    def forward(self, mu, private_attribute):
        mus = torch.split(mu, 1, dim=0)
        batch_size = private_attribute.shape[0]
        k = mus[0].shape[1]

        keys = []
        mu_Fs = {}
        count_Fs = {}
        Sigma_Fs = {}
        #Sigma_a = torch.eye(k).float().to(device=self.device) * self.sigma

        mu_total = torch.zeros(k).to(self.device)
        Sigma_total = torch.zeros((k,k)).to(self.device)

        for i in range(batch_size):
            label = float(private_attribute[i])
            mu_total = mu_total + mus[i]
            if label in mu_Fs:
                mu_Fs[label] = mu_Fs[label] + mus[i]
                count_Fs[label] += 1
            else:
                mu_Fs[label] = mus[i]
                count_Fs[label] = 1

        mu_total = mu_total / batch_size

        for i in range(batch_size):
            label = float(private_attribute[i])
            mu_f = mu_Fs[label] / count_Fs[label]
            Sigma_total = Sigma_total + torch.mm((mus[i] - mu_total).t(), (mus[i] - mu_total))
            if label in Sigma_Fs:
                Sigma_Fs[label] = Sigma_Fs[label] + torch.mm((mus[i] - mu_f).t(), (mus[i] - mu_f))
            else:
                Sigma_Fs[label] = torch.mm((mus[i] - mu_f).t(), (mus[i] - mu_f))

        Sigma_total = self.cov + Sigma_total / batch_size
        result = torch.tensor(0.0).to(device=self.device)

        for key in mu_Fs:
            mu_Fs[key] = mu_Fs[key] / count_Fs[key]
            Sigma_Fs[key] = self.cov + Sigma_Fs[key] / count_Fs[key]
            keys.append(key)

        # 0.02
        # print(len(keys))
        # kltime = datetime.datetime.now()
        # for i in range(len(keys) - 1):
        #     s1, u1 = Sigma_Fs[keys[i]], mu_Fs[keys[i]]
        #     for j in range(1, len(keys)):
        #         if j != i:
        #             s2, u2 = Sigma_Fs[keys[j]], mu_Fs[keys[j]]
        #             result = result + (kldiv(u1, s1, u2, s2, k)[0] * count_Fs[keys[i]] * count_Fs[keys[j]])

        for i in range(len(keys) - 1):
            s, u = Sigma_Fs[keys[i]], mu_Fs[keys[i]]
            result = result + (kldiv(u, s, mu_total, Sigma_total)[0] * count_Fs[keys[i]])

        # kltime = datetime.datetime.now() - kltime
        # print(kltime)
        # 0.4-0.8

        return result / (batch_size)