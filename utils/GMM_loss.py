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
        for i in range(len(keys) - 1):
            s1, u1 = Sigma_Fs[keys[i]], mu_Fs[keys[i]]
            for j in range(1, len(keys)):
                if j != i:
                    s2, u2 = Sigma_Fs[keys[j]], mu_Fs[keys[j]]
                    result = result + (kldiv(u1, s1, u2, s2)[0] * count_Fs[keys[i]] * count_Fs[keys[j]])

        # kltime = datetime.datetime.now() - kltime
        # print(kltime)
        # 0.4-0.8

        return result / (batch_size ** 2)

class Multi_Dim_GMMLoss(nn.Module):
    def __init__(self, feature_shape, sigma, device):
        super(Multi_Dim_GMMLoss, self).__init__()
        self.cov = (torch.eye(feature_shape[2]*feature_shape[3]) * sigma).to(device)
        self.feature_shape = feature_shape
        self.device = device

    def forward(self, mu, private_attribute):
        #print("shape1: ",mu.shape)
        mu = mu.reshape((self.feature_shape[0], self.feature_shape[1], self.feature_shape[2]*self.feature_shape[3]))
        #print("shape2: ",mu.shape)
        mus = torch.split(mu, 1, dim=0)
        #print("shape3: ",mus)
        batch_size = private_attribute.shape[0]
        k = mus[0][0].shape[1]
        final_result = torch.tensor(0.0).to(device=self.device)
        total_dim = self.feature_shape[1]
        print(mus[0].shape)

        for dim in range(total_dim):
            keys = []
            mu_Fs = {}
            count_Fs = {}
            Sigma_Fs = {}
            # Sigma_a = torch.eye(k).float().to(device=self.device) * self.sigma

            for i in range(batch_size):
                label = float(private_attribute[i])
                item = mus[i].squeeze()[dim].reshape((1,256))
                #print("item shape: ",item.shape)
                if label in mu_Fs:
                    mu_Fs[label] = mu_Fs[label] + item
                    count_Fs[label] += 1
                else:
                    mu_Fs[label] = item
                    count_Fs[label] = 1

            for i in range(batch_size):
                label = float(private_attribute[i])
                item = mus[i].squeeze()[dim].reshape((1, 256))
                mu_f = mu_Fs[label] / count_Fs[label]
                if label in Sigma_Fs:
                    Sigma_Fs[label] = Sigma_Fs[label] + torch.mm((item - mu_f).t(), (item - mu_f))
                else:
                    Sigma_Fs[label] = torch.mm((item - mu_f).t(), (item - mu_f))

            result = torch.tensor(0.0).to(device=self.device)

            for key in mu_Fs:
                mu_Fs[key] = mu_Fs[key] / count_Fs[key]
                Sigma_Fs[key] = self.cov + Sigma_Fs[key] / count_Fs[key]
                keys.append(key)

            # 0.02
            # print(len(keys))
            # kltime = datetime.datetime.now()
            for i in range(len(keys) - 1):
                s1, u1 = Sigma_Fs[keys[i]], mu_Fs[keys[i]]
                for j in range(1, len(keys)):
                    if j != i:
                        s2, u2 = Sigma_Fs[keys[j]], mu_Fs[keys[j]]
                        result = result + (kldiv(u1, s1, u2, s2)[0] * count_Fs[keys[i]] * count_Fs[keys[j]])

            final_result += result

        return final_result/total_dim/(batch_size ** 2)

class GMMLoss_pca(nn.Module):
    def __init__(self, feature_shape, sigma, device):
        super(GMMLoss_pca, self).__init__()
        self.device = device
        self.cov = torch.eye(128)*sigma
        self.cov = self.cov.to(self.device)
        self.feature_shape = feature_shape

    def forward(self, mu, private_attribute):
        mu = mu.reshape((self.feature_shape[0], 128, 128))
        mu = torch.pca_lowrank(mu, q=128)[1]
        mus = torch.split(mu, 1, dim=0)
        batch_size = private_attribute.shape[0]
        k = mus[0].shape[1]

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

        result = torch.tensor(0.0).to(device=self.device)

        for key in mu_Fs:
            mu_Fs[key] = mu_Fs[key] / count_Fs[key]
            Sigma_Fs[key] = self.cov + Sigma_Fs[key] / count_Fs[key]
            keys.append(key)

        # 0.02
        # print(len(keys))
        # kltime = datetime.datetime.now()
        for i in range(len(keys) - 1):
            s1, u1 = Sigma_Fs[keys[i]], mu_Fs[keys[i]]
            for j in range(1, len(keys)):
                if j != i:
                    s2, u2 = Sigma_Fs[keys[j]], mu_Fs[keys[j]]
                    result = result + (kldiv(u1, s1, u2, s2)[0] * count_Fs[keys[i]] * count_Fs[keys[j]])

        # kltime = datetime.datetime.now() - kltime
        # print(kltime)
        # 0.4-0.8

        return result / (batch_size ** 2)

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
            result = result + (kldiv(u, s, mu_total, Sigma_total, k)[0] * count_Fs[keys[i]] * batch_size)

        # kltime = datetime.datetime.now() - kltime
        # print(kltime)
        # 0.4-0.8

        return result / (batch_size ** 2)