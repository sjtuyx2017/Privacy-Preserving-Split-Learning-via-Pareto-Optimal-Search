import torch
from torch import nn
import time
from utils.tools import kldiv

class GMMLoss(nn.Module):
    def __init__(self, sigma, device):
        super(GMMLoss, self).__init__()
        self.device = device
        self.sigma = sigma

    def forward(self, mu, private_label):
        print("mu shape: ",mu.shape)
        print("label shape: ",private_label.shape)
        start = time.time()
        s = time.time()
        mus = torch.split(mu, 1, dim=0)
        batch_size, k = mu.shape[0], mu.shape[1]
        #k = mus[0].shape[1]
        s1 = time.time()
        cov = (torch.eye(k) * self.sigma).to(self.device)
        result = torch.tensor(0.0).to(device=self.device)
        print("move time: ",time.time()-s1)

        keys = []
        mu_Fs = {}
        count_Fs = {}
        Sigma_Fs = {}
        #Sigma_a = torch.eye(k).float().to(device=self.device) * self.sigma
        print("other time: ",time.time()-s)

        s = time.time()
        for i in range(batch_size):
            label = float(private_label[i])
            if label in mu_Fs:
                mu_Fs[label] = mu_Fs[label] + mus[i]
                count_Fs[label] += 1
            else:
                mu_Fs[label] = mus[i]
                count_Fs[label] = 1
        print("first loop: ", time.time() - s)

        for key in mu_Fs:
            mu_Fs[key] = mu_Fs[key] / count_Fs[key]

        s = time.time()
        for i in range(batch_size):
            label = float(private_label[i])
            #mu_f = mu_Fs[label] / count_Fs[label]
            mu_f = mu_Fs[label]
            if label in Sigma_Fs:
                Sigma_Fs[label] = Sigma_Fs[label] + torch.mm((mus[i] - mu_f).t(), (mus[i] - mu_f))
            else:
                Sigma_Fs[label] = torch.mm((mus[i] - mu_f).t(), (mus[i] - mu_f))
        print("second loop: ", time.time() - s)

        s = time.time()
        for key in mu_Fs:
            Sigma_Fs[key] = cov + Sigma_Fs[key] / count_Fs[key]
            keys.append(key)
        print("third loop: ", time.time() - s)

        # 0.02
        # print(len(keys))
        # kltime = datetime.datetime.now()
        s = time.time()
        for i in range(len(keys)):
            s1, u1 = Sigma_Fs[keys[i]], mu_Fs[keys[i]]
            for j in range(len(keys)):
                if j != i:
                    s2, u2 = Sigma_Fs[keys[j]], mu_Fs[keys[j]]
                    result = result + (kldiv(u1, s1, u2, s2)[0] * count_Fs[keys[i]] * count_Fs[keys[j]])
        print("last loop: ", time.time() - s)
        # kltime = datetime.datetime.now() - kltime
        # print(kltime)
        # 0.4-0.8
        end = time.time()
        print("total time: ", end - start)

        return result / (batch_size ** 3)

class GMMLoss_fast(nn.Module):
    def __init__(self, sigma, device):
        super(GMMLoss_fast, self).__init__()
        self.device = device
        self.sigma = sigma

    def forward(self, mu, private_label):
        batch_size, k = mu.shape[0], mu.shape[1]
        keys = []
        mu_Fs = {}
        Sigma_Fs = {}

        keys, key_counts = torch.unique(private_label, return_counts=True)
        key_num = keys.shape[0]


def PCA(X, k):
    X_mean = torch.mean(X, 0)
    X = X - X_mean.expand_as(X)
    U, S, V = torch.svd(torch.t(X))
    #print("U shape: ",U.shape)
    result = torch.mm(X, U[:,:k])
    return result

# GMM loss with dimension reduction
class GMMLoss_pca(nn.Module):
    def __init__(self, sigma, dim, device):
        super(GMMLoss_pca, self).__init__()
        self.dim = dim
        self.device = device
        self.cov = (torch.eye(self.dim)*sigma).to(device)

    def forward(self, mu, private_attribute):
        #start = time.time()
        mu = mu.flatten(start_dim=1)
        #s = time.time()
        mu = PCA(mu, self.dim)
        #print("pca time: ",time.time()-s)
        mus = torch.split(mu, 1, dim=0)
        batchsize = private_attribute.shape[0]

        keys = []
        mu_Fs = {}
        count_Fs = {}
        Sigma_Fs = {}
        #Sigma_a = torch.eye(k).float().to(device=self.device) * self.sigma

        #s = time.time()
        for i in range(batchsize):
            label = float(private_attribute[i])
            if label in mu_Fs:
                mu_Fs[label] = mu_Fs[label] + mus[i]
                count_Fs[label] += 1
            else:
                mu_Fs[label] = mus[i]
                count_Fs[label] = 1
        #print("first loop: ",time.time()-s)

        for key in mu_Fs:
            mu_Fs[key] = mu_Fs[key] / count_Fs[key]

        #s = time.time()
        for i in range(batchsize):
            label = float(private_attribute[i])
            mu_f = mu_Fs[label]
            diff = (mus[i] - mu_f)
            if label in Sigma_Fs:
                Sigma_Fs[label] = Sigma_Fs[label] + torch.mm(diff.t(), diff)
            else:
                Sigma_Fs[label] = torch.mm(diff.t(), diff)
        #print("second loop: ", time.time() - s)

        result = torch.tensor(0.0).to(device=self.device)

        #s = time.time()
        for key in mu_Fs:
            #mu_Fs[key] = mu_Fs[key] / count_Fs[key]
            Sigma_Fs[key] = self.cov + Sigma_Fs[key] / count_Fs[key]
            keys.append(key)
        #print("third loop: ", time.time() - s)

        # 0.02
        # print(len(keys))
        # kltime = datetime.datetime.now()
        #s = time.time()
        for i in range(len(keys)):
            s1, u1 = Sigma_Fs[keys[i]], mu_Fs[keys[i]]
            for j in range(len(keys)):
                if j != i:
                    s2, u2 = Sigma_Fs[keys[j]], mu_Fs[keys[j]]
                    result = result + (kldiv(u1, s1, u2, s2)[0] * count_Fs[keys[i]] * count_Fs[keys[j]])
        #print("last loop: ", time.time() - s)

        #end = time.time()
        #print("total time: ", end - start)
        return result / (batchsize ** 3)

# each Gaussian distribution approaches to the total distribution
class GMMLoss2_pca(nn.Module):
    def __init__(self, sigma, dim, device):
        super(GMMLoss2_pca, self).__init__()
        self.dim = dim
        self.device = device
        self.cov = (torch.eye(self.dim)*sigma).to(device)

    def forward(self, mu, private_label):
        start = time.time()
        mu = mu.flatten(start_dim=1)
        # print(mu.shape)
        s = time.time()
        mu = PCA(mu, self.dim)
        print("pca time: ", time.time() - s)
        mus = torch.split(mu, 1, dim=0)
        batch_size = private_label.shape[0]
        k = self.dim
        #print("k: ",k)

        keys = []
        mu_Fs = {}
        count_Fs = {}
        Sigma_Fs = {}
        #Sigma_a = torch.eye(k).float().to(device=self.device) * self.sigma

        mu_total = torch.zeros(k).to(self.device)
        Sigma_total = torch.zeros((k,k)).to(self.device)

        s = time.time()
        for i in range(batch_size):
            label = float(private_label[i])
            mu_total = mu_total + mus[i]
            if label in mu_Fs:
                mu_Fs[label] = mu_Fs[label] + mus[i]
                count_Fs[label] += 1
            else:
                mu_Fs[label] = mus[i]
                count_Fs[label] = 1
        print("first loop: ", time.time() - s)

        mu_total = mu_total / batch_size
        for key in mu_Fs:
            mu_Fs[key] = mu_Fs[key] / count_Fs[key]

        s = time.time()
        for i in range(batch_size):
            label = float(private_label[i])
            mu_f = mu_Fs[label]
            diff = (mus[i] - mu_total)
            Sigma_total = Sigma_total + torch.mm(diff.t(), diff)
            if label in Sigma_Fs:
                Sigma_Fs[label] = Sigma_Fs[label] + torch.mm(diff.t(), diff)
            else:
                Sigma_Fs[label] = torch.mm(diff.t(), diff)
        print("second loop: ", time.time() - s)

        Sigma_total = self.cov + Sigma_total / batch_size
        result = torch.tensor(0.0).to(device=self.device)

        s = time.time()
        for key in mu_Fs:
            #mu_Fs[key] = mu_Fs[key] / count_Fs[key]
            Sigma_Fs[key] = self.cov + Sigma_Fs[key] / count_Fs[key]
            keys.append(key)
        print("third loop: ", time.time() - s)

        s1 = time.time()
        for i in range(len(keys)):
            s, u = Sigma_Fs[keys[i]], mu_Fs[keys[i]]
            result = result + (kldiv(u, s, mu_total, Sigma_total)[0] )
        print("last loop: ", time.time() - s1)
        # kltime = datetime.datetime.now() - kltime
        # print(kltime)
        # 0.4-0.8
        end = time.time()
        print("total time: ",end-start)
        return result / (len(keys)*batch_size)


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

        for i in range(len(keys)):
            s, u = Sigma_Fs[keys[i]], mu_Fs[keys[i]]
            result = result + (kldiv(u, s, mu_total, Sigma_total)[0] * count_Fs[keys[i]])

        # kltime = datetime.datetime.now() - kltime
        # print(kltime)
        # 0.4-0.8

        return result / (batch_size)

