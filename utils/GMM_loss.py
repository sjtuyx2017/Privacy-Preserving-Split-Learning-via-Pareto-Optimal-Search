import torch
from torch import nn
import time
import numpy as np
from sklearn.utils.extmath import randomized_svd
from image.utils.tools import kldiv
from torch.linalg import svd

def PCA(X, k):
    X_mean = torch.mean(X, 0)
    X = X - X_mean.expand_as(X)
    U, S, V = torch.svd(torch.t(X))
    #print("S: ",S)
    result = torch.mm(X, U[:,:k])
    return result


# GMM loss with dimension reduction
class GMMLoss(nn.Module):
    def __init__(self, sigma, PCA=0, pca_dim=256):
        super(GMMLoss, self).__init__()
        self.pca_dim = pca_dim
        self.sigma = sigma
        self.PCA = PCA

    def forward(self, means, private_labels):
        device = means.device
        #start = time.time()
        means = means.flatten(start_dim=1)
        #s = time.time()
        if self.PCA:
            means = PCA(means, self.pca_dim)
        #print("pca time: ",time.time()-s)
        num, dim = means.shape[0], means.shape[1]

        # get private label types, indexes and counts
        keys, key_idx, key_counts = torch.unique(private_labels, sorted=True, return_counts=True, return_inverse=True)
        key_idx, key_counts = key_idx.to(device), key_counts.to(device)
        key_num = len(keys)

        # all the sub-Gaussian distributions share the same covariance matrix
        cov = (torch.eye(dim) * self.sigma).to(device)

        # store the final result
        result = torch.zeros(1, device=device)

        # store the mean and covariance matrix of each private attribute group
        group_mean = torch.zeros((key_num, dim), device=device)
        group_cov = torch.zeros((key_num, dim, dim), device=device)

        # add means of the sub-distributions to the corresponding group
        group_mean.scatter_add_(0, key_idx.unsqueeze(-1).expand(num, dim), means)
        group_mean /= key_counts.unsqueeze(-1)

        centered = means - group_mean[key_idx]
        #centered = centered.to(device)
        sigmas = torch.bmm(centered.unsqueeze(-1), centered.unsqueeze(-2))
        # print("sigmas shape: ",sigmas.shape)

        # calculate the covariance of the new group
        #sigmas = sigmas.to('cpu')
        group_cov.scatter_add_(0, key_idx.unsqueeze(-1).unsqueeze(-1).expand(num, dim, dim), sigmas)
        group_cov /= key_counts.unsqueeze(-1).unsqueeze(-1)
        group_cov += cov


        for i in range(key_num):
            for j in range(key_num):
                if j == i:
                    continue
                u1, s1 = group_mean[keys[i]], group_cov[keys[i]]
                u2, s2 = group_mean[keys[j]], group_cov[keys[j]]
                result = result + (kldiv(u1, s1, u2, s2) * key_counts[i] * key_counts[j])


        #end = time.time()
        #print("total time: ", end - start)
        return result / (num ** 3)


class GMMLoss_pca(nn.Module):
    def __init__(self, sigma,pca, dim=256):
        super(GMMLoss_pca, self).__init__()
        self.dim = dim
        self.sigma = sigma
        self.pca = pca

    def forward(self, mu, private_label):
        device = mu.device
        mu = mu.flatten(start_dim=1)
        #print(mu.shape)
        s = time.time()
        if self.pca:
            mu = PCA(mu, self.dim)
        # mu = truncated_svd(mu, self.dim)
        e = time.time()
        # print("PCA time: ",e-s)
        #print(mu.shape)
        mus = torch.split(mu, 1, dim=0)
        batchsize = private_label.shape[0]
        dim = mu.shape[1]
        cov = (torch.eye(dim) * self.sigma).to(device)

        keys = []
        mu_Fs = {}
        count_Fs = {}
        Sigma_Fs = {}
        #Sigma_a = torch.eye(k).float().to(device=self.device) * self.sigma

        for i in range(batchsize):
            label = float(private_label[i])
            if label in mu_Fs:
                mu_Fs[label] = mu_Fs[label] + mus[i]
                count_Fs[label] += 1
            else:
                mu_Fs[label] = mus[i]
                count_Fs[label] = 1

        for i in range(batchsize):
            label = float(private_label[i])
            mu_f = mu_Fs[label] / count_Fs[label]
            if label in Sigma_Fs:
                Sigma_Fs[label] = Sigma_Fs[label] + torch.mm((mus[i] - mu_f).t(), (mus[i] - mu_f))
            else:
                Sigma_Fs[label] = torch.mm((mus[i] - mu_f).t(), (mus[i] - mu_f))

        result = torch.tensor(0.0).to(device=device)

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

        return result / (batchsize ** 3)