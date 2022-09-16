import numpy as np
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
import os
import torch
import math

# def smooth(a, WSZ):
#     # a:原始数据，NumPy 1-D array containing the data to be smoothed
#     # 必须是1-D的，如果不是，请使用 np.ravel()或者np.squeeze()转化
#     # WSZ: smoothing window size needs, which must be odd number,
#     # as in the original MATLAB implementation
#     out0 = np.convolve(a, np.ones(WSZ, dtype=int), 'valid') / WSZ
#     r = np.arange(1, WSZ - 1, 2)
#     start = np.cumsum(a[:WSZ - 1])[::2] / r
#     stop = (np.cumsum(a[:-WSZ:-1])[::2] / r)[::-1]
#     return np.concatenate((start, out0, stop))

def smooth(array, window_sz):
    return uniform_filter1d(array, window_sz, mode='nearest')

def makedir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def plot_and_save_figure(x_list, y_list, x_label, y_label, figure_label, path, figure_name):
    plt.figure()
    if len(y_list) != 0:
        plt.plot(x_list, y_list, label=figure_label)
    else:
        plt.plot(x_list, label=figure_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(os.path.join(path, figure_name))
    plt.close()

def getNumParams(params):
    numParams, numTrainable = 0, 0
    for param in params:
        npParamCount = np.prod(param.data.shape)
        numParams += npParamCount
        if param.requires_grad:
            numTrainable += npParamCount
    return numParams, numTrainable

# def get_gaussian_md(mu,sigma,device):
#     feature_shape = mu.shape
#     dim2 = int(math.prod(feature_shape) / feature_shape[0])
#     mu = mu.reshape((feature_shape[0],dim2 ))
#     cov = torch.eye(dim2)
#     if sigma == 0:
#         return mu
#     fx, fy = mu.shape
#     # feature1 = feature.cpu().detach().numpy()
#     temp = np.random.multivariate_normal([0 for i in range(fy)], cov, fx)
#     # Data are generated according to covariance matrix and mean
#     # for i in range(1, fx):
#     # temp = np.concatenate((temp, np.random.multivariate_normal([0 for i in range(fy)], self.cov)), axis=0)
#     # Splicing sampling of high dimensional Gaussian distribution data
#     temp.resize((fx, fy))
#     temp = torch.from_numpy(temp).float()
#     # Since the stitched data is one-dimensional,
#     # we redefine it as the original dimension
#     feature = mu + temp.to(device) * (sigma ** 0.5)
#     return feature

# def get_gaussian(mu,sigma,device):
#     #print("mu shape: ", mu.shape)
#     cov = torch.eye(mu.shape[1])
#     if sigma == 0:
#         return mu
#     fx, fy = mu.shape
#     # feature1 = feature.cpu().detach().numpy()
#     temp = np.random.multivariate_normal([0 for i in range(fy)], cov, fx)
#     # Data are generated according to covariance matrix and mean
#     # for i in range(1, fx):
#     # temp = np.concatenate((temp, np.random.multivariate_normal([0 for i in range(fy)], self.cov)), axis=0)
#     # Splicing sampling of high dimensional Gaussian distribution data
#     temp.resize((fx, fy))
#     temp = torch.from_numpy(temp).float()
#     # Since the stitched data is one-dimensional,
#     # we redefine it as the original dimension
#     feature = mu + temp.to(device) * (sigma ** 0.5)
#     return feature

def get_gaussian(mu, sigma):
    #print("mu shape: ", mu.shape)
    #cov = torch.eye(mu.shape[1])
    if sigma == 0:
        return mu

    return mu + torch.randn_like(mu)* (sigma ** 0.5)


def kldiv(u1, s1, u2, s2):
    p_distribution = torch.distributions.MultivariateNormal(u1, s1)
    q_distribution = torch.distributions.MultivariateNormal(u2, s2)
    p_q_kl = torch.distributions.kl_divergence(p_distribution, q_distribution)
    return p_q_kl