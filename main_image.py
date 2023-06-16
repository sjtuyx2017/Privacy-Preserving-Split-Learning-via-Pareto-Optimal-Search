# In the training process, both DSA and DPA are used
import os
import numpy as np
from torch import nn, optim
import torch
from process_data.load_image import load_main_dataset, load_attacker_dataset
from sklearn.metrics import confusion_matrix
from math import sqrt
import time
import logging
# different model partitions
from models.models_image import (
    Encoder1, Classifier1, Discriminator1, InfocensorEncoder1,
    Encoder2, Classifier2, Discriminator2, InfocensorEncoder2,
    Encoder3, Classifier3, Discriminator3, InfocensorEncoder3,
    Decoder
)
import matplotlib.pyplot as plt
import tqdm
from utils.trace_QP import EPO_LP_Trace
from utils.tools import smooth, makedir, plot_and_save_figure, getNumParams, get_gaussian
from utils.GMM_loss import GMMLoss, GMMLoss_pca
from utils.gradient_operations import save_and_clear_grad, reattach_grad, merge_gradients, zeroing_grad
from infocensor_utils.utils import (
    variational_mutual_information_estimator,
    information_bottleneck,
    fair_pred_mutual_information
)
import argparse

plt.switch_backend('Agg')

logger = logging.getLogger(__name__)

# basic parameters
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='celebA', choices=['celebA', 'UTKFace'], type=str)
parser.add_argument('--utility_task', default='Smiling', type=str)
parser.add_argument('--privacy_task', default='Attractive', type=str)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--batch_size', default=500, type=int)
parser.add_argument('--iterations', default=2000, type=int)
parser.add_argument('--defense', default='GMM-LC', choices=['GMM-LC', 'GMM-EPO', 'GMM-RO', 'ADV', 'Noisy', 'Infocensor', 'None'], type=str)
parser.add_argument('--model_idx', default=2, choices=[1, 2, 3], type=int)  # different model partitions :
# 1: local owns 2 convolutional layers
# 2: local owns 3 convolutional layers
# 3: local owns all the 4 convolutional layers
parser.add_argument('--seed', default=10, type=int)

# the number of iterations to test results and save model
parser.add_argument('--record_interval', default=10, type=int)
parser.add_argument('--save_model', default=1, type=bool)

# Data poison attack parameter
parser.add_argument('--poison_ratio', default=0.05, type=float)

# Adversarial training parameters
parser.add_argument('--adv_weight', default=3, type=float) # encoder_gradient = top_gradient - adv_weight * decoder_gradient
parser.add_argument('--flip_label', default=0, type=int)

# Noisy parameter
parser.add_argument('--noise_sigma', default=10, type=float)   # deciding the variance of the noise

# Infocensor parameter
parser.add_argument('--info_lambda', default=0.35, type=float)
parser.add_argument('--info_beta', default=0.0, type=float)
parser.add_argument('--fair_kappa', default=0.5, type=float)

# GMM parameters
parser.add_argument('--loss_type', default=1, type=int)
parser.add_argument('--sigma', default=10, type=float)   # deciding the variance of each Gaussian distribution
parser.add_argument('--pca_dim', default=256, type=int)

# GMM-LC parameters
parser.add_argument('--acc_weight', default=0.5, type=float)

# GMM-RO parameter
parser.add_argument('--warmup_iterations', default=500, type=float)
parser.add_argument('--privacy_constraint', default=0.003, type=float)

# GMM-EPO parameters, these parameters take effect only when the defense method is GMM-EPO
# parser.add_argument('--preference', default=1, type=float) # preference for accuracy task
# parser.add_argument('--eps', default=1e-2, type=float) # EPO non-uniformity restriction

args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# create results saving path
directory = './%s_results'%(args.dataset)
makedir(directory)

#SavePath = directory
SavePath = directory + '/utility=%s_privacy=%s'%(
    args.utility_task, args.privacy_task)
makedir(SavePath)
SavePath = SavePath + '/model-idx=%s'%(str(args.model_idx))
makedir(SavePath)
SavePath = SavePath + '/seed=%s_pr=%s_defense=%s_iter=%s_bs=%s'%(
    str(args.seed), str(args.poison_ratio), args.defense, str(args.iterations), str(args.batch_size))

if args.defense == 'GMM-LC':
    SavePath = SavePath + '_loss=%s_sigma=%s_acc-w=%s' % (str(args.loss_type), str(args.sigma), str(args.acc_weight))
elif args.defense == 'GMM-EPO':
    SavePath = SavePath + '_loss=%s_sigma=%s_pref=%s_eps=%s' %(str(args.loss_type), str(args.sigma),
        str(args.preference), str(args.eps))
elif args.defense == 'GMM-RO':
    SavePath = SavePath + '_loss=%s_sigma=%s_wi=%s_pc=%s' % (str(args.loss_type), str(args.sigma),
        str(args.warmup_iterations), str(args.privacy_constraint))
elif args.defense == 'ADV':
    SavePath = SavePath + '_adv-w=%s_flip=%s' % (str(args.adv_weight), str(args.flip_label))
elif args.defense == 'Noisy':
    SavePath = SavePath + '_noise-sigma=%s' % (str(args.noise_sigma))
elif args.defense == 'Infocensor':
    SavePath = SavePath + '_lambda=%s_beta=%s_kappa=%s' % (str(args.info_lambda), str(args.info_beta), str(args.fair_kappa))

makedir(SavePath)

modelSavePath = SavePath + '/models'
makedir(modelSavePath)
figureSavePath = SavePath + '/figures'
makedir(figureSavePath)
accSavePath = figureSavePath + '/acc_loss'
makedir(accSavePath)
priSavePath = figureSavePath + '/pri_loss'
makedir(priSavePath)
DSASavePath = figureSavePath + '/DSA_loss'
makedir(DSASavePath)
DPASavePath = figureSavePath + '/DPA_loss'
makedir(DPASavePath)
logSavePath = SavePath + '/logs'
makedir(logSavePath)
log_file = logSavePath + '/log.txt'
result_file = logSavePath + '/result.txt'
if os.path.exists(log_file):
    os.remove(log_file)
if os.path.exists(result_file):
    os.remove(result_file)


class Client(nn.Module):
    def __init__(self, encoder, adv_decoder):
        super(Client, self).__init__()
        self.encoder = encoder.to(device)
        self.adv_decoder = adv_decoder.to(device)
        if args.loss_type == 1:
            self.privacy_criterion = GMMLoss_pca(args.sigma, 1, args.pca_dim)
        # elif args.loss_type == 2:
        #     self.privacy_criterion = GMMLoss2_pca(args.sigma, args.pca_dim)
        if args.model_idx == 3:
            self.privacy_criterion = GMMLoss_pca(args.sigma, 0)
        self.accuracy_criterion = torch.nn.CrossEntropyLoss()
        self.optimizer_decoder = optim.Adam(self.adv_decoder.parameters(), lr=args.lr)
        self.optimizer_encoder = optim.Adam(self.encoder.parameters(), lr=args.lr)
        _, n_params = getNumParams(self.encoder.parameters())
        self.n_params = n_params
        # self.preference = np.array([args.preference, 1-args.preference])
        self.mu_list = []
        if args.defense == 'GMM-EPO':
            # self.epo_lp = EPO_LP(m=2, n=n_params, r=self.preference,eps=args.eps)
            print("1")
        elif args.defense == 'GMM-RO':
            self.epo_lp = EPO_LP_Trace(m=2, n=n_params, privacy_constraint=args.privacy_constraint)

    def warm_up(self, x_private, private_label):
        self.encoder.train()
        x_private, private_label = x_private.to(device), private_label.to(device)
        mu_private = self.encoder(x_private)
        #z_private = get_gaussian(mu_private, args.sigma)
        # s = time.time()
        pri_loss = self.privacy_criterion(mu_private, private_label)
        self.optimizer_encoder.zero_grad()
        pri_loss.backward()
        self.optimizer_encoder.step()
        return pri_loss.item()

    def get_feature_no_defense(self, x_private, private_label=None, training=0):
        if training:
            self.encoder.train()
        else:
            self.encoder.eval()
        x_private = x_private.to(device)
        z_private = self.encoder(x_private)
        self.optimizer_encoder.zero_grad()

        return z_private

    def get_feature_noisy(self, x_private, private_label=None, training=0):
        if training:
            self.encoder.train()
        else:
            self.encoder.eval()
        x_private = x_private.to(device)
        z_private = self.encoder(x_private)
        z_private = get_gaussian(z_private, args.noise_sigma)
        self.optimizer_encoder.zero_grad()

        return z_private

    def get_feature_ADV(self, x_private, private_label=None, training=0):
        if training:
            self.encoder.train()
            self.adv_decoder.train()
            x_private, private_label = x_private.to(device), private_label.to(device)
            z_private = self.encoder(x_private)
            decoder_outputs = self.adv_decoder(z_private)
            _, predicted = decoder_outputs.max(1)
            opposite_label = 1-predicted
            adv_loss = self.accuracy_criterion(decoder_outputs, private_label.long())
            if args.flip_label==1:
                adv_loss_encoder = self.accuracy_criterion(decoder_outputs, opposite_label)
                self.optimizer_decoder.zero_grad()
                self.optimizer_encoder.zero_grad()
                adv_loss_encoder.backward(retain_graph=True)
                self.decoder_gradient, self.flatten_dg = save_and_clear_grad(self.encoder)

                self.optimizer_decoder.zero_grad()
                adv_loss.backward(retain_graph=True)
                self.optimizer_decoder.step()
                self.optimizer_encoder.zero_grad()
            else:
                self.optimizer_encoder.zero_grad()
                self.optimizer_decoder.zero_grad()
                adv_loss.backward(retain_graph=True)
                self.decoder_gradient, self.flatten_dg = save_and_clear_grad(self.encoder)
                self.optimizer_decoder.step()
                #self.optimizer_encoder.zero_grad()

            return z_private, adv_loss.item()
        else:
            self.encoder.eval()
            x_private = x_private.to(device)
            z_private = self.encoder(x_private)
            self.optimizer_encoder.zero_grad()

            return z_private

    def get_feature_info(self, x_private, private_label=None, training=0):
        if training:
            self.encoder.train()
            x_private, private_label = x_private.to(device), private_label.to(device)
            mu_private, sigma_private = self.encoder(x_private)
            z_private = mu_private + sigma_private * torch.randn_like(sigma_private)
            #print("z shape: ",z_private.shape)
            mu_flatten = mu_private.flatten(start_dim=1)
            sigma_flatten = sigma_private.flatten(start_dim=1)
            sensitive_mutual_info = variational_mutual_information_estimator(mu_flatten, sigma_flatten,
                                                                             private_label, private_label_num)
            pri_loss = args.info_lambda * sensitive_mutual_info +  \
                       args.info_beta*information_bottleneck(mu_flatten, sigma_flatten)
            self.optimizer_encoder.zero_grad()
            pri_loss.backward(retain_graph=True)
            self.privacy_gradient, self.flatten_pg = save_and_clear_grad(self.encoder)

            return z_private, pri_loss.item()
        else:
            self.encoder.eval()
            x_private = x_private.to(device)
            mu_private, sigma_private = self.encoder(x_private)
            z_private = mu_private + sigma_private * torch.randn_like(sigma_private)

            return z_private


    def get_feature_GMM(self, x_private, private_label=None, training=0, batch_idx=0):
        if training:
            self.encoder.train()
            x_private = x_private.to(device)
            mu_private = self.encoder(x_private)
            z_private = get_gaussian(mu_private, args.sigma)
            #s = time.time()
            pri_loss = self.privacy_criterion(mu_private, private_label)
            #print("GMM time: ",time.time()-s)

            self.privacy_loss = pri_loss.item()
            self.optimizer_encoder.zero_grad()
            #s = time.time()
            pri_loss.backward(retain_graph=True)
            #print("backward time: ",time.time()-s)
            #print('\n')
            self.privacy_gradient, self.flatten_pg = save_and_clear_grad(self.encoder)

            return z_private, pri_loss.item()
        else:
            self.encoder.eval()
            x_private = x_private.to(device)
            mu_private = self.encoder(x_private)
            z_private = get_gaussian(mu_private, args.sigma)
            # pri_loss = self.privacy_criterion(mu_private, private_label)
            #pri_loss = self.privacy_criterion(mu_private, private_label)

            return z_private


    def get_EPO_weight(self, top_loss, mode):
        grads = {}
        losses_vec = []
        losses_vec.append(top_loss)
        losses_vec.append(self.privacy_loss)
        grads[0] = self.flatten_tg
        grads[1] = self.flatten_pg

        grads_list = [torch.cat(grads[i]) for i in range(len(grads))]
        G = torch.stack(grads_list)
        # GG = G @ G.T
        # print("GG shape: ",GG.shape)
        losses_vec = np.stack(losses_vec)
        # calculate the weights
        weights, mu = self.epo_lp.get_alpha(losses_vec, G=G.cpu().numpy(), mode=mode)
        self.mu_list.append(mu)
        if weights is None:  # A patch for the issue in cvxpy
            weights = self.preference / self.preference.sum()
            # n_linscalar_adjusts += 1

        weights = torch.from_numpy(weights).to(device)
        print("weights: ",weights)

        return weights[0]

    def update_no_defense(self):
        self.optimizer_encoder.step()

    def update_noisy(self):
        self.optimizer_encoder.step()

    def update_GMM(self, top_loss, mode):
        self.top_gradient, self.flatten_tg = save_and_clear_grad(self.encoder)
        #self.top_gradient = self.encoder.parameters().grad
        if mode == -1:
            merged_gradient = merge_gradients(self.top_gradient, self.privacy_gradient, [1, 0])
        else:
            if args.defense == 'GMM-LC':
                top_weight = args.acc_weight
            elif args.defense == 'GMM-EPO' or args.defense == 'GMM-RO':
                top_weight = self.get_EPO_weight(top_loss, mode)
            #merged_gradient = merge_gradients(self.top_gradient, self.privacy_gradient, [top_weight, 1-top_weight])
            merged_gradient = merge_gradients(self.top_gradient, self.privacy_gradient, [top_weight, 1-top_weight])
        reattach_grad(self.encoder, merged_gradient)
        #self.encoder.parameters().grad = top_weight * self.top_gradient + (1-top_weight)* self.privacy_gradient
        self.optimizer_encoder.step()

    def update_ADV(self):
        self.top_gradient, self.flatten_tg = save_and_clear_grad(self.encoder)
        #self.optimizer_encoder.zero_grad()
        if args.flip_label:
            merged_gradient = merge_gradients(self.top_gradient, self.decoder_gradient, [1, args.adv_weight])
        else:
            merged_gradient = merge_gradients(self.top_gradient, self.decoder_gradient, [1, -args.adv_weight])
        reattach_grad(self.encoder, merged_gradient)
        self.optimizer_encoder.step()

    def update_info(self, pred, private_label):
        private_label = private_label.to(device)
        self.top_gradient, self.flatten_tg = save_and_clear_grad(self.encoder)
        #privacy_loss2 = args.fair_kappa * fair_pred_mutual_information(pred, private_label, private_label_num)
        #privacy_loss2.backward(retain_graph = True)
        #self.privacy_gradient2, self.flatten_pg2 = save_and_clear_grad(self.encoder)
        #merged_privacy_gradient = merge_gradients(self.privacy_gradient1, self.privacy_gradient2, [1, 1])
        merged_gradient = merge_gradients(self.top_gradient, self.privacy_gradient, [1, 1])
        reattach_grad(self.encoder, merged_gradient)
        self.optimizer_encoder.step()


class Cloud(nn.Module):
    def __init__(self, top_model, DPAdecoder, f_tilde, DSAdecoder, discriminator):
        super(Cloud, self).__init__()
        self.top_model = top_model.to(device)
        self.DPAdecoder = DPAdecoder.to(device)
        self.f_tilde = f_tilde.to(device)
        self.DSAdecoder = DSAdecoder.to(device)
        self.D = discriminator.to(device)
        self.accuracy_criterion = torch.nn.CrossEntropyLoss()
        self.decoder_criterion = torch.nn.CrossEntropyLoss()
        self.optimizer_DPAdecoder = optim.Adam(self.DPAdecoder.parameters(), lr=args.lr)
        self.optimizer_top = optim.Adam(self.top_model.parameters(), lr=args.lr)
        self.optimizer_D = optim.Adam(self.D.parameters(), lr=args.lr)

        # when using DSA, f_tilde and decoder are decoupled and updated separately
        self.optimizer_ftilde = optim.Adam(self.f_tilde.parameters(), lr=args.lr)
        self.optimizer_DSAdecoder = optim.Adam(self.DSAdecoder.parameters(), lr=args.lr)

    def batch_test(self, z_private, label):
        self.top_model.eval()
        batch_test_correct = 0
        label = label.to(device)
        outputs = self.top_model(z_private)
        acc_loss = self.accuracy_criterion(outputs, label.long())

        _, predicted = outputs.max(1)
        batch_test_correct += predicted.eq(label).sum().item()

        return batch_test_correct

    def batch_attack(self, z_private, private_label, decoder):
        decoder.eval()
        batch_attack_correct = 0
        private_label = private_label.to(device)
        outputs = decoder(z_private)
        acc_loss = self.accuracy_criterion(outputs, private_label.long())

        _, predicted = outputs.max(1)
        batch_attack_correct += predicted.eq(private_label).sum().item()
        cm = confusion_matrix(private_label.cpu(), predicted.cpu())

        return batch_attack_correct, cm

    # cloud server trains model with both DSA and DPA
    def train_step(self, z_private, x_public, private_label, public_label, label, attacker_label):
        batch_correct = 0
        batch_DSA_correct = 0
        batch_DPA_correct = 0
        self.top_model.train()
        self.DPAdecoder.train()
        self.f_tilde.train()
        self.DSAdecoder.train()
        self.D.train()

        # accuracy loss
        x_public = x_public.to(device)
        private_label, public_label, label, attacker_label = private_label.to(device), public_label.to(device), label.to(device), attacker_label.to(device)
        outputs = self.top_model(z_private)
        acc_loss = self.accuracy_criterion(outputs, label.long())

        # this privacy loss should calculated by the local encoder
        if args.defense == 'Infocensor':
            privacy_loss2 = args.fair_kappa * fair_pred_mutual_information(outputs, private_label, private_label_num)

        # DPA decoder training loss
        DPAdecoder_output = self.DPAdecoder(z_private[:poison_num].detach())
        DPAdecoder_loss = self.accuracy_criterion(DPAdecoder_output, private_label[:poison_num].long())

        # calculate the total correctly classified examples in a batch
        _, predicted = outputs.max(1)
        batch_correct += predicted.eq(label).sum().item()

        # f_tilde loss
        z_public = self.f_tilde(x_public)
        # fake_outputs = self.top_model(z_public)
        # fake_acc_loss = self.accuracy_criterion(fake_outputs, attacker_label.long())
        adv_public_logits = self.D(z_public)
        ftilde_loss = -torch.mean(adv_public_logits)

        # DSA decoder training loss
        DSAdecoder_outputs_public = self.DSAdecoder(z_public.detach())  # detach z_public to prevent decoder loss backward to f_tilde
        DSAdecoder_loss = self.decoder_criterion(DSAdecoder_outputs_public, public_label.long())

        # discriminator on attacker's feature-space
        # detach z_public and z_private to prevent the D loss gradient backward to encoder and f_tilde
        adv_public_logits_detached = self.D(z_public.detach())
        adv_private_logits_detached = self.D(z_private.detach())

        loss_discr_true = torch.mean(adv_public_logits_detached)
        loss_discr_fake = -torch.mean(adv_private_logits_detached)
        # discriminator's loss
        D_loss = loss_discr_true + loss_discr_fake

        with torch.no_grad():
            # map to data space (for evaluation and style loss)
            # DSA
            DSAdecoder_outputs_private = self.DSAdecoder(z_private)
            _, predicted = DSAdecoder_outputs_private.max(1)
            batch_DSA_correct += predicted.eq(private_label).sum().item()
            loss_c_verification = self.accuracy_criterion(DSAdecoder_outputs_private, private_label.long())
            losses_c_verification = loss_c_verification.detach()

            # DPA
            DPAdecoder_outputs_private = self.DPAdecoder(z_private[poison_num:])
            _, predicted = DPAdecoder_outputs_private.max(1)
            batch_DPA_correct += predicted.eq(private_label[poison_num:]).sum().item()


        self.optimizer_top.zero_grad()
        self.optimizer_DPAdecoder.zero_grad()
        self.optimizer_ftilde.zero_grad()
        self.optimizer_DSAdecoder.zero_grad()
        self.optimizer_D.zero_grad()

        DPAdecoder_loss.backward()
        ftilde_loss.backward()
        #zeroing_grad(self.top_model)
        if args.defense == 'Infocensor':
            privacy_loss2.backward(retain_graph = True)
            #zeroing_grad(self.top_model)
        acc_loss.backward()
        # f_tilde loss has gradient on Discriminator, so before D loss backward, gradients should be cleared
        zeroing_grad(self.D)
        DSAdecoder_loss.backward()
        D_loss.backward()

        self.optimizer_top.step()
        self.optimizer_DPAdecoder.step()
        self.optimizer_ftilde.step()
        self.optimizer_DSAdecoder.step()
        self.optimizer_D.step()

        # record training losses
        batch_acc_loss = acc_loss.item()
        batch_ftilde_loss = ftilde_loss.item()
        batch_DSAdecoder_loss = DSAdecoder_loss.item()
        batch_D_loss = D_loss.item()
        batch_DPAdecoder_loss = DPAdecoder_loss.item()

        return outputs, batch_acc_loss, batch_DPAdecoder_loss, batch_ftilde_loss, batch_DSAdecoder_loss, batch_D_loss, batch_correct, batch_DSA_correct, batch_DPA_correct

def get_feature_test(client, x):
    if args.defense == 'None':
        z_private = client.get_feature_no_defense(x)
    if args.defense == 'Noisy':
        z_private = client.get_feature_noisy(x)
    elif args.defense == 'ADV':
        z_private = client.get_feature_ADV(x)
    elif args.defense == 'GMM-LC' or args.defense == 'GMM-EPO' or args.defense == 'GMM-RO':
        z_private = client.get_feature_GMM(x)
    elif args.defense == 'Infocensor':
        z_private = client.get_feature_info(x)

    return z_private

def test(client, cloud, test_loader):
    test_correct = 0
    with torch.no_grad():
        for batch_idx, (x, label, private_label) in enumerate(test_loader):
            x, label, private_label = x.to(device), label.to(device), private_label.to(device)
            z_private = get_feature_test(client, x)
            batch_val_correct = cloud.batch_test(z_private, label)
            test_correct += batch_val_correct

    test_acc = test_correct / float(len(test_loader.dataset))
    return test_acc

def attack(client, cloud, train_loader, test_loader):
    train_set_DSA_correct = 0
    train_set_DPA_correct = 0
    #test_set_attack_correct = 0

    # DSA can be used to attack both training data and test data
    with torch.no_grad():
        # attack training dataset
        for batch_idx, (x, label, private_label) in enumerate(train_loader):
            x, label, private_label = x.to(device), label.to(device), private_label.to(device)
            z_private = get_feature_test(client, x)
            # features = self.encoder(inputs)
            batch_DSA_correct, batch_DSA_cm = cloud.batch_attack(z_private, private_label, cloud.DSAdecoder)
            batch_DPA_correct, batch_DPA_cm = cloud.batch_attack(z_private[poison_num:], private_label[poison_num:], cloud.DPAdecoder)
            train_set_DSA_correct += batch_DSA_correct
            train_set_DPA_correct += batch_DPA_correct
            if batch_idx == 0:
                train_DSA_cm = batch_DSA_cm
                train_DPA_cm = batch_DPA_cm
            else:
                train_DSA_cm += batch_DSA_cm
                train_DPA_cm += batch_DPA_cm

        # attack test dataset
        # for batch_idx, (x, label, private_label) in enumerate(test_loader):
        #     x, label, private_label = x.to(device), label.to(device), private_label.to(device)
        #     z_private = get_feature_test(client, x)
        #     # features = self.encoder(inputs)
        #     batch_attack_correct, batch_cm = cloud.batch_attack(z_private, private_label)
        #     test_set_attack_correct += batch_attack_correct
        #     if batch_idx == 0:
        #         test_cm = batch_cm
        #     else:
        #         test_cm += batch_cm

    train_set_DSA_acc = train_set_DSA_correct / float(len(train_loader.dataset))
    train_set_DPA_acc = train_set_DPA_correct / float(len(train_loader.dataset)*(1-args.poison_ratio))
    #test_set_attack_acc = test_set_attack_correct / float(len(test_loader.dataset))

    train_DSA_Gmean = sqrt(train_DSA_cm[0][0] / (train_DSA_cm[0][0] + train_DSA_cm[0][1]) * train_DSA_cm[1][1] / (train_DSA_cm[1][0] + train_DSA_cm[1][1]))
    train_DPA_Gmean = sqrt(train_DPA_cm[0][0] / (train_DPA_cm[0][0] + train_DPA_cm[0][1]) * train_DPA_cm[1][1] / (
                train_DPA_cm[1][0] + train_DPA_cm[1][1]))
    #test_Gmean = sqrt(test_cm[0][0] / (test_cm[0][0] + test_cm[0][1]) * test_cm[1][1] / (test_cm[1][0] + test_cm[1][1]))

    return train_set_DSA_acc, train_set_DPA_acc, train_DSA_Gmean, train_DPA_Gmean

def warmup_training(client, train_loader):
    client_iterator = iter(train_loader)
    train_pri_losses = np.array([])
    print("WARMUP TRAINING...")
    # get data iterator
    iterator = list(range(args.warmup_iterations))

    for i in tqdm.tqdm(iterator, total=args.warmup_iterations):
        try:
            x_private, label, private_label = next(client_iterator)
            # drop the last incomplete batch
            if x_private.size(0) != args.batch_size:
                client_iterator = iter(train_loader)
                x_private, label, private_label = next(client_iterator)
        # All the data in the data set is traversed, reconstruct the dataloader
        except StopIteration:
            client_iterator = iter(train_loader)
            x_private, label, private_label = next(client_iterator)

        #print("x private: ",x_private)
        #print("label: ",private_label)
        batch_privacy_loss = client.warm_up(x_private, private_label)
        train_pri_losses = np.append(train_pri_losses, batch_privacy_loss)


def train(client, cloud, train_loader, test_loader, decoder_loader):
    client_iterator = iter(train_loader)
    attacker_iterator = iter(decoder_loader)

    train_acc_losses = []
    train_pri_losses = []
    train_adv_losses = []

    # DSA losses
    train_ftilde_losses = []
    train_DSAdecoder_losses = []
    train_D_losses = []

    # DPA losses
    train_DPAdecoder_losses = []

    # train and attack accuracy
    train_acc_list = []
    train_DSA_acc_list = []
    train_DPA_acc_list = []
    test_acc_list = []
    train_set_DSA_acc_list = []
    train_set_DPA_acc_list = []
    #test_set_attack_acc_list = np.array([])
    train_set_DSA_Gmean_list = []
    train_set_DPA_Gmean_list = []
    #test_set_attack_Gmean_list = np.array([])

    if args.defense == 'GMM-LC':
        logger.info('Iteration \t Train Acc Loss \t Train GMM Loss \t Train Acc \t Test Acc \t Train DSA Acc \t Train DSA G-mean \t Train DPA Acc \t Train DPA G-mean')
    elif args.defense == 'GMM-EPO' or args.defense == 'GMM-RO':
        logger.info('Iteration \t Train Acc Loss \t Train GMM Loss \t r1l1 \t r2l2 \t non-uniformity \t Train Acc \t Test Acc \t Train DSA Acc \t Train DSA G-mean \t Train DPA Acc \t Train DPA G-mean')
    elif args.defense == 'ADV':
        logger.info('Iteration \t Train Acc Loss \t Train Adv Loss \t Train Acc \t Test Acc \t Train DSA Acc \t Train DSA G-mean \t Train DPA Acc \t Train DPA G-mean')
    elif args.defense == 'Infocensor':
        logger.info('Iteration \t Train Acc Loss \t Train Info Loss \t Train Acc \t Test Acc \t Train DSA Acc \t Train DSA G-mean \t Train DPA Acc \t Train DPA G-mean')
    else:
        logger.info('Iteration \t Train Acc Loss \t Train Acc \t Test Acc \t Train DSA Acc \t Train DSA G-mean \t Train DPA Acc \t Train DPA G-mean')

    print("RUNNING...")
    # get data iterator
    iterator = list(range(args.iterations))

    train_acc_loss = 0.0
    train_GMM_loss = 0.0
    train_adv_loss = 0.0
    train_info_loss = 0.0
    train_acc = 0
    mode = 0
    for i in tqdm.tqdm(iterator, total=args.iterations):
        try:
            x_private, label, private_label = next(client_iterator)
            # drop the last incomplete batch
            if x_private.size(0) != args.batch_size:
                client_iterator = iter(train_loader)
                x_private, label, private_label = next(client_iterator)
        # All the data in the data set is traversed, reconstruct the dataloader
        except StopIteration:
            client_iterator = iter(train_loader)
            x_private, label, private_label = next(client_iterator)

        try:
            x_public, atk_acc_label, public_label = next(attacker_iterator)
            if x_public.size(0) != args.batch_size:
                attacker_iterator = iter(decoder_loader)
                x_public, atk_acc_label, public_label = next(attacker_iterator)
        except StopIteration:
            attacker_iterator = iter(decoder_loader)
            x_public, atk_acc_label, public_label = next(attacker_iterator)

        # client calculates feature on a minibatch
        if args.defense == 'None':
            z_private = client.get_feature_no_defense(x_private, private_label, 1)
        elif args.defense == 'Noisy':
            z_private = client.get_feature_noisy(x_private, private_label, 1)
        elif args.defense == 'GMM-LC' or args.defense == 'GMM-EPO' or args.defense == 'GMM-RO':
            z_private, batch_privacy_loss = client.get_feature_GMM(x_private, private_label, 1, i)
            train_pri_losses.append(batch_privacy_loss)
            train_GMM_loss += batch_privacy_loss
        elif args.defense == 'ADV':
            z_private, batch_adv_loss = client.get_feature_ADV(x_private, private_label, 1)
            train_adv_losses.append(batch_adv_loss)
            train_adv_loss += batch_adv_loss
        elif args.defense == 'Infocensor':
            z_private, batch_privacy_loss = client.get_feature_info(x_private, private_label, 1)
            train_pri_losses.append(batch_privacy_loss)
            train_info_loss += batch_privacy_loss
        else:
            raise ValueError(args.defense)


        # cloud server trains one step
        outputs, batch_acc_loss, batch_DPAdecoder_loss, batch_ftilde_loss, batch_DSAdecoder_loss, batch_D_loss, batch_correct, batch_DSA_correct, batch_DPA_correct\
            = cloud.train_step(z_private, x_public, private_label, public_label, label, atk_acc_label)

        # recoder batch results
        train_ftilde_losses.append(batch_ftilde_loss)
        train_acc_loss += batch_acc_loss
        train_acc_losses.append(batch_acc_loss)
        train_acc_list.append(batch_correct / args.batch_size)
        train_DSAdecoder_losses.append(batch_DSAdecoder_loss)
        train_D_losses.append(batch_D_loss)
        train_DSA_acc_list.append(batch_DSA_correct / args.batch_size)
        train_DPAdecoder_losses.append(batch_DPAdecoder_loss)
        train_DPA_acc_list.append(batch_DPA_correct / args.batch_size)

        train_acc += batch_correct

        # client updates its local model
        if args.defense == 'None':
            client.update_no_defense()
        elif args.defense == 'Noisy':
            client.update_noisy()
        elif args.defense == 'GMM-LC' or args.defense == 'GMM-EPO' or args.defense == 'GMM-RO':
            client.update_GMM(batch_acc_loss, mode=mode)
            mode = 1- mode
            # if i < 100:
            #     mode = 0
        elif args.defense == 'ADV':
            client.update_ADV()
        elif args.defense == 'Infocensor':
            client.update_info(outputs, private_label)
        else:
            raise ValueError(args.defense)


        # test and record results occasionally
        interval = args.record_interval
        if (i+1) % interval == 0:
            test_acc = test(client, cloud, test_loader)
            train_set_DSA_acc, train_set_DPA_acc, train_DSA_Gmean, train_DPA_Gmean = attack(client, cloud, train_loader, test_loader)

            # calculate average losses in a interval
            train_acc_loss /= interval
            train_GMM_loss /= interval
            train_adv_loss /= interval
            train_info_loss /= interval

            # record the attack results of each interval
            test_acc_list.append(test_acc)
            train_set_DSA_acc_list.append(train_set_DSA_acc)
            train_set_DPA_acc_list.append(train_set_DPA_acc)
            #test_set_attack_acc_list = np.append(test_set_attack_acc_list, test_set_attack_acc)
            train_set_DSA_Gmean_list.append(train_DSA_Gmean)
            train_set_DPA_Gmean_list.append(train_DPA_Gmean)
            #test_set_attack_Gmean_list = np.append(test_set_attack_Gmean_list, test_Gmean)
            train_acc /= (interval*args.batch_size)
            print("train accuracy: ", train_acc)
            print("test accuracy: ", test_acc)
            print("train set DSA accuracy: ", train_set_DSA_acc)
            print("train set DSA G-mean: ", train_DSA_Gmean)
            print("train set DPA accuracy: ", train_set_DPA_acc)
            print("train set DPA G-mean: ", train_DPA_Gmean)

            if args.defense == 'GMM-LC':
                logger.info('%d \t\t %.4f \t\t %.4f \t\t %.4f \t\t %.4f \t\t %.4f \t\t %.4f \t\t %.4f \t\t %.4f',
                            (i + 1), train_acc_loss, train_GMM_loss,
                            train_acc , test_acc, train_set_DSA_acc, train_DSA_Gmean, train_set_DPA_acc, train_DPA_Gmean
                            )
            elif args.defense == 'GMM-EPO' or args.defense == 'GMM-RO':
                logger.info('%d \t\t %.4f \t\t %.4f \t\t %.4f \t\t %.4f \t\t %.4f \t\t %.4f \t\t %.4f \t\t %.4f \t\t %.4f \t\t %.4f',
                            (i + 1), train_acc_loss, train_GMM_loss, train_acc_loss, train_GMM_loss,
                            train_acc, test_acc, train_set_DSA_acc, train_DSA_Gmean, train_set_DPA_acc, train_DPA_Gmean
                            )
            elif args.defense == 'ADV':
                logger.info('%d \t\t %.4f \t\t %.4f \t\t %.4f \t\t %.4f \t\t %.4f \t\t %.4f \t\t %.4f \t\t %.4f',
                            (i + 1), train_acc_loss, train_adv_loss,
                            train_acc , test_acc, train_set_DSA_acc, train_DSA_Gmean, train_set_DPA_acc, train_DPA_Gmean
                            )
            elif args.defense == 'Infocensor':
                logger.info('%d \t\t %.4f \t\t %.4f \t\t %.4f \t\t %.4f \t\t %.4f \t\t %.4f \t\t %.4f \t\t %.4f',
                            (i + 1), train_acc_loss, train_info_loss,
                            train_acc, test_acc, train_set_DSA_acc, train_DSA_Gmean, train_set_DPA_acc, train_DPA_Gmean
                            )
            else:
                logger.info('%d \t\t %.4f \t\t %.4f \t\t %.4f \t\t %.4f \t\t %.4f \t\t %.4f \t\t %.4f',
                            (i + 1), train_acc_loss,
                            train_acc , test_acc, train_set_DSA_acc, train_DSA_Gmean, train_set_DPA_acc, train_DPA_Gmean
                            )

            train_acc_loss = 0.0
            train_GMM_loss = 0.0
            train_adv_loss = 0.0
            train_info_loss = 0.0
            train_acc = 0

            # save model parameters
            if args.save_model:
                torch.save(encoder.state_dict(), os.path.join(modelSavePath, 'encoder_%s.pkl' % (str(i + 1))))
                torch.save(top_model.state_dict(), os.path.join(modelSavePath, 'top_model_%s.pkl' % (str(i + 1))))

    max_test_acc = max(test_acc_list)
    max_train_DSA_acc = max(train_set_DSA_acc_list)
    max_train_DSA_Gmean = max(train_set_DSA_Gmean_list)
    max_train_DPA_acc = max(train_set_DPA_acc_list)
    max_train_DPA_Gmean = max(train_set_DPA_Gmean_list)

    f = open(result_file, 'a')
    f.write(str(max_test_acc) + '\n')
    f.write(str(max_train_DSA_acc) + '\n')
    f.write(str(max_train_DSA_Gmean) + '\n')
    f.write(str(max_train_DPA_acc) + '\n')
    f.write(str(max_train_DPA_Gmean) + '\n')
    f.close()

    print("max test acc: ",max_test_acc)
    print("max train set DSA acc: ", max_train_DSA_acc)
    print("max train set DSA G-mean: ", max_train_DSA_Gmean)
    print("max train set DPA acc: ", max_train_DPA_acc)
    print("max train set DPA G-mean: ", max_train_DPA_Gmean)
    #print("max test set attack acc: ", max(test_set_attack_acc_list))
    #print("max test set attack G-mean: ", max(test_set_attack_Gmean_list))
    logger.info('Max test accuracy: \t %.4f', max_test_acc)
    logger.info('Max train set DSA accuracy: \t %.4f', max_train_DSA_acc)
    logger.info('Max train set DSA G-mean: \t %.4f', max_train_DSA_Gmean)
    logger.info('Max train set DPA accuracy: \t %.4f', max_train_DPA_acc)
    logger.info('Max train set DSA G-mean: \t %.4f', max_train_DPA_Gmean)
    #logger.info('Max test set attack accuracy: \t %.4f', max(test_set_attack_acc_list))
    #logger.info('Max test set attack G-mean: \t %.4f', max(test_set_attack_Gmean_list))

    # save figures
    WINDOW_SIZE = 15


    if args.defense != 'None':
        if args.defense == 'ADV':
            train_pri_loss_list = smooth(train_adv_losses, WINDOW_SIZE)
        #print("train privacy loss list: ",train_pri_loss_list)
        #print("privacy save path: ",priSavePath)
        plot_and_save_figure(train_pri_losses, [], 'batch', 'loss', 'train privacy loss', priSavePath, 'privacy_loss.png')
        plot_and_save_figure(train_acc_losses, train_pri_losses, 'accuracy loss', 'privacy loss',
                             'optimization process', figureSavePath, 'opt_process.png')

    train_acc_losses = smooth(train_acc_losses, WINDOW_SIZE)
    train_acc_list = smooth(train_acc_list, WINDOW_SIZE)

    plot_and_save_figure(train_acc_losses, [], 'batch', 'loss', 'train top loss', accSavePath, 'train_top_loss.png')
    plot_and_save_figure(train_acc_list, [], 'batch', 'accuracy', 'train accuracy', accSavePath, 'train_accuracy.png')

    train_ftilde_losses = smooth(train_ftilde_losses, WINDOW_SIZE)
    train_DSAdecoder_losses = smooth(train_DSAdecoder_losses, WINDOW_SIZE)
    train_D_losses = smooth(train_D_losses, WINDOW_SIZE)
    train_DPAdecoder_losses = smooth(train_DPAdecoder_losses, WINDOW_SIZE)
    train_set_DSA_acc_list = smooth(train_set_DSA_acc_list, WINDOW_SIZE)
    train_set_DPA_acc_list = smooth(train_set_DPA_acc_list, WINDOW_SIZE)

    plot_and_save_figure(train_ftilde_losses, [], 'batch', 'loss', 'train f_tilde loss', DSASavePath, 'train_ftilde_loss.png')
    plot_and_save_figure(train_DSAdecoder_losses, [], 'batch', 'loss', 'train DSA decoder loss', DSASavePath, 'DSAdecoder_loss.png')
    plot_and_save_figure(train_D_losses, [], 'batch', 'loss', 'train D loss', DSASavePath, 'train_D_loss.png')
    plot_and_save_figure(train_set_DSA_acc_list, [], 'batch', 'accuracy', 'train set DSA accuracy', DSASavePath, 'train_set_DSA_acc.png')

    plot_and_save_figure(train_DPAdecoder_losses, [], 'batch', 'loss', 'train DPA decoder loss', DPASavePath,
                         'DPAdecoder_loss.png')
    plot_and_save_figure(train_set_DPA_acc_list, [], 'batch', 'accuracy', 'train set DPA accuracy', DPASavePath,
                         'train_set_DPA_acc.png')
    #plot_and_save_figure(test_set_attack_acc_list, [], 'batch', 'accuracy', 'test set attack accuracy', topSavePath, 'test_set_attack_acc.png')

if __name__ == '__main__':
    dataset = args.dataset
    utility_task = args.utility_task
    privacy_task = args.privacy_task
    batch_size = args.batch_size
    model_idx = args.model_idx
    defense_method = args.defense

    poison_num = int(args.batch_size * args.poison_ratio)
    print("poison num: ",poison_num)

    logging.basicConfig(format='[%(asctime)s] - %(message)s',
                        datefmt='%Y/%m/%d %H:%M:%S',
                        level=logging.INFO,
                        filename=log_file)

    logger.info(args)


    print("defense method: ", defense_method)

    print("preparing datasets")
    train_loader, test_loader, label_num, private_label_num = load_main_dataset(dataset, utility_task, privacy_task, batch_size)
    decoder_loader = load_attacker_dataset(dataset, utility_task, privacy_task, batch_size)
    print("datasets ready")


    if model_idx == 1:
        Encoder = Encoder1
        InfoEncoder = InfocensorEncoder1
        Classifier = Classifier1
        Discriminator = Discriminator1
    elif model_idx == 2:
        Encoder = Encoder2
        InfoEncoder = InfocensorEncoder2
        Classifier = Classifier2
        Discriminator = Discriminator2
    elif model_idx == 3:
        Encoder = Encoder3
        InfoEncoder = InfocensorEncoder3
        Classifier = Classifier3
        Discriminator = Discriminator3
    else:
        raise ValueError(model_idx)

    if args.defense == 'Infocensor':
        encoder = InfoEncoder()
    else:
        encoder = Encoder()
    top_model = Classifier(label_num)
    f_tilde = Encoder()
    # determine the shape of the feature
    test_input = torch.randn((1, 1, 64, 64))
    test_feature = f_tilde(test_input)
    print("input shape: ", test_feature.shape)
    input_shape = test_feature.flatten().shape[0]
    DSAdecoder = Decoder(input_shape, private_label_num)
    D = Discriminator()
    adv_decoder = Classifier(private_label_num)
    DPAdecoder = Classifier(private_label_num)
    client = Client(encoder, adv_decoder)
    cloud = Cloud(top_model, DPAdecoder, f_tilde, DSAdecoder, D)

    if defense_method == 'GMM-RO':
        warmup_training(client, train_loader)

    train(client, cloud, train_loader, test_loader, decoder_loader)

    print("utility task: ", utility_task)
    print("privacy task: ", privacy_task)