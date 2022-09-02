# this file uses GMMLoss as the privacy loss
import numpy as np
import os
from torch import nn, optim
import torch
from load_data import load_main_dataset, load_attacker_dataset
import logging
# different model partitions
from models import (
    Encoder1, Classifier1, Discriminator1,
    Encoder2, Classifier2, Discriminator2,
    Encoder3, Classifier3, Discriminator3,
    Encoder4, Classifier4, Discriminator4,
    Encoder5, Classifier5, Discriminator5,
    Decoder
)
import matplotlib.pyplot as plt
import tqdm
from utils.epo_lp import EPO_LP
from utils.tools import smooth, makedir, plot_and_save_figure, getNumParams, get_gaussian
from utils.GMM_loss import GMMLoss, GMMLoss_pca
from utils.gradient_operations import save_and_clear_grad, reattach_grad, merge_gradients, zeroing_grad
import argparse

plt.switch_backend('Agg')

logger = logging.getLogger(__name__)

# basic parameters
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='celebA', choices=['celebA', 'UTKFace'], type=str)
parser.add_argument('--utility_task', default='BlackHair', type=str)
parser.add_argument('--privacy_task', default='Male', type=str)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--batch_size', default=500, type=int)
parser.add_argument('--iterations', default=3000, type=int)
parser.add_argument('--attack', default='DSA', choices=['DSA', 'None'], type=str)
parser.add_argument('--defense', default='ADV', choices=['GMM-LC', 'GMM-EPO', 'GMM-ADV', 'ADV', 'Noisy', 'None'], type=str)
parser.add_argument('--model_idx', default=3, choices=[1, 2, 3, 4, 5], type=int)  # different model partitions :
# 1: local owns 2 convolutional layers
# 2: local owns 2 convolutional layers and compress the output to 1024-dimensional vectors
# 3: local owns 3 convolutional layers
# 4: local owns 3 convolutional layers and compress the output to 2048-dimensional vectors
# 5: local owns all the 4 convolutional layers

# the number of iterations to test results and save model
parser.add_argument('--record_interval', default=50, type=int)

# DSA parameters
# parser.add_argument('--cal_acc', default=False, type=bool)  #whether the DSA attacker f_tilde simulates the accuracy task
# parser.add_argument('--top_ratio', default=0.5, type=float)

# Adversarial training parameters
parser.add_argument('--adv_weight', default=0.5, type=float) # encoder_gradient = top_gradient - adv_weight * decoder_gradient
parser.add_argument('--flip_label', default=True, type=bool)

# Noisy parameter
parser.add_argument('--noise_sigma', default=1, type=float)   # deciding the variance of the noise

# GMM parameters
parser.add_argument('--sigma', default=1, type=float)   # deciding the variance of each Gaussian distribution
parser.add_argument('--privacy_constraint', default=1, type=float)  # the privacy loss constraint in GMM training process
parser.add_argument('--restrict_mode', default='None',choices=['start', 'each-batch', 'None'],
                    type=str)  # whether the privacy loss constraint acts on each batch

# GMM-LC parameters
parser.add_argument('--acc_weight', default=0.7, type=float)

# GMM-EPO parameters, these parameters take effect only when the defense method is GMM-EPO
parser.add_argument('--preference', default=0.2, type=float) # preference for accuracy task
parser.add_argument('--eps', default=1e-3, type=float) # EPO non-uniformity restriction
parser.add_argument('--allow_privacy_ascent', default=True, type=bool)  # whether the privacy loss is allowed to ascent in training process


args = parser.parse_args()
print(args)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# create results saving path
directory = './%s_new_results'%(args.dataset)
makedir(directory)

#SavePath = directory
SavePath = directory + '/utility=%s_privacy=%s'%(
    args.utility_task, args.privacy_task)
makedir(SavePath)

SavePath = SavePath + '/model-idx=%s'%(str(args.model_idx))
makedir(SavePath)

SavePath = SavePath + '/defense=%s_iter=%s_bs=%s'%(
    args.defense, str(args.iterations), str(args.batch_size))

if args.defense == 'GMM-LC':
    SavePath = SavePath + '_sigma=%s_acc-w=%s' % (str(args.sigma), str(args.acc_weight))
elif args.defense == 'GMM-EPO':
    SavePath = SavePath + '_sigma=%s_pref=%s_eps=%s_pc=%s_ascent=%s_mode=%s' %(str(args.sigma),
        str(args.preference), str(args.eps), str(args.privacy_constraint), str(args.allow_privacy_ascent), str(args.restrict_mode))
elif args.defense == 'ADV':
    SavePath = SavePath + '_adv-w=%s_flip=%s' % (str(args.adv_weight), str(args.flip_label))
elif args.defense == 'GMM-ADV':
    SavePath = SavePath + '_sigma=%s_acc-w=%s_adv-w=%s' % (str(args.sigma), str(args.acc_weight), str(args.adv_weight))
elif args.defense == 'Noisy':
    SavePath = SavePath + '_noise-sigma=%s' % (str(args.noise_sigma))


if args.attack == 'DSA':
    SavePath = SavePath + '_attack=DSA'
else:
    SavePath = SavePath + '_attack=None'

makedir(SavePath)

modelSavePath = SavePath + '/models'
makedir(modelSavePath)
figureSavePath = SavePath + '/figures'
makedir(figureSavePath)
topSavePath = figureSavePath + '/top_loss'
makedir(topSavePath)
priSavePath = figureSavePath + '/pri_loss'
makedir(priSavePath)
logSavePath = SavePath + '/logs'
makedir(logSavePath)
logfile = logSavePath + '/log.txt'
if os.path.exists(logfile):
    os.remove(logfile)


class Client(nn.Module):
    def __init__(self, encoder, adv_decoder):
        super(Client, self).__init__()
        self.encoder = encoder.to(device)
        self.adv_decoder = adv_decoder.to(device)
        self.privacy_criterion = GMMLoss_pca(args.sigma, device)
        self.accuracy_criterion = torch.nn.CrossEntropyLoss()
        self.optimizer_decoder = optim.Adam(self.adv_decoder.parameters(), lr=args.lr)
        self.optimizer_encoder = optim.Adam(self.encoder.parameters(), lr=args.lr)
        _, n_params = getNumParams(self.encoder.parameters())
        self.n_params = n_params
        self.preference = np.array([args.preference, 1-args.preference])
        self.epo_lp = EPO_LP(m=2, n=n_params, r=self.preference,eps=args.eps)

    def get_feature_no_defense(self, x_private, private_label, training=0):
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
            if args.flip_label:
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

    def get_feature_GMM(self, x_private, private_label=None, training=0, batch_idx=0):
        if training:
            self.encoder.train()
            x_private = x_private.to(device)
            mu_private = self.encoder(x_private)
            z_private = get_gaussian(mu_private, args.sigma)
            pri_loss = self.privacy_criterion(mu_private, private_label)

            # only the privacy loss of the first batch of data will be restricted
            if args.restrict_mode == 'start':
                if batch_idx == 0:
                    while pri_loss > args.privacy_constraint:
                        print("privacy cannot satisfy requirement , stop uploading")
                        self.optimizer_encoder.zero_grad()
                        pri_loss.backward()
                        self.optimizer_encoder.step()
                        mu_private = self.encoder(x_private)
                        z_private = get_gaussian(mu_private, args.sigma)
                        pri_loss = self.privacy_criterion(mu_private, private_label)

            # privacy loss of each batch of data will be restricted
            elif args.restrict_mode == 'each_batch':
                while pri_loss > args.privacy_constraint:
                    print("privacy cannot satisfy requirement , stop uploading")
                    self.optimizer_encoder.zero_grad()
                    pri_loss.backward()
                    self.optimizer_encoder.step()
                    mu_private = self.encoder(x_private)
                    z_private = get_gaussian(mu_private, args.sigma)
                    pri_loss = self.privacy_criterion(mu_private, private_label)
            self.privacy_loss = pri_loss.item()
            self.optimizer_encoder.zero_grad()
            pri_loss.backward(retain_graph=True)
            self.privacy_gradient, self.flatten_pg = save_and_clear_grad(self.encoder)

            return z_private, pri_loss.item()
        else:
            self.encoder.eval()
            x_private = x_private.to(device)
            mu_private = self.encoder(x_private)
            z_private = get_gaussian(mu_private, args.sigma)
            #pri_loss = self.privacy_criterion(mu_private, private_label)

            return z_private

    def get_feature_GMM_ADV(self, x_private, private_label=None, training=0):
        if training:
            self.encoder.train()
            self.adv_decoder.train()
            x_private, private_label = x_private.to(device), private_label.to(device)
            mu_private = self.encoder(x_private)
            z_private = get_gaussian(mu_private, args.sigma)
            decoder_outputs = self.adv_decoder(z_private)
            adv_loss = self.accuracy_criterion(decoder_outputs, private_label.long())
            pri_loss = self.privacy_criterion(mu_private, private_label)

            self.optimizer_encoder.zero_grad()
            self.optimizer_decoder.zero_grad()
            adv_loss.backward(retain_graph=True)
            self.decoder_gradient, self.flatten_pg = save_and_clear_grad(self.encoder)
            pri_loss.backward(retain_graph=True)
            self.privacy_gradient, self.flatten_pg = save_and_clear_grad(self.encoder)
            self.optimizer_decoder.step()
            #self.optimizer_encoder.zero_grad()

            return z_private, adv_loss.item(), pri_loss.item()
        else:
            self.encoder.eval()
            x_private = x_private.to(device)
            mu_private = self.encoder(x_private)
            z_private = get_gaussian(mu_private, args.sigma)

            return z_private


    def get_EPO_weight(self, top_loss):
        grads = {}
        losses_vec = []
        losses_vec.append(top_loss)
        losses_vec.append(self.privacy_loss)
        grads[0] = self.flatten_tg
        grads[1] = self.flatten_pg

        grads_list = [torch.cat(grads[i]) for i in range(len(grads))]
        G = torch.stack(grads_list)
        GG = G @ G.T
        # print("GG shape: ",GG.shape)
        losses_vec = np.stack(losses_vec)
        # calculate the weights
        weights = self.epo_lp.get_alpha(losses_vec, G=GG.cpu().numpy(), C=True, relax=True,
                                 allow_privacy_ascent=args.allow_privacy_ascent)
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

    def update_GMM(self, top_loss):
        self.top_gradient, self.flatten_tg = save_and_clear_grad(self.encoder)
        #self.top_gradient = self.encoder.parameters().grad
        if args.defense == 'GMM-LC':
            top_weight = args.acc_weight
        elif args.defense == 'GMM-EPO':
            top_weight = self.get_EPO_weight(top_loss)
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

    def update_GMM_ADV(self):
        self.top_gradient, self.flatten_tg = save_and_clear_grad(self.encoder)
        merged_gradient = merge_gradients(self.top_gradient, self.privacy_gradient, [args.acc_weight, 1-args.acc_weight])
        final_gradient = merge_gradients(merged_gradient, self.decoder_gradient, [1, -args.adv_weight])
        reattach_grad(self.encoder, final_gradient)
        self.optimizer_encoder.step()


class Cloud(nn.Module):
    def __init__(self, top_model, f_tilde, decoder, discriminator):
        super(Cloud, self).__init__()
        self.top_model = top_model.to(device)
        self.f_tilde = f_tilde.to(device)
        self.decoder = decoder.to(device)
        self.D = discriminator.to(device)
        self.accuracy_criterion = torch.nn.CrossEntropyLoss()
        self.optimizer_top = optim.Adam(self.top_model.parameters(), lr=args.lr)
        self.optimizer_D = optim.Adam(self.D.parameters(), lr=args.lr)
        # when using FSHA, f_tilde and decoder will be treated as an autoencoder
        #self.optimizer_AE = optim.Adam([{'params': self.f_tilde.parameters()}, {'params': self.decoder.parameters()}], lr=args.lr)

        # when using DSA, f_tilde and decoder are decoupled and updated separately
        self.optimizer_ftilde = optim.Adam(self.f_tilde.parameters(), lr=args.lr)
        self.optimizer_decoder = optim.Adam(self.decoder.parameters(), lr=args.lr)

    def batch_test(self, z_private, label):
        self.top_model.eval()
        batch_test_correct = 0
        label = label.to(device)
        outputs = self.top_model(z_private)
        acc_loss = self.accuracy_criterion(outputs, label.long())

        _, predicted = outputs.max(1)
        batch_test_correct += predicted.eq(label).sum().item()

        return batch_test_correct

    def batch_attack(self, z_private, private_label):
        self.decoder.eval()
        batch_attack_correct = 0
        private_label = private_label.to(device)
        outputs = self.decoder(z_private)
        acc_loss = self.accuracy_criterion(outputs, private_label.long())

        _, predicted = outputs.max(1)
        batch_attack_correct += predicted.eq(private_label).sum().item()

        return batch_attack_correct


    # cloud server no attacks
    def train_step_no_attack(self, z_private, label):
        batch_correct = 0
        self.top_model.train()

        label = label.to(device)
        outputs = self.top_model(z_private)
        acc_loss = self.accuracy_criterion(outputs, label.long())

        # calculate the total correctly classified examples in a batch
        _, predicted = outputs.max(1)
        batch_correct += predicted.eq(label).sum().item()

        self.optimizer_top.zero_grad()
        acc_loss.backward()
        self.optimizer_top.step()

        train_batch_acc = batch_correct / args.batch_size
        batch_acc_loss = acc_loss.item()

        return batch_acc_loss, batch_correct

    # cloud server uses DSA
    def train_step_DSA(self, z_private, x_public, private_label, public_label, label, attacker_label):
        batch_correct = 0
        batch_attack_correct = 0
        self.top_model.train()
        self.f_tilde.train()
        self.decoder.train()
        self.D.train()

        # accuracy loss
        x_public = x_public.to(device)
        private_label, public_label, label, attacker_label = private_label.to(device), public_label.to(device), label.to(device), attacker_label.to(device)
        outputs = self.top_model(z_private)
        acc_loss = self.accuracy_criterion(outputs, label.long())

        # calculate the total correctly classified examples in a batch
        _, predicted = outputs.max(1)
        batch_correct += predicted.eq(label).sum().item()

        # f loss
        z_public = self.f_tilde(x_public)
        fake_outputs = self.top_model(z_public)
        fake_acc_loss = self.accuracy_criterion(fake_outputs, attacker_label.long())
        adv_public_logits = self.D(z_public)
        ftilde_loss = -torch.mean(adv_public_logits)
        # if args.cal_acc:
        #     ftilde_loss = args.top_ratio * fake_acc_loss + (1-args.acc_ratio)*ftilde_loss


        # decoder training loss
        decoder_outputs_public = self.decoder(z_public.detach()) # detach z_public to prevent decoder loss backward to f_tilde
        decoder_loss = self.accuracy_criterion(decoder_outputs_public, public_label.long())

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
            decoder_outputs_private = self.decoder(z_private)
            _, predicted = decoder_outputs_private.max(1)
            batch_attack_correct += predicted.eq(private_label).sum().item()
            loss_c_verification = self.accuracy_criterion(decoder_outputs_private, private_label.long())
            losses_c_verification = loss_c_verification.detach()

        self.optimizer_top.zero_grad()
        self.optimizer_ftilde.zero_grad()
        self.optimizer_decoder.zero_grad()
        self.optimizer_D.zero_grad()

        ftilde_loss.backward()
        zeroing_grad(self.top_model)
        acc_loss.backward()
        # f_tilde loss has gradient on Discriminator, so before D loss backward, gradients should be cleared
        zeroing_grad(self.D)
        decoder_loss.backward()
        D_loss.backward()

        self.optimizer_top.step()
        self.optimizer_ftilde.step()
        self.optimizer_decoder.step()
        self.optimizer_D.step()

        # record training losses
        batch_acc_loss = acc_loss.item()
        batch_ftilde_loss = ftilde_loss.item()
        batch_decoder_loss = decoder_loss.item()
        batch_D_loss = D_loss.item()

        return batch_acc_loss, batch_ftilde_loss, batch_decoder_loss, batch_D_loss, batch_correct, batch_attack_correct

    # # cloud server uses FSHA
    # def train_step_FSHA(self, z_private, x_public, private_label, public_label, label):
    #     batch_correct = 0
    #     batch_attack_correct = 0
    #     self.top_model.train()
    #     self.f_tilde.train()
    #     self.decoder.train()
    #     self.D.train()
    #
    #     x_public = x_public.to(device)
    #     private_label, public_label, label = private_label.to(device),public_label.to(device),label.to(device)
    #
    #     # encoder loss
    #     outputs = self.top_model(z_private)
    #     acc_loss = self.accuracy_criterion(outputs,label.long())
    #     adv_private_logits = self.D(z_private)
    #     f_loss = torch.mean(adv_private_logits)
    #     # attacker takes accuracy into account
    #     if args.use_acc:
    #         encoder_loss = args.acc_ratio*acc_loss + (1-args.acc_ratio)*f_loss
    #     else:
    #         encoder_loss = f_loss
    #
    #     # calculate the total correctly classified examples in a batch
    #     _, predicted = outputs.max(1)
    #     batch_correct += predicted.eq(label).sum().item()
    #
    #     # autoencoder training loss
    #     z_public = self.f_tilde(x_public)
    #     # print("z shape: ",z_public.shape)
    #     decoder_outputs_public = self.decoder(z_public)
    #     decoder_loss = self.accuracy_criterion(decoder_outputs_public, public_label.long())
    #
    #     # discriminator on attacker's feature-space
    #     adv_public_logits = self.D(z_public.detach())
    #     adv_private_logits_detached = self.D(z_private.detach())
    #
    #     loss_discr_true = torch.mean(adv_public_logits)
    #     loss_discr_fake = -torch.mean(adv_private_logits_detached)
    #     # discriminator's loss
    #     D_loss = loss_discr_true + loss_discr_fake
    #
    #     with torch.no_grad():
    #         # map to data space (for evaluation and style loss)
    #         decoder_outputs_private = self.decoder(z_private)
    #         _, predicted = decoder_outputs_private.max(1)
    #         batch_attack_correct += predicted.eq(private_label).sum().item()
    #         loss_c_verification = self.accuracy_criterion(decoder_outputs_private, private_label.long())
    #         losses_c_verification = loss_c_verification.detach()
    #
    #     self.optimizer_top.zero_grad()
    #     self.optimizer_AE.zero_grad()
    #     self.optimizer_D.zero_grad()
    #
    #     encoder_loss.backward()
    #     zeroing_grad(self.D)
    #     decoder_loss.backward()
    #     D_loss.backward()
    #
    #     self.optimizer_top.step()
    #     self.optimizer_AE.step()
    #     self.optimizer_D.step()
    #
    #     batch_acc_loss = acc_loss.item()
    #     batch_f_loss = f_loss.item()
    #     batch_decoder_loss = decoder_loss.item()
    #     batch_D_loss = D_loss.item()
    #
    #     return batch_acc_loss, batch_f_loss, batch_decoder_loss, batch_D_loss, batch_correct, batch_attack_correct

def get_feature_test(client, x):
    if args.defense == 'None':
        z_private = client.get_feature_no_defense(x)
    if args.defense == 'Noisy':
        z_private = client.get_feature_noisy(x)
    elif args.defense == 'ADV':
        z_private = client.get_feature_ADV(x)
    elif args.defense == 'GMM-LC' or args.defense == 'GMM-EPO':
        z_private = client.get_feature_GMM(x)
    elif args.defense == 'GMM-ADV':
        z_private = client.get_feature_GMM_ADV(x)

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

def atk(client, cloud, train_loader, test_loader):
    train_set_attack_correct = 0
    test_set_attack_correct = 0

    # DSA can be used to attack both training data and test data
    with torch.no_grad():
        # attack training dataset
        for batch_idx, (x, label, private_label) in enumerate(train_loader):
            x, label, private_label = x.to(device), label.to(device), private_label.to(device)
            z_private = get_feature_test(client, x)
            # features = self.encoder(inputs)
            batch_attack_correct = cloud.batch_attack(z_private, private_label)
            train_set_attack_correct += batch_attack_correct

        # attack test dataset
        for batch_idx, (x, label, private_label) in enumerate(test_loader):
            x, label, private_label = x.to(device), label.to(device), private_label.to(device)
            z_private = get_feature_test(client, x)
            # features = self.encoder(inputs)
            batch_attack_correct = cloud.batch_attack(z_private, private_label)
            test_set_attack_correct += batch_attack_correct

    train_set_attack_acc = train_set_attack_correct / float(len(train_loader.dataset))
    test_set_attack_acc = test_set_attack_correct / float(len(test_loader.dataset))

    return train_set_attack_acc, test_set_attack_acc

def train(client, cloud, train_loader, test_loader, decoder_loader):
    client_iterator = iter(train_loader)
    attacker_iterator = iter(decoder_loader)

    train_acc_loss_list = np.array([])
    train_pri_loss_list = np.array([])
    train_adv_loss_list = np.array([])

    # DSA losses
    train_f_loss_list = np.array([])
    train_decoder_loss_list = np.array([])
    train_D_loss_list = np.array([])

    train_acc_list = np.array([])
    train_attack_acc_list = np.array([])
    test_acc_list = np.array([])
    train_set_attack_acc_list = np.array([])
    test_set_attack_acc_list = np.array([])

    if args.defense == 'GMM-LC' or args.defense == 'GMM-EPO':
        logger.info('Iteration \t Train Acc Loss \t Train GMM Loss \t Train Acc \t Test Acc \t Train Set Attack Acc \t Test Set Attack Acc')
    elif args.defense == 'ADV':
        logger.info('Iteration \t Train Acc Loss \t Train Adversary Loss \t Train Acc \t Test Acc \t Train Set Attack Acc \t Test Set Attack Acc')
    elif args.defense == 'GMM-ADV':
        logger.info('Iteration \t Train Acc Loss \t Train GMM Loss \t Train Adversary Loss \t Train Acc \t Test Acc \t Train Set Attack Acc \t Test Set Attack Acc')
    else:
        logger.info('Iteration \t Train Acc Loss \t Train Acc \t Test Acc \t Train Set Attack Acc \t Test Set Attack Acc')

    print("RUNNING...")
    # get data iterator
    iterator = list(range(args.iterations))

    train_acc_loss = 0.0
    train_GMM_loss = 0.0
    train_adv_loss = 0.0
    train_acc = 0
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
        elif args.defense == 'GMM-LC' or args.defense == 'GMM-EPO':
            z_private, batch_privacy_loss = client.get_feature_GMM(x_private, private_label, 1, i)
            train_pri_loss_list = np.append(train_pri_loss_list, batch_privacy_loss)
            train_GMM_loss += batch_privacy_loss
        elif args.defense == 'ADV':
            z_private, batch_adv_loss = client.get_feature_ADV(x_private, private_label, 1)
            train_adv_loss_list = np.append(train_adv_loss_list, batch_adv_loss)
            train_adv_loss += batch_adv_loss
        elif args.defense == 'GMM-ADV':
            z_private, batch_adv_loss, batch_privacy_loss = client.get_feature_GMM_ADV(x_private, private_label, 1)
            train_pri_loss_list = np.append(train_pri_loss_list, batch_privacy_loss)
            train_adv_loss_list = np.append(train_adv_loss_list, batch_adv_loss)
            train_GMM_loss += batch_privacy_loss
            train_adv_loss += batch_adv_loss
        else:
            raise ValueError(args.defense)


        # cloud server trains one step
        if args.attack == 'None':
            batch_acc_loss, batch_correct = cloud.train_step_no_attack(z_private, label)
        elif args.attack == 'DSA':
            batch_acc_loss, batch_ftilde_loss, batch_decoder_loss, batch_D_loss, batch_correct, batch_attack_correct = cloud.train_step_DSA(z_private, x_public, private_label, public_label, label, atk_acc_label)
            train_f_loss_list = np.append(train_f_loss_list, batch_ftilde_loss)
            train_acc_loss += batch_acc_loss
        # elif args.attack == 'FSHA':
        #     batch_acc_loss, batch_f_loss, batch_decoder_loss, batch_D_loss, batch_correct, batch_attack_correct = cloud.train_step_FSHA(z_private, x_public, private_label, public_label, label)
        #     train_f_loss_list = np.append(train_f_loss_list, batch_f_loss)

        else:
            raise ValueError(args.attack)

        train_acc += batch_correct

        # client updates its local model
        if args.defense == 'None':
            client.update_no_defense()
        elif args.defense == 'Noisy':
            client.update_noisy()
        elif args.defense == 'GMM-LC' or args.defense == 'GMM-EPO':
            client.update_GMM(batch_acc_loss)
        elif args.defense == 'ADV':
            client.update_ADV()
        elif args.defense == 'GMM-ADV':
            client.update_GMM_ADV()
        else:
            raise ValueError(args.defense)

        # recoder batch results
        train_acc_loss_list = np.append(train_acc_loss_list, batch_acc_loss)
        train_acc_list = np.append(train_acc_list, batch_correct/args.batch_size)

        if args.attack != 'None':
            train_decoder_loss_list = np.append(train_decoder_loss_list, batch_decoder_loss)
            train_D_loss_list = np.append(train_D_loss_list, batch_D_loss)
            train_attack_acc_list = np.append(train_attack_acc_list, batch_attack_correct / args.batch_size)


        # test and record results occasionally
        interval = args.record_interval
        if (i+1) % interval == 0:
            test_acc = test(client, cloud, test_loader)
            train_set_attack_acc, test_set_attack_acc = atk(client, cloud, train_loader, test_loader)

            test_acc_list = np.append(test_acc_list, test_acc)
            train_set_attack_acc_list = np.append(train_set_attack_acc_list, train_set_attack_acc)
            test_set_attack_acc_list = np.append(test_set_attack_acc_list, test_set_attack_acc)
            train_acc /= (interval*args.batch_size)
            print("train accuracy: ", train_acc)
            print("test accuracy: ", test_acc)
            print("train set attack accuracy: ", train_set_attack_acc)
            print("test set attack accuracy: ", test_set_attack_acc)

            if args.defense == 'GMM-LC' or args.defense == 'GMM-EPO':
                logger.info('%d \t\t %.4f \t\t %.4f \t\t %.4f \t\t %.4f \t\t %.4f \t\t %.4f',
                            (i + 1), train_acc_loss / interval, train_GMM_loss / interval,
                            train_acc , test_acc, train_set_attack_acc, test_set_attack_acc
                            )
            elif args.defense == 'ADV':
                logger.info('%d \t\t %.4f \t\t %.4f \t\t %.4f \t\t %.4f \t\t %.4f \t\t %.4f',
                            (i + 1), train_acc_loss / interval, train_adv_loss / interval,
                            train_acc , test_acc, train_set_attack_acc, test_set_attack_acc
                            )
            elif args.defense == 'GMM-ADV':
                logger.info('%d \t\t %.4f \t\t %.4f \t\t %.4f \t\t %.4f \t\t %.4f \t\t %.4f \t\t %.4f',
                            (i + 1), train_acc_loss / interval, train_GMM_loss / interval, train_adv_loss / interval,
                            train_acc , test_acc, train_set_attack_acc, test_set_attack_acc
                            )
            else:
                logger.info('%d \t\t %.4f \t\t %.4f \t\t %.4f \t\t %.4f \t\t %.4f',
                            (i + 1), train_acc_loss / interval,
                            train_acc , test_acc, train_set_attack_acc, test_set_attack_acc
                            )

            train_acc_loss = 0.0
            train_GMM_loss = 0.0
            train_adv_loss = 0.0
            train_acc = 0

            # save model parameters
            torch.save(encoder.state_dict(), os.path.join(modelSavePath, 'encoder_%s.pkl' % (str(i + 1))))
            torch.save(top_model.state_dict(), os.path.join(modelSavePath, 'top_model_%s.pkl' % (str(i + 1))))

    print("max test acc: ",max(test_acc_list))
    print("max train set attack acc: ", max(train_set_attack_acc_list))
    print("max test set attack acc: ", max(test_set_attack_acc_list))
    logger.info('Max test accuracy: \t %.4f', max(test_acc_list))
    logger.info('Max train set attack accuracy: \t %.4f', max(train_set_attack_acc_list))
    logger.info('Max test set attack accuracy: \t %.4f', max(test_set_attack_acc_list))

    # save figures
    WINDOW_SIZE = 15
    train_acc_loss_list = smooth(train_acc_loss_list, WINDOW_SIZE)
    train_acc_list = smooth(train_acc_list, WINDOW_SIZE)

    plot_and_save_figure(train_acc_loss_list, [], 'batch', 'loss', 'train top loss', topSavePath, 'train_top_loss.png')
    plot_and_save_figure(train_acc_list, [], 'batch', 'accuracy', 'train accuracy', topSavePath, 'train_accuracy.png')

    if args.defense != 'None':
        if args.defense == 'ADV':
            train_pri_loss_list = smooth(train_adv_loss_list, WINDOW_SIZE)
        #print("train privacy loss list: ",train_pri_loss_list)
        #print("privacy save path: ",priSavePath)
        plot_and_save_figure(train_pri_loss_list, [], 'batch', 'loss', 'train privacy loss', priSavePath, 'privacy_loss.png')
        plot_and_save_figure(train_acc_loss_list, train_pri_loss_list, 'accuracy loss', 'privacy loss',
                             'optimization process', figureSavePath, 'opt_process.png')

        if args.defense == 'GMM-ADV':
            train_adv_loss_list = smooth(train_adv_loss_list, WINDOW_SIZE)
            plot_and_save_figure(train_adv_loss_list, [], 'batch', 'loss', 'train adversary loss', priSavePath,
                                 'adversary_loss.png')

    if args.attack != 'None':
        train_f_loss_list = smooth(train_f_loss_list, WINDOW_SIZE)
        train_decoder_loss_list = smooth(train_decoder_loss_list, WINDOW_SIZE)
        train_D_loss_list = smooth(train_D_loss_list, WINDOW_SIZE)
        train_set_attack_acc_list = smooth(train_set_attack_acc_list, WINDOW_SIZE)
        test_set_attack_acc_list = smooth(test_set_attack_acc_list, WINDOW_SIZE)

        plot_and_save_figure(train_f_loss_list, [], 'batch', 'loss', 'train f loss', topSavePath, 'train_f_loss.png')
        plot_and_save_figure(train_decoder_loss_list, [], 'batch', 'loss', 'train decoder loss', topSavePath, 'decoder_loss.png')
        plot_and_save_figure(train_D_loss_list, [], 'batch', 'loss', 'train D loss', topSavePath, 'train_D_loss.png')
        plot_and_save_figure(train_set_attack_acc_list, [], 'batch', 'accuracy', 'train set attack accuracy', topSavePath, 'train_set_attack_acc.png')
        plot_and_save_figure(test_set_attack_acc_list, [], 'batch', 'accuracy', 'test set attack accuracy', topSavePath, 'test_set_attack_acc.png')

if __name__ == '__main__':
    dataset = args.dataset
    utility_task = args.utility_task
    privacy_task = args.privacy_task
    batch_size = args.batch_size
    model_idx = args.model_idx
    attack = args.attack
    defense = args.defense

    logging.basicConfig(format='[%(asctime)s] - %(message)s',
                        datefmt='%Y/%m/%d %H:%M:%S',
                        level=logging.INFO,
                        filename=logfile)

    logger.info(args)


    print("attack method: ",attack)
    print("defense method: ", defense)

    print("preparing datasets")
    train_loader, test_loader, label_num, private_label_num = load_main_dataset(dataset, utility_task, privacy_task, batch_size)
    decoder_loader = load_attacker_dataset(dataset, utility_task, privacy_task, batch_size)
    print("datasets ready")


    if model_idx == 1:
        Encoder = Encoder1
        Classifier = Classifier1
        Discriminator = Discriminator1
    elif model_idx == 2:
        Encoder = Encoder2
        Classifier = Classifier2
        Discriminator = Discriminator2
    elif model_idx == 3:
        Encoder = Encoder3
        Classifier = Classifier3
        Discriminator = Discriminator3
    elif model_idx == 4:
        Encoder = Encoder4
        Classifier = Classifier4
        Discriminator = Discriminator4
    elif model_idx == 5:
        Encoder = Encoder5
        Classifier = Classifier5
        Discriminator = Discriminator5
    else:
        raise ValueError(model_idx)

    encoder = Encoder()
    top_model = Classifier(label_num)
    f_tilde = Encoder()
    # determine the shape of the feature
    test_input = torch.randn((1, 1, 64, 64))
    test_feature = f_tilde(test_input)
    input_shape = test_feature.flatten().shape[0]
    #print("input shape: ",input_shape)
    decoder = Decoder(input_shape, private_label_num)
    D = Discriminator()
    adv_decoder = Classifier(private_label_num)
    client = Client(encoder, adv_decoder)
    cloud = Cloud(top_model, f_tilde, decoder, D)

    train(client, cloud, train_loader, test_loader, decoder_loader)


    print("utility task: ", utility_task)
    print("privacy task: ", privacy_task)