import numpy as np
import os
from torch import nn, optim
import torch
from load_data import load_main_dataset, load_attacker_dataset, construct_data_loader
from sklearn.metrics import confusion_matrix
from math import sqrt
from models import (
    Encoder1, Classifier1, InfocensorEncoder1,
    Encoder2, Classifier2, InfocensorEncoder2,
    Encoder3, Classifier3, InfocensorEncoder3,
)
from utils.tools import smooth, get_gaussian, plot_and_save_figure, makedir
import matplotlib.pyplot as plt
import tqdm
import logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--batch_size', default=500, type=int)
parser.add_argument('--iterations', default=1000, type=int)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--test_interval', default=50, type=int)
parser.add_argument('--dir', default='./celebA_results_63/utility=Smiling_privacy=Male/model-idx=2/seed=10_pr=0.05_defense=ADV_iter=2000_bs=500_adv-w=0.5_flip=1', type=str)

args = parser.parse_args()
print(args)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(args.seed)
np.random.seed(args.seed)
# if torch.cuda.is_available():
#     device = 'cuda'
#     torch.cuda.manual_seed(args.seed)
# else:
#     device = 'cpu'

logger = logging.getLogger(__name__)

class Client(nn.Module):
    def __init__(self, encoder):
        super(Client, self).__init__()
        self.encoder = encoder.to(device)

    def get_feature(self, x_private):
        self.encoder.eval()
        if defense == 'None' or defense == 'ADV':
            x_private = x_private.to(device)
            z_private = self.encoder(x_private)
        elif defense == 'GMM-LC' or defense == 'GMM-RO' or defense == 'GMM-ADV' or defense == 'Noisy':
            x_private = x_private.to(device)
            mu_private = self.encoder(x_private)
            z_private = get_gaussian(mu_private, sigma)
        elif defense == 'Infocensor':
            x_private = x_private.to(device)
            mu_private, sigma_private = self.encoder(x_private)
            z_private = mu_private + sigma_private * torch.randn_like(sigma_private)
        else:
            raise ValueError(defense)

        return z_private.detach()

class Attacker(nn.Module):
    def __init__(self, top_model, decoder):
        super(Attacker, self).__init__()
        self.top_model = top_model.to(device)
        self.decoder = decoder.to(device)
        self.accuracy_criterion = torch.nn.CrossEntropyLoss()
        self.optimizer_decoder = optim.Adam(self.decoder.parameters(), lr=args.lr)

    def train_step(self, z_private, private_label):
        batch_correct = 0
        self.decoder.train()

        private_label = private_label.to(device)
        outputs = self.decoder(z_private)
        acc_loss = self.accuracy_criterion(outputs, private_label.long())

        # calculate the total correctly classified examples in a batch
        _, predicted = outputs.max(1)
        batch_correct += predicted.eq(private_label).sum().item()

        self.optimizer_decoder.zero_grad()
        acc_loss.backward()
        self.optimizer_decoder.step()

        batch_acc_loss = acc_loss.item()

        return batch_acc_loss, batch_correct

    def batch_test(self, z_private, label, private_label):
        self.top_model.eval()
        self.decoder.eval()
        batch_attack_correct = 0
        batch_top_correct = 0

        private_label = private_label.to(device)
        label = label.to(device)
        top_outputs = self.top_model(z_private)
        decoder_outputs = self.decoder(z_private)
        acc_loss = self.accuracy_criterion(top_outputs, label.long())
        decoder_loss = self.accuracy_criterion(decoder_outputs, private_label.long())

        _, predicted = top_outputs.max(1)
        batch_top_correct += predicted.eq(label).sum().item()
        _, decoder_predicted = decoder_outputs.max(1)
        batch_attack_correct += decoder_predicted.eq(private_label).sum().item()
        cm = confusion_matrix(private_label.cpu(), decoder_predicted.cpu())

        return batch_top_correct, batch_attack_correct, cm

def test(client, attacker, test_loader):
    decoder_correct = 0
    top_correct = 0
    decoder_acc_list = np.array([])
    top_acc_list = np.array([])

    print("TESTING...")
    with torch.no_grad():
        for batch_idx, (x, label, private_label) in enumerate(test_loader):
            # send original data to the fixed encoder and get the intermediate features
            x, label, private_label = x.to(device), label.to(device), private_label.to(device)
            z_private = client.get_feature(x)
            batch_top_correct, batch_attack_correct, batch_cm = attacker.batch_test(z_private, label, private_label)
            decoder_correct += batch_attack_correct
            top_correct += batch_top_correct
            if batch_idx == 0:
                cm = batch_cm
            else:
                cm += batch_cm
    print("top correct: ",top_correct)
    attack_accuracy = decoder_correct/float(len(test_loader.dataset))
    top_accuracy = top_correct/float(len(test_loader.dataset))
    attack_Gmean = sqrt(cm[0][0] / (cm[0][0] + cm[0][1]) * cm[1][1] / (cm[1][0] + cm[1][1]))

    return attack_accuracy, top_accuracy, attack_Gmean

def train_decoder(client, attacker, decoder_loader, test_loader):
    attacker_iterator = iter(decoder_loader)

    train_decoder_acc_list = np.array([])
    train_decoder_loss_list = np.array([])
    test_acc_list = np.array([])
    test_attack_acc_list = np.array([])
    attack_Gmean_list = np.array([])

    logger.info('Iteration \t Train Attacker Loss \t Train Attack Acc \t Test Acc \t Test Attack Acc \t Attack G-mean')

    print("TRAINING DECODER...")
    iterator = list(range(args.iterations))
    train_decoder_correct = 0
    train_decoder_loss = 0.0
    for i in tqdm.tqdm(iterator, total=args.iterations):
        try:
            x, label, private_label = next(attacker_iterator)
            if x.size(0) != args.batch_size:
                attacker_iterator = iter(decoder_loader)
                x, label, private_label = next(attacker_iterator)
        except StopIteration:
            attacker_iterator = iter(decoder_loader)
            x, label, private_label = next(attacker_iterator)

        z_private = client.get_feature(x)
        batch_loss, batch_correct = attacker.train_step(z_private, private_label)

        train_decoder_correct += batch_correct
        train_decoder_loss += batch_loss

        train_decoder_loss_list = np.append(train_decoder_loss_list, batch_loss)
        train_decoder_acc_list = np.append(train_decoder_acc_list, batch_correct / args.batch_size)

        interval = args.test_interval
        if (i+1) % interval == 0:
            test_attack_acc, test_acc, attack_Gmean = test(client, attacker, test_loader)
            test_acc_list = np.append(test_acc_list, test_acc)
            test_attack_acc_list = np.append(test_attack_acc_list, test_attack_acc)
            attack_Gmean_list = np.append(attack_Gmean_list, attack_Gmean)
            print("attack accuracy: ", test_attack_acc)
            print("test accuracy: ", test_acc)
            print("attack G-mean: ", attack_Gmean)
            logger.info('%d \t\t %.4f \t\t %.4f \t\t %.4f \t\t %.4f \t\t %.4f',
                        (i + 1), train_decoder_loss / interval,
                        train_decoder_correct / (interval * args.batch_size), test_acc, test_attack_acc, attack_Gmean
                        )

            train_decoder_correct = 0
            train_decoder_loss = 0.0

    max_test_acc = max(test_acc_list)
    max_attack_acc = max(test_attack_acc_list)
    max_attack_Gmean = max(attack_Gmean_list)

    last_test_acc = test_acc_list[-1]
    last_attack_acc = test_attack_acc_list[-1]
    last_attack_Gmean = attack_Gmean_list[-1]

    print("last test acc: ", last_test_acc)
    print("max attack acc: ", max_attack_acc)
    print("max attack G-mean: ", max_attack_Gmean)
    logger.info('Last test accuracy: \t %.4f', last_test_acc)
    logger.info('Max test attack accuracy: \t %.4f', max_attack_acc)
    logger.info('Max attack G-mean: \t %.4f', max_attack_Gmean)

    plot_and_save_figure(train_decoder_loss_list, [], 'batch', 'loss', 'train decoder loss', figure_save_path, 'train_decoder_loss.png')
    plot_and_save_figure(train_decoder_acc_list, [], 'batch', 'accuracy', 'attack accuracy', figure_save_path, 'train_decoder_accuracy.png')

    return last_test_acc, max_attack_acc, max_attack_Gmean


if __name__ == '__main__':
    #p='./celebA_results_413/utility=BlackHair_privacy=Male/model-idx=2/seed=1_pr=0.05_defense=GMM-RO_iter=3000_bs=500_loss=1_sigma=10_pref=0.05_eps=0.01_wi=500_pc=0.005'
    #directory = './UTKFace_NewGMMLoss_average_results/cloud=gender_attack=white/model-idx=3/defense=GMM-ADV_iter=2000_bs=500_sigma=1_acc-w=0.9_adv-w=0.5'
    directory = args.dir
    split_directory = directory.split('/')
    print(split_directory)
    #'./celebA_results\utility=BlackHair_privacy=Male\model-idx=2\seed=1_pr=0.05_defense=None_iter=3000_bs=500'

    dataset = split_directory[1].split('_')[0]
    utility_task = split_directory[2].split('_')[0].split('=')[1]
    privacy_task = split_directory[2].split('_')[1].split('=')[1]
    model_idx = int(split_directory[3].split('_')[0].split('=')[1])
    defense = split_directory[4].split('_')[2].split('=')[1]

    print("dataset: ", dataset)
    print("cloud: ", utility_task)
    print("attack: ", privacy_task)
    print("defense method: ", defense)
    print("model index: ", model_idx)

    if dataset == 'UTKFace':
        idx = 2000
    elif dataset == 'celebA':
        idx = 2000

    if defense == 'GMM-LC' or defense == 'GMM-RO' or defense == 'GMM-ADV' or defense == 'Noisy':
        sigma = float(split_directory[4].split('_')[6].split('=')[1])
        print("sigma: ",sigma)

    encoder_path = directory + '/models/encoder_%s.pkl'%(str(idx))
    top_model_path = directory + '/models/top_model_%s.pkl'%(str(idx))
    #save_path = directory + '/blackbox-results_iterations=%s' % (str(args.iterations))
    figure_save_path = directory + '/figures/Black_box_loss'
    #save_path = directory + '/blackbox'
    result_file = directory + '/logs/result.txt'

    makedir(figure_save_path)

    logfile = directory + '/logs/black_box_log.txt'
    if os.path.exists(logfile):
        os.remove(logfile)

    logging.basicConfig(format='[%(asctime)s] - %(message)s',
                        datefmt='%Y/%m/%d %H:%M:%S',
                        level=logging.INFO,
                        filename=logfile)

    logger.info(args)

    batch_size = args.batch_size
    iterations = args.iterations

    train_loader, test_loader, label_num, private_label_num = load_main_dataset(dataset, utility_task,
                                                                                            privacy_task, batch_size)
    decoder_loader = load_attacker_dataset(dataset, utility_task, privacy_task, batch_size)

    if model_idx == 1:
        Encoder = Encoder1
        InfoEncoder = InfocensorEncoder1
        Classifier = Classifier1
    elif model_idx == 2:
        Encoder = Encoder2
        InfoEncoder = InfocensorEncoder2
        Classifier = Classifier2
    elif model_idx == 3:
        Encoder = Encoder3
        InfoEncoder = InfocensorEncoder3
        Classifier = Classifier3
    else:
        raise ValueError(model_idx)

    if defense == 'Infocensor':
        encoder = InfoEncoder()
    else:
        encoder = Encoder()

    top_model = Classifier(label_num)
    decoder = Classifier(private_label_num)
    encoder.load_state_dict(torch.load(encoder_path))
    top_model.load_state_dict(torch.load(top_model_path))

    client = Client(encoder)
    attacker = Attacker(top_model, decoder)

    last_test_acc, max_attack_acc, max_attack_Gmean = train_decoder(client, attacker, decoder_loader, test_loader)

    f = open(result_file, 'a')
    f.write(str(last_test_acc) + '\n')
    f.write(str(max_attack_acc) + '\n')
    f.write(str(max_attack_Gmean) + '\n')
    f.close()

    #attack_accuracy, top_accuracy = test(client, attacker, test_loader)

    print("cloud task: ", utility_task)
    print("attack task: ", privacy_task)
