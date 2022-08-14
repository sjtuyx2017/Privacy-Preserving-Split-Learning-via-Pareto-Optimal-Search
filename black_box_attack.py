import numpy as np
import os
from torch import nn, optim
import torch
from load_data import load_main_dataset, load_attacker_dataset, construct_data_loader
from models import (
    Encoder1, Classifier1,
    Encoder2, Classifier2,
    Encoder3, Classifier3,
)
import collections
from utils.tools import smooth, get_gaussian, plot_and_save_figure, makedir
import matplotlib.pyplot as plt
import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--batch_size', default=200, type=int)
parser.add_argument('--iterations', default=1000, type=int)
parser.add_argument('--seed', default=1, type=int)

args = parser.parse_args()
print(args)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# torch.manual_seed(args.seed)
# np.random.seed(args.seed)
# if torch.cuda.is_available():
#     device = 'cuda'
#     torch.cuda.manual_seed(args.seed)
# else:
#     device = 'cpu'



class Client(nn.Module):
    def __init__(self, encoder):
        super(Client, self).__init__()
        self.encoder = encoder.to(device)

    def get_feature(self, x_private):
        self.encoder.eval()
        if defense == 'None' or defense == 'ADV':
            x_private = x_private.to(device)
            z_private = self.encoder(x_private)
        elif defense == 'GMM-LC' or defense == 'GMM-EPO':
            x_private = x_private.to(device)
            mu_private = self.encoder(x_private)
            z_private = get_gaussian(mu_private, sigma, device)

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

    def test(self, z_private, label, private_label):
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


        return batch_top_correct, batch_attack_correct

def train_decoder(client, attacker, decoder_loader):
    attacker_iterator = iter(decoder_loader)
    decoder_correct = 0
    train_decoder_acc_list = np.array([])
    train_decoder_loss_list = np.array([])

    print("TRAINING DECODER...")
    iterator = list(range(args.iterations))
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
        batch_acc_loss, batch_correct = attacker.train_step(z_private, private_label)

        train_decoder_loss_list = np.append(train_decoder_loss_list, batch_acc_loss)
        train_decoder_acc_list = np.append(train_decoder_acc_list, batch_correct / args.batch_size)

    plot_and_save_figure(train_decoder_loss_list, [], 'batch', 'loss', 'train decoder loss', save_path, 'train_decoder_loss.png')
    plot_and_save_figure(train_decoder_acc_list, [], 'batch', 'accuracy', 'attack accuracy', save_path, 'train_decoder_accuracy.png')


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
            batch_top_correct, batch_attack_correct = attacker.test(z_private, label, private_label)
            decoder_correct += batch_attack_correct
            top_correct += batch_top_correct
    print("top correct: ",top_correct)
    attack_accuracy = decoder_correct/float(len(test_loader.dataset))
    top_accuracy = top_correct/float(len(test_loader.dataset))

    return attack_accuracy, top_accuracy


if __name__ == '__main__':

    directory = './celebA_results/cloud=BlackHair_attack=Male/model-idx=3/defense=GMM-EPO_iteration=3000_sigma=0.1_preference=0.8_eps=0.001_pc=1_ascent=True_mode=None/attack=DSA_cal-acc=False_top-ratio=0.5'
    split_directory = directory.split('/')
    print(split_directory)

    dataset = split_directory[1].split('_')[0]
    cloud_task = split_directory[2].split('_')[0].split('=')[1]
    attack_task = split_directory[2].split('_')[1].split('=')[1]
    model_idx = int(split_directory[3].split('_')[0].split('=')[1])
    defense = split_directory[4].split('_')[0].split('=')[1]

    print(dataset)
    print(cloud_task)
    print(attack_task)
    print(defense)
    print(model_idx)

    if defense == 'GMM-LC' or defense == 'GMM-EPO':
        sigma = float(split_directory[4].split('_')[2].split('=')[1])

    encoder_path = directory + '/models/encoder_3000.pkl'
    top_model_path = directory + '/models/top_model_3000.pkl'
    #save_path = directory + '/blackbox-results_iterations=%s' % (str(args.iterations))
    save_path = directory + '/b'

    makedir(save_path)

    batch_size = args.batch_size
    iterations = args.iterations

    train_loader, test_loader, label_num, private_label_num = load_main_dataset(dataset, cloud_task, attack_task, batch_size)
    decoder_loader = load_attacker_dataset(dataset, cloud_task, attack_task, batch_size)

    if model_idx == 1:
        Encoder = Encoder1
        Classifier = Classifier1
    elif model_idx == 2:
        Encoder = Encoder2
        Classifier = Classifier2
    elif model_idx == 3:
        Encoder = Encoder3
        Classifier = Classifier3
    else:
        raise ValueError(model_idx)


    encoder = Encoder()
    top_model = Classifier(label_num)
    decoder = Classifier(private_label_num)
    encoder.load_state_dict(torch.load(encoder_path))
    top_model.load_state_dict(torch.load(top_model_path))

    client = Client(encoder)
    attacker = Attacker(top_model, decoder)

    train_decoder(client, attacker, decoder_loader)
    attack_accuracy, top_accuracy = test(client, attacker, test_loader)

    log_path = save_path + '/print_log.txt'
    fw = open(log_path, 'w')

    fw.write("top accuracy: " + str(top_accuracy))
    fw.write('\n')
    fw.write("attack accuracy: " + str(attack_accuracy))
    fw.write('\n')

    fw.close()

    print("attack accuracy: ",attack_accuracy)
    print("top accuracy: ",top_accuracy)


    print("cloud task: ", cloud_task)
    print("attack task: ", attack_task)
