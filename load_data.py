import torch
import numpy as np
import pickle
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import time

seed = 1
torch.manual_seed(seed)
np.random.seed(seed)

def load_main_dataset(dataset_name, label_name, private_attribute_name, batch_size):

    if dataset_name == 'celebA':
        attribute_list = ['5oClock_Shadow', 'ArchedEyebrows', 'Attractive', 'BagsUnderEyes', 'Bald', 'Bangs',
                          'BigLips', 'BigNose', 'BlackHair', 'BlondHair', 'Blurry', 'BrownHair', 'BushyEyebrows',
                          'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'HeavyMakeup',
                          'HighCheekbones', 'Male', 'MouthSlightlyOpen', 'Mustache', 'NarrowEyes', 'NoBeard',
                          'OvalFace', 'PaleSkin', 'PointyNose', 'RecedingHairline', 'RosyCheeks', 'Sideburns',
                          'Smiling', 'StraightHair', 'WavyHair', 'WearingEarrings', 'WearingHat',
                          'WearingLipstick', 'WearingNecklace', 'WearingNecktie', 'Young']
    elif dataset_name == 'UTKFace':
        attribute_list = ['age', 'gender', 'race', 'white']
    else:
        raise ValueError(dataset_name)

    print('dataset: ', dataset_name)
    print('utility task: ', label_name)
    print('privacy task: ', private_attribute_name)

    if label_name not in attribute_list:
        raise ValueError(label_name)
    if private_attribute_name not in attribute_list:
        raise ValueError(private_attribute_name)
    label_idx = attribute_list.index(label_name)
    private_attribute_idx = attribute_list.index(private_attribute_name)
    
    #print("private attribute number: ",private_attribute_number)

    dataPath = './Data/' + dataset_name + '/'
    f1 = open(dataPath + 'train_data_clip.pickle', 'rb')
    f2 = open(dataPath + 'train_label_clip.pickle', 'rb')
    f3 = open(dataPath + 'test_data_clip.pickle', 'rb')
    f4 = open(dataPath + 'test_label_clip.pickle', 'rb')

    X_train = np.array(pickle.load(f1), dtype='float32')
    y_train = pickle.load(f2)
    X_test = np.array(pickle.load(f3), dtype='float32')
    y_test = pickle.load(f4)

    f1.close()
    f2.close()
    f3.close()
    f4.close()

    train_data = torch.from_numpy(X_train)
    train_label = torch.from_numpy(y_train[:,label_idx])
    test_data = torch.from_numpy(X_test)
    test_label = torch.from_numpy(y_test[:,label_idx])
    train_private_label = torch.from_numpy(y_train[:,private_attribute_idx])
    test_private_label = torch.from_numpy(y_test[:,private_attribute_idx])

    #print(len(np.unique(np.array(train_label))))
    label_num = len(np.unique(np.array(train_label)))
    private_label_num = len(np.unique(np.array(train_private_label)))

    print("train label ratio: ",Counter(np.array(train_label)))
    print("test label ratio: ",Counter(np.array(test_label)))
    print("train private label ratio: ",Counter(np.array(train_private_label)))
    print("test private label ratio: ",Counter(np.array(test_private_label)))

    train_loader = construct_data_loader(train_data, train_label, train_private_label, batch_size)
    test_loader = construct_data_loader(test_data, test_label, test_private_label, batch_size)

    return train_loader, test_loader, label_num, private_label_num

def load_attacker_dataset(dataset_name, label_name, private_attribute_name, batchsize):
    if dataset_name == 'celebA':
        attribute_list = ['5oClock_Shadow', 'ArchedEyebrows', 'Attractive', 'BagsUnderEyes', 'Bald', 'Bangs',
                          'BigLips', 'BigNose', 'BlackHair', 'BlondHair', 'Blurry', 'BrownHair', 'BushyEyebrows',
                          'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'HeavyMakeup',
                          'HighCheekbones', 'Male', 'MouthSlightlyOpen', 'Mustache', 'NarrowEyes', 'NoBeard',
                          'OvalFace', 'PaleSkin', 'PointyNose', 'RecedingHairline', 'RosyCheeks', 'Sideburns',
                          'Smiling', 'StraightHair', 'WavyHair', 'WearingEarrings', 'WearingHat',
                          'WearingLipstick', 'WearingNecklace', 'WearingNecktie', 'Young']
    elif dataset_name == 'UTKFace':
        attribute_list = ['age', 'gender', 'race', 'white']
    else:
        raise ValueError(dataset_name)

    print('utility task: ', label_name)
    print('privacy task: ', private_attribute_name)
    if label_name not in attribute_list:
        raise ValueError(label_name)
    if private_attribute_name not in attribute_list:
        raise ValueError(private_attribute_name)
    label_idx = attribute_list.index(label_name)
    private_attribute_idx = attribute_list.index(private_attribute_name)

    dataPath = './Data/' + dataset_name + '/'
    f1 = open(dataPath + 'decoder_data_clip.pickle', 'rb')
    f2 = open(dataPath + 'decoder_label_clip.pickle', 'rb')
    X_train = np.array(pickle.load(f1), dtype='float32')
    y_train = pickle.load(f2)
    f1.close()
    f2.close()

    train_data = torch.from_numpy(X_train)
    train_label = torch.from_numpy(y_train[:, label_idx])
    train_private_label = torch.from_numpy(y_train[:, private_attribute_idx])
    decoder_loader = construct_data_loader(train_data, train_label, train_private_label,batchsize)

    return decoder_loader

# this data loader only contains data and accuracy label
def construct_data_loader(data, label, private_label, batch_size):
    dataset = TensorDataset(data, label, private_label)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    return data_loader




