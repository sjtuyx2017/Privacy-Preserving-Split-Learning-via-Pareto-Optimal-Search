import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

DATA_DIR = '../data/health/'
HEALTH_PATH = DATA_DIR + 'health.csv'


def create_health_dataset(attr='age', binarize=True):
    d = pd.read_csv(HEALTH_PATH)
    d = d[d['YEAR_t'] == 'Y3']
    sex = d['sexMISS'] == 0
    age = d['age_MISS'] == 0
    d = d.drop(['DaysInHospital', 'MemberID_t', 'YEAR_t'], axis=1)
    d = d[sex & age]

    def gather_labels(df):
        labels = []
        for j in range(df.shape[1]):
            if type(df[0, j]) is str:
                labels.append(np.unique(df[:, j]).tolist())
            else:
                labels.append(np.median(df[:, j]))
        return labels

    ages = d[['age_%d5' % (i) for i in range(0, 9)]]
    sexs = d[['sexMALE', 'sexFEMALE']]
    charlson = d['CharlsonIndexI_max']

    x = d.drop(
        ['age_%d5' % (i) for i in range(0, 9)] + ['sexMALE', 'sexFEMALE', 'CharlsonIndexI_max', 'CharlsonIndexI_min',
                                                  'CharlsonIndexI_ave', 'CharlsonIndexI_range', 'CharlsonIndexI_stdev',
                                                  'trainset'], axis=1).values

    labels = gather_labels(x)
    xs = np.zeros_like(x)
    for i in range(len(labels)):
        xs[:, i] = x[:, i] > labels[i]

    col_indices = np.nonzero(np.mean(xs, axis=0) > 0.05)[0]
    x = x[:, col_indices]
    if binarize:
        x = xs[:, col_indices].astype(np.float32)
    else:
        x = (x - np.min(x, axis=0)) / np.max(x, axis=0)
        # mn = np.mean(x, axis=0)
        # std = np.std(x, axis=0)
        # x = whiten(x, mn, std)

    u = sexs.values[:, 0]
    v = np.argmax(ages.values, axis=1)
    a = u if attr == 'gender' else v            # sensitive attr(u:gender  v:age)
    # s = pd.Series(a)
    # print('sensitive attr:',s.value_counts())

    y = (charlson.values > 0).astype(np.int64)  # target attr

    return x, y, a


class Feeder(torch.utils.data.Dataset):

    def __init__(self,
                 X, y, s,
                 is_training = True,
                 normalization=True
                 ):

        self.X, self.y, self.s = X.astype(np.float32), y, s
        self.is_training = is_training


    def __len__(self):
        return self.X.shape[0]

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data, target, slabel = self.X[index], self.y[index], self.s[index]

        return data, target, slabel


class Feeder2(torch.utils.data.Dataset):

    def __init__(self,
                 X, s,
                 is_training = True,
                 normalization=True
                 ):

        self.X, self.s = X.astype(np.float32), s
        self.is_training = is_training


    def __len__(self):
        return self.X.shape[0]

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data, slabel = self.X[index], self.s[index]
        return data, slabel


def load_health(attr='gender', train_size=0.8, random_seed=0,
                binarize=True, batch_size=128):
    # 加载
    X, y, s = create_health_dataset(attr, binarize)
    # print('X.shape:',X.shape)  # 55924
    # print('gender:', s, s.shape)
    # print('gender_1:', 1-s.sum()/s.shape)
    # 划分
    X_atk = X[:15978,:]
    # print('X_atk.shape:',X_atk.shape)   # 15978
    y_atk = y[:15978]
    s_atk = s[:15978]
    # print('gender_1:', 1 - s_atk.sum() / s_atk.shape)

    X = X[15978:,:]
    # print('X.shape:',X.shape)  # 39946
    y = y[15978:]
    s = s[15978:]

    ## Another
    # X_atk = X[39946:, :]
    # # print('X_atk.shape:',X_atk.shape)   # 15978
    # y_atk = y[39946:]
    # s_atk = s[39946:]
    #
    # X = X[:39946, :]
    # # print('X.shape:',X.shape)  # 39946
    # y = y[:39946]
    # s = s[:39946]

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(X, y, s,
                                                                         train_size=train_size,
                                                                         random_state=random_seed)

    train_loader = DataLoader(
        dataset=Feeder(X=X_train, y=y_train, s=s_train),
        batch_size=batch_size, shuffle=False,
        num_workers=0,
        drop_last=True)

    valid_loader = DataLoader(
        dataset=Feeder(X=X_test, y=y_test, s=s_test),
        batch_size=batch_size, shuffle=False,
        num_workers=0,
        drop_last=True)

    # For attack
    test_loader = DataLoader(
        dataset=Feeder(X=X_atk, y=y_atk, s=s_atk),    # is_training=False
        batch_size=batch_size, shuffle=True,
        num_workers=0)


    ## return train DataLoader, test DataLoader, input_dim, target_num_classes, sensitive_num_classes
    return train_loader, valid_loader, test_loader, X_train.shape[1], y_train.max()+1, s_train.max()+1


if __name__ == '__main__':
    attr = 'gender'
    train_loader, valid_loader, test_loader, _, target_num_classes, sensitive_num_classes = load_health(attr=attr, binarize=False, batch_size=128)
    print(len(train_loader))
    print(target_num_classes)
    print(sensitive_num_classes)
