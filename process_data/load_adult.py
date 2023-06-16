import pandas as pd
import numpy as np

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def data_process(df,attr,model):
    # df = pd.read_csv("../adult_dataset/adult.data",header = None, names = ['age', 'workclass', 'fnlwgt', 'education',
    #                                                                        'education-num', 'marital-status', 'occupation',
    #                                                                        'relationship',  'race', 'sex', 'capital-gain',
    #                                                                        'capital-loss', 'hours-per-week', 'native-country',
    #                                                                        'income'])
    # print('Original df:',df)   #[32561,15]

    # 缺省值处理
    df.replace(" ?", pd.NaT, inplace=True)
    if model == 'train':
        df.replace(" >50K", 1, inplace=True)
        df.replace(" <=50K", 0, inplace=True)
        if attr == 'relationship':
            # Sensitive attr：relationship:Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried
            df.replace(" Wife", 0, inplace=True)
            df.replace(" Own-child", 1, inplace=True)
            df.replace(" Husband", 2, inplace=True)
            df.replace(" Not-in-family", 3, inplace=True)
            df.replace(" Other-relative", 4, inplace=True)
            df.replace(" Unmarried", 5, inplace=True)
        elif attr == 'sex':
            # Sensitive attr：sex
            df.replace(" Male", 1, inplace=True)
            df.replace(" Female", 0, inplace=True)
        elif attr == 'education':
            # Sensitive attr：education  Doctorate, 5th-6th, Preschool
            df.replace(" Bachelors", 0, inplace=True)
            df.replace(" Some-college", 1, inplace=True)
            df.replace(" 11th", 2, inplace=True)
            df.replace(" HS-grad", 3, inplace=True)
            df.replace(" Prof-school", 4, inplace=True)
            df.replace(" Assoc-acdm", 5, inplace=True)
            df.replace(" 9th", 6, inplace=True)
            df.replace(" 7th-8th", 7, inplace=True)
            df.replace(" 12th", 8, inplace=True)
            df.replace(" Masters", 9, inplace=True)
            df.replace(" 1st-4th", 10, inplace=True)
            df.replace(" 10th", 11, inplace=True)
            df.replace(" Doctorate", 12, inplace=True)
            df.replace(" 5th-6th", 13, inplace=True)
            df.replace(" Preschool", 14, inplace=True)
            df.replace(" Assoc-voc", 15, inplace=True)

    if model == 'test':
        df.replace(" >50K.", 1, inplace=True)
        df.replace(" <=50K.", 0, inplace=True)
        if attr == 'relationship':
            # Sensitive attr：relationship:Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried
            df.replace(" Wife", 0, inplace=True)
            df.replace(" Own-child", 1, inplace=True)
            df.replace(" Husband", 2, inplace=True)
            df.replace(" Not-in-family", 3, inplace=True)
            df.replace(" Other-relative", 4, inplace=True)
            df.replace(" Unmarried", 5, inplace=True)
        elif attr == 'sex':
            # Sensitive attr：sex
            df.replace(" Male", 1, inplace=True)
            df.replace(" Female", 0, inplace=True)
        elif attr == 'education':
            # Sensitive attr：education
            df.replace(" Bachelors", 0, inplace=True)
            df.replace(" Some-college", 1, inplace=True)
            df.replace(" 11th", 2, inplace=True)
            df.replace(" HS-grad", 3, inplace=True)
            df.replace(" Prof-school", 4, inplace=True)
            df.replace(" Assoc-acdm", 5, inplace=True)
            df.replace(" 9th", 6, inplace=True)
            df.replace(" 7th-8th", 7, inplace=True)
            df.replace(" 12th", 8, inplace=True)
            df.replace(" Masters", 9, inplace=True)
            df.replace(" 1st-4th", 10, inplace=True)
            df.replace(" 10th", 11, inplace=True)
            df.replace(" Doctorate", 12, inplace=True)
            df.replace(" 5th-6th", 13, inplace=True)
            df.replace(" Preschool", 14, inplace=True)
            df.replace(" Assoc-voc", 15, inplace=True)

    trans = {'workclass': df['workclass'].mode()[0], 'occupation': df['occupation'].mode()[0],
             'native-country': df['native-country'].mode()[0]}
    df.fillna(trans, inplace=True)
    # 删除无用数据
    df.drop('fnlwgt', axis=1, inplace=True)
    df.drop('capital-gain', axis=1, inplace=True)
    df.drop('capital-loss', axis=1, inplace=True)

    # # 判断并收集每一列的数据类型
    # for i in df.columns:
    #     print(f"The column {i}'s dtype is {df.loc[:, i].dtype}")
    # print('Processed df_1:', df)

    df_object_col = [col for col in df.columns if df[col].dtype.name == 'object']
    df_int_col = [col for col in df.columns if df[col].dtype.name != 'object' and col != 'income' and col != attr]


    y = df["income"]
    s = df[attr]
    # a = pd.Series(s)
    # print('sensitive attr:',a.value_counts())
    ## sex:male = 0.6670   relationship:Husband = 0.40    education:HS-grad = 0.32
    ## race:white = 0.85     workclass:Private = 0.75

    X = pd.concat([df[df_int_col], pd.get_dummies(df[df_object_col])], axis=1)

    return X, y, s


def add_missing_columns(d, columns):
    missing_col = set(columns) - set(d.columns)
    for col in missing_col:
        d[col] = 0


def fix_columns(d, columns):
    add_missing_columns(d, columns)
    assert (set(columns) - set(d.columns) == set())
    d = d[columns]
    return d


class Feeder(torch.utils.data.Dataset):

    def __init__(self,
                 X, y, s,
                 is_training = True,
                 normalization=True
                 ):

        self.X, self.y, self.s = X, y, s
        self.is_training = is_training


    def __len__(self):
        return self.X.shape[0]

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data, target, slabel = self.X[index], self.y[index], self.s[index]

        return data, target, slabel




def load_adult(attr='sex', batch_size=128):
    df_train = pd.read_csv("../data/adult/adult.data", header=None,
                           names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                                  'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                                  'hours-per-week', 'native-country', 'income'])
    df_test = pd.read_csv("../data/adult/adult.test", header=None, skiprows=1,
                          names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                                 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                                 'hours-per-week', 'native-country', 'income'])

    X_train, y_train, s_train = data_process(df_train, attr,'train')
    X_test, y_test, s_test = data_process(df_test, attr,'test')

    #         进行独热编码对齐
    X_test = fix_columns(X_test, X_train.columns)

    X_train = X_train.apply(lambda x: (x - x.mean()) / x.std())
    X_test = X_test.apply(lambda x: (x - x.mean()) / x.std())


    y_train, y_test = np.array(y_train), np.array(y_test)
    s_train, s_test = np.array(s_train), np.array(s_test)
    X_train, X_test = np.array(X_train, dtype=np.float32), np.array(X_test, dtype=np.float32)

    # 测试集处理
    isnan = np.isnan(X_test)
    X_test[np.where(isnan)] = 0.0

    y_test = torch.tensor(y_test, dtype=torch.int64)
    X_test = torch.FloatTensor(X_test)
    s_test = torch.tensor(s_test, dtype=torch.int64)

    # 训练集划分: 前百分之八十的数据作为训练集，其余作为验证集
    y_valid = torch.tensor(y_train, dtype=torch.int64)[int(len(y_train) * 0.8):]
    X_valid = torch.FloatTensor(X_train)[int(len(X_train) * 0.8):]
    s_valid = torch.tensor(s_train, dtype=torch.int64)[int(len(s_train) * 0.8):]

    y_train = torch.tensor(y_train, dtype=torch.int64)[: int(len(y_train) * 0.8)]
    X_train = torch.FloatTensor(X_train)[: int(len(X_train) * 0.8)]
    s_train = torch.tensor(s_train, dtype=torch.int64)[: int(len(s_train) * 0.8)]
    # print(X_train.shape,X_valid.shape,X_test.shape)  ## train：26048；valid：6513；test：16281


    train_loader = DataLoader(
        dataset=Feeder(X=X_train, y=y_train, s=s_train),
        batch_size=batch_size, shuffle=True,
        num_workers=0,
        drop_last=True)

    valid_loader = DataLoader(
        dataset=Feeder(X=X_valid, y=y_valid, s=s_valid),
        batch_size=batch_size, shuffle=False,
        num_workers=0,
        drop_last=True)

    # For attack
    test_loader = DataLoader(
        dataset=Feeder(X=X_test, y=y_test, s=s_test),  # is_training=False
        batch_size=batch_size, shuffle=True,
        num_workers=0)

    return train_loader,valid_loader,test_loader,X_train.shape[1], int(y_train.max()+1),int(s_train.max()+1)



if __name__ == '__main__':
    train_loader, valid_loader, test_loader, input_dim, target_num_classes, sensitive_num_classes = load_adult(attr='relationship',
                                                                                                               batch_size=300)
    print(len(test_loader))
    print(input_dim)
    print(target_num_classes)
    print(sensitive_num_classes)
