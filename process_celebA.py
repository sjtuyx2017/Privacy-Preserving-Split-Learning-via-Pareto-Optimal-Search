import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import random
import matplotlib
matplotlib.use('TkAgg')
from  matplotlib import pyplot as plt
from PIL import Image
from collections import Counter
import pickle
import cv2
import os
import argparse
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--train_num',  default=40000, type=int)
parser.add_argument('--test_num',  default=10000, type=int)
parser.add_argument('--decoder_num',  default=20000, type=int)
args = parser.parse_args()
print(args)


image_set = []

directory_name = './Data/celebA_original'

total = 202599
num = args.train_num + args.test_num + args.decoder_num
current = 0

print(os.listdir(directory_name)[0])
path_list = os.listdir(directory_name)
path_list.sort(key=lambda x:int(x[:-4]))
print(path_list[0])

for filename in path_list:
    current += 1
    print(current)
    print(filename)
    img = cv2.imread(directory_name + "/" + filename,0)
    img = cv2.resize(img,(64,64))
    img = (img / (255/2) - 1)
    img = np.clip(img,-1,1)
    #img = img/255.
    print(img.shape)
    print(type(img))
    #np.reshape(img,(64,64,3))
    #img = tf.image.resize(img, (64, 64))
    #cv2.imshow(filename, img)
    #cv2.waitKey(0)
    #image_set = np.append(image_set,img)
    image_set.append(img)
    if current == num:
        break


image_set = np.array(image_set).reshape((num,1,64,64))
print(image_set.shape)


labels = pd.read_csv('./Data/list_attr_celeba.csv')

labels.drop('image_id',axis=1, inplace=True)
labels.replace({-1:0,1:1},inplace=True)

labels = np.array(labels)
print("labels: ",labels)
print(labels.shape)

X_train = image_set[:args.train_num]
X_test = image_set[args.train_num : args.train_num + args.test_num]
y_train = np.array(labels[:args.train_num])
y_test = np.array(labels[args.train_num : args.train_num + args.test_num])

print(X_train.shape)
print(y_train.shape)

decoder_X = image_set[args.train_num + args.test_num : args.train_num + args.test_num + args.decoder_num]
decoder_y = np.array(labels[args.train_num + args.test_num : args.train_num + args.test_num + args.decoder_num])


f1 = open('./Data/celebA/train_data_clip.pickle','wb')
f2 = open('./Data/celebA/train_label_clip.pickle','wb')
f3 = open('./Data/celebA/test_data_clip.pickle','wb')
f4 = open('./Data/celebA/test_label_clip.pickle','wb')
f5 = open('./Data/celebA/decoder_data_clip.pickle','wb')
f6 = open('./Data/celebA/decoder_label_clip.pickle','wb')

pickle.dump(X_train,f1,protocol=4)
pickle.dump(y_train,f2)
pickle.dump(X_test,f3,protocol=4)
pickle.dump(y_test,f4)
pickle.dump(decoder_X,f5,protocol=4)
pickle.dump(decoder_y,f6)


f1.close()
f2.close()
f3.close()
f4.close()
f5.close()
f6.close()

'''
labels = pd.read_csv('./data/list_attr_celeba.csv')
labels.replace({-1:0,1:1},inplace=True)
attribute_list = list(labels.columns)
attribute_list.remove('image_id')
print(attribute_list)

for item in attribute_list:
    label = labels[item]
    print(item,(Counter(np.array(label))[0]/202599))
'''