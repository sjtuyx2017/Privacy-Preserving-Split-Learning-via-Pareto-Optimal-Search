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
parser.add_argument('--train_num',  default=12000, type=int)
parser.add_argument('--test_num',  default=3000, type=int)
parser.add_argument('--decoder_num',  default=6000, type=int)
parser.add_argument('--clip_shape',  default=64, type=int)
args = parser.parse_args()
print(args)


image_set = []

directory_name = './Data/UTKFace_original'

total = 23705
num = args.train_num + args.test_num + args.decoder_num
current = 0

print(os.listdir(directory_name)[0])
path_list = os.listdir(directory_name)
print(type(path_list))
random.shuffle(path_list)
#path_list.sort(key=lambda x:int(x[:-4]))

age_list = []
gender_list = []
race_list = []
white_list = []
labels = []

for filename in path_list:
    current += 1
    print(current)
    print(filename)
    label = list(filename.split('_'))[:3]
    age, gender, race = int(label[0]), int(label[1]), int(label[2])
    print("age: ",age)
    print("gender: ", gender)
    print("race: ", race)
    if age <= 10:
        age = 0
    elif age > 10 and age <= 20:
        age = 1
    elif age > 20 and age <= 35:
        age = 2
    elif age > 35 and age <= 60:
        age = 3
    else:
        age = 4
    if race == 0:
        white = 1
    else:
        white = 0
    age_list.append(age)
    gender_list.append(gender)
    race_list.append(race)
    white_list.append(white)

    img = cv2.imread(directory_name + "/" + filename,0)
    img = cv2.resize(img,(args.clip_shape, args.clip_shape))
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
age_list = np.array(age_list)
gender_list = np.array(gender_list)
race_list = np.array(race_list)
labels = np.stack((age_list, gender_list, race_list, white_list)).T
print(labels.shape)
print("labels: ",labels)
print(Counter(age_list))
print(Counter(gender_list))
print(Counter(race_list))
print(Counter(white_list))

X_train = image_set[:args.train_num]
X_test = image_set[args.train_num : args.train_num + args.test_num]
y_train = np.array(labels[:args.train_num])
y_test = np.array(labels[args.train_num : args.train_num + args.test_num])

print(X_train.shape)
print(y_train.shape)

decoder_X = image_set[args.train_num + args.test_num : args.train_num + args.test_num + args.decoder_num]
decoder_y = np.array(labels[args.train_num + args.test_num : args.train_num + args.test_num + args.decoder_num])


f1 = open('./Data/UTKFace/train_data_clip.pickle','wb')
f2 = open('./Data/UTKFace/train_label_clip.pickle','wb')
f3 = open('./Data/UTKFace/test_data_clip.pickle','wb')
f4 = open('./Data/UTKFace/test_label_clip.pickle','wb')
f5 = open('./Data/UTKFace/decoder_data_clip.pickle','wb')
f6 = open('./Data/UTKFace/decoder_label_clip.pickle','wb')

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
