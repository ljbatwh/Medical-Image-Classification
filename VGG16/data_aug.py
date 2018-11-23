import os
from glob import glob
import random
import pandas as pd
import numpy as np
#import matplotlib.gridspec as gridspec
#import seaborn as sns
import zlib
import itertools
import sklearn
import itertools
import scipy
import skimage
from skimage.transform import resize
import csv
from tqdm import tqdm
from sklearn import model_selection
from sklearn.model_selection import train_test_split, learning_curve,KFold,cross_val_score,StratifiedKFold
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
import keras
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Lambda, MaxPool2D, BatchNormalization
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import models, layers, optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils import class_weight
from keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta
from keras.models import Sequential, model_from_json
from keras.layers import Activation,Dense, Dropout, Flatten, Conv2D, MaxPool2D,MaxPooling2D,AveragePooling2D, BatchNormalization
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras import backend as K
from keras.preprocessing import image
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.layers import Dense , Activation
from keras.layers import Dropout , GlobalAveragePooling2D
from keras.layers import Flatten
import pickle
from os.path import basename

def train_dir(args):
    return args.data_dir+"/train/"

def get_generator(data_directory,input_size,batch_size,save_dir):
        
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                samplewise_center=True,
                samplewise_std_normalization=True,
                #zca_whitening=True,
                #zca_epsilon=1e-6,
                rotation_range=3,
                width_shift_range=0.05,
                height_shift_range=0.05,
                shear_range=0.05,
                zoom_range=0.05,
                #channel_shift_range=10,
                fill_mode='constant',
                cval=0.,
                horizontal_flip=True,
                vertical_flip=True)
    
    image_resize_height = input_size
    image_resize_width = input_size

    generator = train_datagen.flow_from_directory(
        data_directory,
        #color_mode='grayscale',
        target_size=(image_resize_height, image_resize_width),
        batch_size=batch_size,
        class_mode='categorical',
        save_to_dir=save_dir,
        save_prefix="aug",
        seed=1234)

    #print(generator.class_indices)
    #print(generator.classes)
    #with open(save_dir+"/data.pkl", 'w') as outfile:
    #    outfile.dump(generator.filenames)
    #    outfile.dump(generator.classes)
    #print(generator.filenames)
    return generator

def gen_aug(args):
    generator = get_generator(train_dir(args),args.input_size,args.batch_size, args.save_dir )
    count = 0
    class_map = {v: k for k, v in generator.class_indices.items()}
    class_count = {k:0 for k,v in class_map.items()}
    
    for x, y in tqdm(generator):
        #print(x.shape)
        #print(y)
        i = np.argmax(y)
        #print(class_map[np.argmax(y)])
        class_count[i] = class_count[i]+1
        count += 1
        if count >= args.aug_num:
            break
    print(str(class_count))
    move_file_to_dir(args,generator.class_indices,generator.classes,generator.filenames)

def move_file_to_dir(args,class_indices,classes,filenames):
    for k in class_indices:
        if not os.path.exists(args.save_dir+"/"+k):
            os.makedirs(args.save_dir+"/"+k)
    current_files = [d for d in os.listdir(args.save_dir) if d.endswith('.png')]
    print(len(current_files))
    for f in tqdm(current_files):
        fname = basename(f)
        #print(fname)
        i = int(fname.split('_')[1])
        #print(i)
        cls = classes[i]
        on = filenames[i]
        newname = on[:-4]+fname
        #print(fname,i,cls,on,newname)
        os.rename(args.save_dir+"/"+fname, args.save_dir+"/"+newname)

    
if __name__ == "__main__":
    import os
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="data generate.")
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--save_dir', default='./gen_aug')
    parser.add_argument('--data_dir', default='../chest_xray',
                        help="the base of data dir")    
    parser.add_argument('--input_size', default=640,
                        help="the size of input image, default value is 299")    
    parser.add_argument('--aug_num', default=20000, type=int,
                        help="the number of aug image, default value is nb_train_samples")     

    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    args_file = open(args.save_dir+"/args_file.txt","w")
    args_file.write(str(args)); 
    args_file.close();

    if not os.path.exists(args.data_dir):
        print(args.data_dir+" is not exist")
        sys.exit()

    gen_aug(args)

