import os
from glob import glob
import random
import pandas as pd
import numpy as np
import zlib
import itertools
import sklearn
import itertools
import skimage
from skimage.transform import resize
from tqdm import tqdm
from keras.preprocessing import image



def show_img_inline(x):
    import matplotlib.pyplot as plt
    print(x.shape)
    #plt.imshow(x/255.,cmap='gray')
    plt.imshow(x/255.)
    plt.show()

def get_labels_dict(folder):
    dirs = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)) and not d.startswith('.')]
    if len(dirs) == 3:
        return {'NORMAL': 0 , 'BACTERIA': 1, 'VIRUS': 2}
    elif len(dirs) == 2:
        if 'NORMAL' in dirs:
            return {'NORMAL': 0, 'PNEUMONIA':1}
        elif 'VIRUS' in dirs:
            return {'BACTERIA': 0, 'VIRUS': 1}
    elif len(dirs) == 4:
        return {'CNV':0, 'DME':1, 'DRUSEN':2, 'NORMAL':3}
    else:
       os.exit(); 
    
def get_data(folder, dest_size,count_per_class,show_img=False):
    X = []
    y = []
    files=[]
    label_dict = get_labels_dict(folder)

    for folderName in os.listdir(folder):
        if not folderName.startswith('.'):
            label = label_dict[folderName]
            imgfiles =os.listdir(folder + folderName) 
            if(count_per_class != -1):
                imgfiles=imgfiles[:count_per_class]
            for image_filename in tqdm(imgfiles):
                if not image_filename.startswith('.'):
                    img_path = folder + folderName + '/' + image_filename
                    img_file=image.load_img(img_path,grayscale=False, target_size=(dest_size, dest_size,3))
                    if img_file is not None:
                        img_arr = image.img_to_array(img_file)
                        if show_img:
                            show_img_inline(img_arr)
                        X.append(img_arr)
                        y.append(label)
                        if show_img:
                            print(img_path)
                        files.append(img_path)
    X = np.asarray(X)
    y = np.asarray(y)
    return X,y,files

def load_xary_data(train_dir,test_dir,dest_size,load_count=-1,show_img=False):
    x_train, y_train,_ = get_data(train_dir,dest_size,load_count,show_img)
    x_test, y_test,_= get_data(test_dir,dest_size,load_count,show_img)
    return (x_train, y_train), (x_test, y_test)
    
if __name__ == "__main__":
    train_dir = "../../chest_xray_bv/train/"
    test_dir =  "../../chest_xray_bv/test/"
    input_default_size = 320
    (x_train, y_train), (x_test, y_test) = load_xary_data(train_dir,test_dir,input_default_size,2,True)
    print(x_train.shape) 
    print(y_train.shape)
    print(y_train)
    print(x_test.shape)
    print(y_test.shape)
    print(y_test)

