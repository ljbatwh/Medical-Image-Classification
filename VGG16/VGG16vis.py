# -*- coding: utf-8 -*-
import os
from glob import glob
import matplotlib.pyplot as plt
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
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense , Activation
from keras.layers import Dropout , GlobalAveragePooling2D
from keras.layers import Flatten
#from imblearn.over_sampling import RandomOverSampler
#from imblearn.under_sampling import RandomUnderSampler
#from sklearn.metrics import roc_auc_score
#from sklearn.metrics import roc_curve
#from sklearn.metrics import auc

import warnings
warnings.filterwarnings("ignore")

nb_train_samples = 5232
nb_validation_samples = 624
nb_normal_test_sample=234
nb_bacteria_test_sample=242
nb_virus_test_sample=148
nb_pneumonia_test_sample=390
nb_classes = 2


def train_dir(args):
    return args.data_dir+"/train/"

def test_dir(args):
    return args.data_dir+"/test/"



def plotKerasLearningCurve():
    plt.figure(figsize=(10,5))
    metrics = np.load('logs.npy')[()]
    filt = ['acc'] # try to add 'loss' to see the loss learning curve
    for k in filter(lambda x : np.any([kk in x for kk in filt]), metrics.keys()):
        l = np.array(metrics[k])
        plt.plot(l, c= 'r' if 'val' not in k else 'b', label='val' if 'val' in k else 'train')
        x = np.argmin(l) if 'loss' in k else np.argmax(l)
        y = l[x]
        plt.scatter(x,y, lw=0, alpha=0.25, s=100, c='r' if 'val' not in k else 'b')
        plt.text(x, y, '{} = {:.4f}'.format(x,y), size='15', color= 'r' if 'val' not in k else 'b')   
    plt.legend(loc=4)
    plt.axis([0, None, None, None]);
    plt.grid()
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy')

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize = (5,5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_learning_curve(history):
    plt.figure(figsize=(8,8))
    plt.subplot(1,2,1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./accuracy_curve.png')
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./loss_curve.png')
    
def train_generator(args):
    data_directory = train_dir(args)
        
    if(args.aug):
        if args.aug_mode == 2:
            train_datagen = ImageDataGenerator(rescale=1. / 255,
                #samplewise_center=True,
                #samplewise_std_normalization=True,
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
        else:
            transformation_ratio = .05  # how aggressive will be the data augmentation/transformation
            train_datagen = ImageDataGenerator(rescale=1. / 255,
	       rotation_range=transformation_ratio,
	       shear_range=transformation_ratio,
	       zoom_range=transformation_ratio,
	       horizontal_flip=True,
	       vertical_flip=True)
    else:
        train_datagen = ImageDataGenerator(rescale=1. / 255)

    image_resize_height = args.input_size
    image_resize_width = args.input_size

    generator = train_datagen.flow_from_directory(
        data_directory,
        #color_mode='grayscale',
        target_size=(image_resize_height, image_resize_width),
        batch_size=args.batch_size,
        class_mode='categorical',
        seed=1234)
    return generator

def test_generator(args) :
    validation_datagen = ImageDataGenerator(rescale=1. / 255)    
    image_resize_height = args.input_size
    image_resize_width = args.input_size
    data_directory = test_dir(args)
    
    generator = validation_datagen.flow_from_directory(
        data_directory,
        #color_mode='grayscale',
        target_size=(image_resize_height, image_resize_width),
        batch_size=args.batch_size,
        class_mode='categorical')

    return generator

def createModel(pretrainedmodel,args):

    base_model = pretrainedmodel # Topless

    x = Sequential()
    x.add(base_model)
    # Add top layer
    if (args.model == 3):
        x.add(GlobalAveragePooling2D(name='avg_pool'))
        x.add(Dense(512, activation='relu', name='fc1'))
        x.add(Dropout(0.5))
        x.add(Dense(256, activation='relu', name='fc3'))
        x.add(Dropout(0.5))
        x.add(Dense(128, activation='relu', name='fc4'))
        x.add(Dropout(0.5))
    elif (args.model == 2):
        #incenptionv3 original
        x.add(GlobalAveragePooling2D(name='avg_pool'))
    elif (args.model == 4):
        x.add(GlobalAveragePooling2D(name='avg_pool'))
        x.add(Dense(512, activation='relu', name='fc1'))
        x.add(Dropout(0.5))     
    elif (args.model == 5):
        x.add(GlobalAveragePooling2D(name='avg_pool'))
        x.add(Dense(512, activation='relu', name='fc1'))
        x.add(Dropout(0.5))
        x.add(Dense(512, activation='relu', name='fc2'))
        x.add(Dropout(0.5))
        x.add(Dense(256, activation='relu', name='fc3'))
        x.add(Dropout(0.5))
    elif (args.model == 1):
        #VGG original
        x.add(GlobalAveragePooling2D(name='avg_pool'))
        x.add(Dense(4096, activation='relu', name='fc1'))
        x.add(Dense(4096, activation='relu', name='fc2'))
    elif (args.model == 6):
        x.add(GlobalAveragePooling2D(name='avg_pool'))
        x.add(Dense(512, activation='relu', name='fc1'))
        if args.batchnorm:
            x.add(BatchNormalization())        
        x.add(Dropout(args.dropout1))
        x.add(Dense(256, activation='relu', name='fc3'))
        if args.batchnorm:
            x.add(BatchNormalization())    
        x.add(Dropout(args.dropout2))
        x.add(Dense(128, activation='relu', name='fc4'))
        x.add(Dropout(args.dropout3))
    elif (args.model == 7):
        x = Sequential()
        for l in base_model.layers[0:-1]:
            x.add(l)    
        if args.batchnorm:
            x.add(BatchNormalization())  
        x.add(Dropout(args.dropout1))
        x.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))
        x.add(GlobalAveragePooling2D(name='avg_pool'))
        if args.model7_fc1:
            x.add(Dense(16, name='fc1'))
            if args.batchnorm:
                x.add(BatchNormalization())    
            x.add(Activation('relu'))
            x.add(Dropout(args.dropout2))
    elif (args.model==8):
        #VGG add bn
        x.add(GlobalAveragePooling2D(name='avg_pool'))
        x.add(Dense(4096, name='fc1'))
        x.add(BatchNormalization())    
        x.add(Activation('relu'))
        x.add(Dropout(args.dropout1))
        x.add(Dense(2048, name='fc2'))
        x.add(BatchNormalization())    
        x.add(Activation('relu'))
        x.add(Dropout(args.dropout2))
    elif (args.model==9):
        #VGG add bn
        x.add(GlobalAveragePooling2D(name='avg_pool'))
        x.add(Dense(4096, name='fc1'))
        x.add(BatchNormalization())    
        x.add(Activation('relu'))
        x.add(Dropout(args.dropout1))
        x.add(Dense(2048, name='fc2'))
        x.add(BatchNormalization())    
        x.add(Activation('relu'))
        x.add(Dropout(args.dropout1))
        x.add(Dense(2048, name='fc3'))
        x.add(BatchNormalization())    
        x.add(Activation('relu'))
        x.add(Dropout(args.dropout2))
    elif (args.model==10):
        #VGG add bn
        x.add(GlobalAveragePooling2D(name='avg_pool'))
        x.add(Dense(4096, name='fc1'))
        x.add(BatchNormalization())    
        x.add(Activation('relu'))
        x.add(Dropout(args.dropout1))
        x.add(Dense(2048, name='fc2'))
        x.add(BatchNormalization())    
        x.add(Activation('relu'))
        x.add(Dropout(args.dropout2))        
    elif (args.model==11):
        #VGG add max dropout  
        x.add(Dropout(args.dropout1))        
        x.add(GlobalAveragePooling2D(name='avg_pool'))
        x.add(Dense(4096, name='fc1'))
        x.add(BatchNormalization())    
        x.add(Activation('relu'))
        x.add(Dropout(args.dropout1))
        x.add(Dense(2048, name='fc2'))
        x.add(BatchNormalization())    
        x.add(Activation('relu'))
        x.add(Dropout(args.dropout2)) 
    elif (args.model == 12):
        x = base_model.output
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(nb_classes, activation='softmax', name='predictions')(x)
        model = Model(base_model.input, x)

    if(args.model != 12):
        x.add(Dense(nb_classes, activation='softmax', name='predictions'))
        model = x
        
    # Train top layer
    #if (args.testing or args.vis) :
    #    for layer in model.layers:
    #        layer.trainable = False
    #else:
    for layer in base_model.layers[0:args.tune_layer]:
        layer.trainable = False
    if args.tune_layer >= 0:
        for layer in base_model.layers[args.tune_layer:]:
            layer.trainable = False
    else:
        for layer in base_model.layers[args.tune_layer:]:
            print("layer",layer.name," set to:",True)
            layer.trainable = True
        
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
            loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    model.summary()
    summary_file= open(args.save_dir+"/model_summary.txt","w")
    model.summary(print_fn=lambda x: summary_file.write(x + '\n'))
    summary_file.close()
    return model

def train(model,args):
    # callbacks
    from keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint,LearningRateScheduler,EarlyStopping
    log = CSVLogger(args.save_dir + '/log.csv')
    tb = TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=int(args.debug))
    checkpoint = ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_acc', mode='max',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))
    early_stop = EarlyStopping(monitor='val_acc', patience=args.stopnum, verbose=1)

    if args.aug:
        nb_samples = args.aug_num 
    else:
        nb_samples = nb_train_samples

    # Fit model
    #history = model.fit(xtrain,ytrain, epochs=numepochs, class_weight=classweight, validation_data=(xtest,ytest), verbose=1,callbacks = [MetricsCheckpoint('logs')])
    history = model.fit_generator(generator=train_generator(args),
                        steps_per_epoch=int(nb_samples/ args.batch_size),
                        epochs=args.epochs,
                        use_multiprocessing=True,
                        validation_data=test_generator(args),
                        validation_steps=int(nb_validation_samples/args.batch_size),
                        callbacks=[log, tb, checkpoint, lr_decay, early_stop],
                        verbose=1)

    #model.save_weights(args.save_dir + '/trained_model.h5')
    
    # Evaluate model
    score = model.evaluate_generator(generator=test_generator(args), verbose=0)
    print('\nKeras CNN - accuracy:', score[1], '\n')
    with open(args.save_dir+"/model_summary.txt", "a") as summary_file:
        summary_file.write(str(model.metrics_names)+"\n")
        summary_file.write(str(score))

    return model

def get_all_y(data_generator):
    data_list = []
    batch_index = 0
    while batch_index <= data_generator.batch_index:
        data = data_generator.next()
        data_list.append(data[1])
        batch_index = batch_index + 1
    return data_list

    
def test(model,args):
    #y_test = get_all_y(test_generator(args))
    test_gen = test_generator(args)
    y_pred = model.predict_generator(test_gen,steps = len(test_gen.filenames),verbose=1)
    print(y_pred)
    #print(sklearn.metrics.classification_report(np.where(ytest > 0)[1], np.argmax(y_pred, axis=1), target_names=list(labels.values()))) 
    #Y_pred_classes = np.argmax(y_pred,axis = 1) 
    #Y_true = np.argmax(ytest,axis = 1) 
    #confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
    #print(confusion_mtx)
    #plot_confusion_matrix(confusion_mtx, classes = list(labels.values()))
    #plt.show()
    return model

def load_xray_test(args,load_count=-1):
    from xray_dataset import get_data
    x_test, y_test, imgfiles = get_data(test_dir(args),args.input_size,load_count)
    return (x_test, y_test,imgfiles)

def get_class_labels(args):
    from xray_dataset import get_labels_dict
    label_dict = get_labels_dict(train_dir(args))
    return list(label_dict.keys())

def test_one_by_one(model,args):
    #load balanced sample count per class
    (x_test,y_test,imgfiles) = load_xray_test(args, args.cpc)

    test_result_file = open(args.save_dir+"/"+os.path.basename(args.weights)+"_test_result.txt","w")
    model.summary(print_fn=lambda x: test_result_file.write(x + '\n'))

    Y_true = []
    Y_pred_classes = []
    for im,real_y,f in zip(x_test,y_test,imgfiles):
        y_pred = model.predict(im.reshape(-1,args.input_size,args.input_size,3).astype('float32') / 255.,verbose=0)[0]
        #print(y_pred)
        pred_class=np.argmax(y_pred)
        #print(pred_class)
        Y_pred_classes.append(pred_class)
        Y_true.append(real_y)
        #print(f,"pred result is",pred_class==real_y)
        #test_result_file.write(str(f)+" pred result is "+str(pred_class==real_y)+"\n")
    labels = get_class_labels(args)
    test_result_file.write(sklearn.metrics.classification_report(Y_true, Y_pred_classes , target_names=labels)+"\n")
    print(sklearn.metrics.classification_report(Y_true, Y_pred_classes , target_names=labels)) 
    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
    print(confusion_mtx)
    test_result_file.write(str(confusion_mtx)+"\n")
    
    #plot_confusion_matrix(confusion_mtx, classes = labels)
    #plt.show()
    test_result_file.close()
    return model

def show_activation(model,layer_idx):
    from vis.visualization import visualize_activation
    from vis.input_modifiers import Jitter
    # 1 is the imagenet category for 'PNEUMONIA'
    im = visualize_activation(model, layer_idx, filter_indices=None, max_iter=500,input_modifiers=[Jitter(16)], verbose=False)
    plt.imshow(im)
    plt.show()

def show_saliency(model,layer_idx,images,outs):
    from vis.visualization import visualize_saliency

    #plt.figure()
    f, ax = plt.subplots(nb_classes,args.cpc,figsize=(15,15))
    ax=ax.reshape((len(images)))
    plt.suptitle('Saliency for predicted classes')


    # New output containing the output result for the saliency visualization 
    gradsSaliency=[]
    certainties=[]
    classKeys=[]

    for i, img in enumerate(images): 
        classKey=np.argmax(outs[i])
        classKeys.append(classKey)
        certainty=outs[i][classKey]
        certainties.append(certainty)
        
        #grads = visualize_saliency(model, layer_idx, filter_indices=classKeys[i], seed_input=img, backprop_modifier='guided')        
        grads = visualize_saliency(model, layer_idx, filter_indices=None, seed_input=img, backprop_modifier='guided')        
        gradsSaliency.append(grads)
        
        ax[i].imshow(grads,cmap='jet')
        ax[i].set_title('pred:' + str(classKeys[i]) +'('+ str(round(certainties[i]*100,3))+' %)')
    plt.show()
    return gradsSaliency 

def show_cam(model,layer_idx,images,outs):
    import matplotlib.cm as cm
    # KERAS visualize_cam
    from vis.visualization import visualize_cam, overlay 

    #plt.figure()
    f, ax = plt.subplots(nb_classes,args.cpc,figsize=(15,15))
    ax=ax.reshape((len(images)))

    # New list containing the output image result of the Grad-Cam visualization. 
    gradsCAM=[]
    certainties=[]
    classKeys=[]
    plt.suptitle('grad-CAM for predicted classes') 


    for i, img in enumerate(images):    
        classKey=np.argmax(outs[i])
        classKeys.append(classKey)
        certainty=outs[i][classKey]
        certainties.append(certainty) 

        # Visualization with the Grad-Cam output. 
        #grads = visualize_cam(model, layer_idx, filter_indices=classKeys[i], seed_input=img, backprop_modifier='guided')        
        grads = visualize_cam(model, layer_idx, filter_indices=None, seed_input=img, backprop_modifier='guided')        
        # Lets overlay the heatmap onto original image. 
        gradsCAM.append(grads)
        t=plt.imshow(grads,cmap='jet')
        l=t.get_array()
        ax[i].imshow(overlay(l,img))
        ax[i].set_title('pred : ' + str(classKeys[i]) +'('+ str(round(certainties[i]*100,3))+' %)')
    plt.show()
    return gradsCAM

def show_salcam(gradsSaliency, gradsCAM,images,outs):
    from matplotlib import colors
    #plt.figure()
    f, ax = plt.subplots(nb_classes,args.cpc,figsize=(15,15))
    ax=ax.reshape((nb_classes*args.cpc))
    plt.suptitle('grad-CAM + saliency for predicted classes')

    certainties=[]
    classKeys=[]
    for i, img in enumerate(images):    
        classKey=np.argmax(outs[i])
        classKeys.append(classKey)
        certainty=outs[i][classKey]
        certainties.append(certainty) 

        ax[i].imshow((gradsSaliency[i][:,:,2]*1/(1.1+gradsCAM[i][:,:,2])),cmap='Blues',vmin=150),
        ax[i].set_title('pred : ' + str(classKeys[i]) +'('+ str(round(certainties[i]*100,3))+' %)')
    plt.show()

def vis(model,args):
    from vis.utils import utils
    from keras import activations
    
    # Utility to search for layer index by name. 
    # Alternatively we can specify this as -1 since it corresponds to the last layer.
    # Anyway, we are interested in the last layer, where the prediction happens 
    layer_idx = utils.find_layer_idx(model, 'predictions')

    #To visualize activation over final dense layer outputs, we need to switch the softmax activation out for linear
    #since gradient of output node will depend on all the other node activations.
    # Swap softmax with linear
    model.layers[layer_idx].activation = activations.linear
    model = utils.apply_modifications(model)

    layer_idx=args.layer_idx

    #We define the softmax function to translate the output of the CNN into a probability for each class. 
    def softmax(x):
        """
        Compute softmax values for each sets of scores in x.
        
        Rows are scores for each class. 
        Columns are predictions (samples).
        """
        scoreMatExp = np.exp(np.asarray(x))
        return scoreMatExp / scoreMatExp.sum(0)

    def predictImage(args):
        from os.path import basename
        load_count = args.cpc 
        (x_test,y_test,imgfiles) = load_xray_test(args,load_count)
        images=[]
        outs=[]

        if not args.noshowpredict: 
            #plt.figure()
            f, ax = plt.subplots(nb_classes, load_count,figsize=(15,15))
            ax=ax.reshape((nb_classes*load_count))
            plt.suptitle('predicted classes')

        i = 0
        for im,real_y,fn in zip(x_test,y_test,imgfiles):
            images.append(im)
            out=softmax(model.predict(im.reshape(-1,args.input_size,args.input_size,3).astype('float32') / 255.)[0])
            print(out)
            print(fn)
            outs.append(out)
            classKey=np.argmax(out)
            
            # Look in the dictionary for the specific term for the image identification. 
            certainty=out[classKey]
            
            # green to gray
            #from skimage.color import rgb2gray
            #im=rgb2gray(im)

            if not args.noshowpredict:
                if len(y_test)>1:
                    ax[i].imshow(im/255.)
                    ax[i].set_title(basename(fn)+" pred: " + str(classKey) + '(' + str(round(certainty*100,3)) + '%)')
                    i+=1
                else :
                    ax.imshow(im/255.)
                    ax.set_title(basename(fn)+" pred: " + str(classKey) + '(' + str(round(certainty*100,3)) + '%)')
        return images,outs

    images,outs = predictImage(args)
    if not args.noshowpredict:
        plt.show()
    if args.vis == "act" or args.vis == "all":
        show_activation(model,layer_idx)
    elif args.vis == "sal" or args.vis == "all":
        show_saliency(model,layer_idx,images,outs)
    elif args.vis == "cam" or args.vis == "all":
        show_cam(model,layer_idx,images,outs)
    elif args.vis == "salcam" or args.vis == "all":
        sal = show_saliency(model,layer_idx,images,outs)
        cam = show_cam(model,layer_idx,images,outs)
        show_salcam(sal,cam,images,outs)
        
    
    
def get_images_path(args):
    # Parse paths
    full_paths = [os.path.join(os.getcwd(), path) for path in args.path]
    files = set()
    for path in full_paths:
        if os.path.isfile(path):
            files.add(path)
        else:
            files |= set(glob.glob(path + '/*' + args.extension))
    return files

def get_class_number(args):
    from xray_dataset import get_labels_dict
    label_dict = get_labels_dict(train_dir(args))
    return len(label_dict)

if __name__ == "__main__":
    import os
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs 0.9 0.99")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('--data_dir', default='../chest_xray',
                        help="the base of data dir")    
    parser.add_argument('--input_size', default=224,
                        help="the size of input image, default value is 299")    
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    parser.add_argument('--pretrain_weights', default='imagenet',
                        help="The path of the pretrained weights. default is imagenet")
    parser.add_argument('--aug', action="store_true", help="if use data augmentation")
    parser.add_argument('--vis', help="generate visualization options are: act,sal,cam,all")
    #parser.add_argument('path', nargs='*', help='Path of a file or a folder of files.')
    parser.add_argument('--cpc', default=2, type=int, help="load how many image per class")
    parser.add_argument('--layer_idx', default=-1, type=int, help="the index of layer that will be vis")
    parser.add_argument('--noshowpredict', action="store_true", help="skip show predict")
    parser.add_argument('--aug_num', default=nb_train_samples, type=int,
                        help="the number of aug image, default value is nb_train_samples")     
    parser.add_argument('--stopnum', default=3, type=int,
                        help="the number of early stop, default value is 3")     
    parser.add_argument('--model', default=3, type=int,
                        help="the model, 1 - original, 2 - simple, 3 - complex ")     
    parser.add_argument('--aug_mode', default=1, type=int,
                        help="the model, 1 - simple, 2 - complex ")   
    parser.add_argument('--net', default="vgg16", 
                        help="the net, vgg16 or inceptionv3")   
    parser.add_argument('--tune_layer', default=0, type=int,
                        help="the tune layer in pretrained model, 0 is not fine tune the pretrained model, -1 means the last layer of pretrained model,... ")  
    parser.add_argument('--dropout1', default=0.5, type=float,
                        help="the dropout of first dense layer. ")  
    parser.add_argument('--dropout2', default=0.5, type=float,
                        help="the dropout of second dense layer. ")  
    parser.add_argument('--dropout3', default=0.5, type=float,
                        help="the dropout of third dense layer. ")  
    parser.add_argument('--batchnorm', action='store_true',
                        help="if add batch normal in model 6 after activition")
    parser.add_argument('--model7_fc1', action='store_true',
                        help="if model 7 include fc1 layer")


    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    else:
        if not (args.testing or args.vis) :
            print(args.save_dir+" existed!!!!! save the old result first")
            os.exit()

    if not (args.testing or args.vis):
        args_file = open(args.save_dir+"/args_file.txt","w")
        args_file.write(str(args)+"\n"); 
        args_file.close();

    if not os.path.exists(args.data_dir):
        print(args.data_dir+" is not exist")
        sys.exit()

    nb_classes = get_class_number(args)

    #'imagenet'
    if(args.net == "vgg16"):
        args.input_size = 224
        pretrained_model = VGG16(weights = args.pretrain_weights, include_top=False,input_shape=(args.input_size ,args.input_size ,3))
    else:
        args.input_size = 299
        pretrained_model = InceptionV3(weights = args.pretrain_weights, include_top=False,input_shape=(args.input_size ,args.input_size,3))
	
    model = createModel(pretrained_model,args)

    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)
        
    if args.testing:  # as long as weights are given, will run testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        test_one_by_one(model=model, args=args)
    elif args.vis :
        if args.weights is None:
            print('No weights are provided for vis.')
            sys.exit()
        #if args.path is None:
        #    print('No path are provided for vis.')
        #    sys.exit()

        vis(model=model, args=args)
    else:
        train(model=model,  args=args)
