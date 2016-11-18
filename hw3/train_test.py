
# coding: utf-8

# In[1]:

import os
import sys

os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"

#function_name = 'train'
function_name = sys.argv[1]
#directory_path = 'data/'
directory_path = sys.argv[2]
#model_file = 'model.h5'
model_file = sys.argv[3] + '.h5'

if function_name == 'test':
    #prediction_file = 'prediction.csv'
    prediction_file = sys.argv[4]


# In[23]:

from keras import backend as K
from keras.models import Sequential
from keras.layers import Convolution2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense, Dropout

K.set_image_dim_ordering('th')

model = Sequential()

model.add(Convolution2D(64, 3, 3, input_shape=(3, 32, 32), border_mode='same'))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(128, 3, 3, border_mode='same'))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))

model.add(Convolution2D(128, 3, 3, border_mode='same'))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(256, 3, 3, border_mode='same'))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))

model.add(Convolution2D(256, 3, 3, border_mode='same'))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, init='normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(10, init='normal'))
model.add(BatchNormalization())
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[24]:

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.125,
    height_shift_range=0.125,
    horizontal_flip=True,
    cval=0)

modelcheckpoint = ModelCheckpoint(model_file, monitor='val_loss', save_best_only=True, save_weights_only=False)
earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5)


# In[25]:

import pickle
import numpy as np
import csv
from keras.utils.np_utils import to_categorical

def read_training_data():
    print "Reading data..."
    all_label_x = np.asarray(pickle.load(open(directory_path + 'all_label.p', 'rb')))
    all_unlabel_x = np.asarray(pickle.load(open(directory_path + 'all_unlabel.p', 'rb')))
    all_label_x = np.reshape(all_label_x, (5000, 3, 32, 32))
    all_unlabel_x = np.reshape(all_unlabel_x, (45000, 3, 32, 32))
    return all_label_x, all_unlabel_x

def get_training_and_validation_data(all_label_x):
    train_x = all_label_x
    train_y = np.zeros(shape=(5000,10))
    for i in range(5000):
        train_y[i, int(i / 500)] = 1
    p = np.random.permutation(len(train_x))
    train_x = train_x[p]
    train_y = train_y[p]
    return train_x[:4500], train_y[:4500], train_x[4500:], train_y[4500:]
  
def train():
    all_label_x, all_unlabel_x = read_training_data()
    train_x, train_y, val_x, val_y = get_training_and_validation_data(all_label_x)
    unlabel_x = all_unlabel_x
    datagen.fit(train_x)
    model.fit_generator(datagen.flow(train_x, train_y, batch_size=100), samples_per_epoch=len(train_x)*5, nb_epoch=50, 
                        validation_data=(val_x, val_y), callbacks=[modelcheckpoint, earlystopping])
    while len(unlabel_x) > 0:
        model.load_weights(model_file)
        unlabel_proba = model.predict_proba(unlabel_x, batch_size=100)
        new_label_dict = {}
        for i in range(10):
            new_label_dict[i] = []
        #class_num_limit = 0
        for index, label in enumerate(unlabel_proba):
            proba_max = label[0]
            for cla, proba in enumerate(label):
                if proba > proba_max:
                    proba_max = proba
            if proba_max > 0.999:
                new_label_dict[cla].append(index)
                #class_num_limit += 1
        #class_num_limit = int(class_num_limit / 10)
        #for cla in range(10):
        #    while len(new_label_dict[cla]) > class_num_limit:
        #        new_label_dict[cla].pop()
        new_label_indexes = []
        for cla in range(10):
            new_label_indexes += new_label_dict[cla]
        index_dict = dict(zip(sorted(new_label_indexes), [i for i in range(len(new_label_indexes))]))
        new_label_y = np.zeros((len(new_label_indexes), 10))
        for cla in range(10):
            for index in new_label_dict[cla]:
                new_label_y[index_dict[index], cla] = 1
        new_label_x = unlabel_x[new_label_indexes]
        #mask = np.ones(len(unlabel_x), np.bool)
        #mask[new_label_indexes] = 0
        #unlabel_x = unlabel_x[mask]
        #new_label_y = to_categorical(new_label_classes, nb_classes=10)
        print "Add", len(new_label_indexes), "data into", len(train_x), "data"
        train_x = np.concatenate((train_x, new_label_x))
        train_y = np.concatenate((train_y, new_label_y))
        p = np.random.permutation(len(train_x))
        train_x = train_x[p]
        train_y = train_y[p]
        print "Training", len(train_x), "data..."
        model.fit_generator(datagen.flow(train_x, train_y, batch_size=100), samples_per_epoch=len(train_x)*5, nb_epoch=25, 
                            validation_data=(val_x, val_y), callbacks=[modelcheckpoint, earlystopping])


# In[26]:

def read_testing_data():
    print "Reading data..."
    test_x = pickle.load(open(directory_path + 'test.p', 'rb'))
    test_x = np.asarray(test_x['data'])
    test_x = np.reshape(test_x, (10000, 3, 32, 32))
    return test_x
    
def write_prediction_file(classes):
    test_classes = [['ID', 'class']]
    for i, n in enumerate(classes):
        l = [str(i), str(n)]
        test_classes.append(l)
        
    f = open(prediction_file, 'w')
    w = csv.writer(f)
    w.writerows(test_classes)
    f.close()
    
def test():
    test_x = read_testing_data()
    model.load_weights(model_file)
    classes = model.predict_classes(test_x, batch_size=100)
    write_prediction_file(classes)


# In[ ]:

globals()[function_name]()


# In[ ]:



