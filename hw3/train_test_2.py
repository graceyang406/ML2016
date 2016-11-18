
# coding: utf-8

# In[1]:

import os
import sys

os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"

#function_name = 'train'
function_name = sys.argv[1]
#directory_path = 'data/'
directory_path = sys.argv[2]
#model1_file = 'trained_model1.h5'
model1_file = sys.argv[3] + '.h5'
#model2_file = 'trained_model2.h5'
model2_file = sys.argv[4] + '.h5'

if function_name == 'test':
    #prediction_file = 'prediction.csv'
    prediction_file = sys.argv[5]


# In[18]:

import pickle
import csv
import numpy as np

from keras.layers import Input, Dense, Lambda, BatchNormalization, Activation
from keras.models import Sequential, Model
from keras import backend as K
from keras import objectives
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping

batch_size = 100
original_dim = 3072
latent_dim = 2
intermediate_dim = 256
nb_epoch = 50
epsilon_std = 1.0

x = Input(batch_shape=(batch_size, original_dim))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                              std=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)


def vae_loss(x, x_decoded_mean):
    xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss + kl_loss

vae = Model(x, x_decoded_mean)
vae.compile(optimizer='rmsprop', loss=vae_loss)


# In[9]:

# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# build a generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)


# In[ ]:

model = Sequential()

model.add(Dense(4, init='normal', input_dim=2))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(8, init='normal', input_dim=2))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(10, init='normal'))
model.add(BatchNormalization())
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

modelcheckpoint = ModelCheckpoint(model1_file, monitor='val_loss', save_best_only=True, save_weights_only=False)
earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5)


# In[23]:

def read_training_data():
    all_label_x = np.asarray(pickle.load(open(directory_path + 'all_label.p', 'rb')))
    all_unlabel_x = np.asarray(pickle.load(open(directory_path + 'all_unlabel.p', 'rb')))
    all_label_x = np.reshape(all_label_x, (5000, 3072))
    all_unlabel_x = np.reshape(all_unlabel_x, (45000, 3072))
    return all_label_x.astype('float32') / 255., all_unlabel_x.astype('float32') / 255.

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
    vae.fit(train_x, train_x,
            shuffle=True,
            nb_epoch=nb_epoch,
            batch_size=batch_size,
            validation_data=(val_x, val_x))
    encoder.save_weights(model2_file)
    train_x = encoder.predict(train_x, batch_size=batch_size)
    val_x = encoder.predict(val_x, batch_size=batch_size)
    unlabel_x = encoder.predict(all_unlabel_x, batch_size=batch_size)
    model.fit(train_x, train_y, batch_size=100, nb_epoch=50, 
            validation_data=(val_x, val_y), callbacks=[modelcheckpoint, earlystopping])
    for it in range(5):
        model.load_weights(model1_file)
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
        model.fit(train_x, train_y, batch_size=100, nb_epoch=50, 
                validation_data=(val_x, val_y), callbacks=[modelcheckpoint, earlystopping])


# In[24]:

def read_testing_data():
    test_x = pickle.load(open(directory_path + 'test.p', 'rb'))
    test_x = np.asarray(test_x['data'])
    test_x = np.reshape(test_x, (10000, 3072))
    return test_x.astype('float32') / 255.
    
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
    model.load_weights(model1_file)
    encoder.load_weights(model2_file)
    test_x = encoder.predict(test_x, batch_size=batch_size)
    classes = model.predict_classes(test_x, batch_size=100)
    write_prediction_file(classes)


# In[25]:

globals()[function_name]()


# In[ ]:



