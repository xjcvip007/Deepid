from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Convolution2D, MaxPooling2D, Flatten, merge, Input, BatchNormalization
from keras.datasets import mnist
from keras.optimizers import SGD, Adam, Adadelta
from keras.callbacks import EarlyStopping,ProgbarLogger,History,LearningRateScheduler
from keras.utils import np_utils
from keras.regularizers import WeightRegularizer,l1l2,l1,l2,activity_l1,activity_l2,activity_l1l2
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import os, pickle
import math

def load_data_split_pickle(dataset):
    def get_files(vec_folder):
        file_names = os.listdir(vec_folder)
        file_names.sort()
        if not vec_folder.endswith('/'):
            vec_folder += '/'
        for i in range(len(file_names)):
            file_names[i] = vec_folder + file_names[i]
        return file_names

    def load_data_xy(file_names):
        datas  = []
        labels = []
        for file_name in file_names:
            f = open(file_name, 'rb')
            x, y = pickle.load(f)
            datas.append(x)
            labels.append(y)
        combine_d = np.vstack(datas)
        combine_l = np.hstack(labels)
        return combine_d, combine_l

    valid_folder, train_folder = dataset
    valid_file_names = get_files(valid_folder)
    train_file_names = get_files(train_folder)
    valid_set = load_data_xy(valid_file_names)
    train_set = load_data_xy(train_file_names)
    # print valid_set
    train_set_x, train_set_y = train_set[0], train_set[1]
    valid_set_x, valid_set_y = valid_set[0], valid_set[1]
    return [(train_set_x, train_set_y), (valid_set_x, valid_set_y)]
    # print train_set_x.shape, train_set_y

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)
    return x

def save_and_print():
    with open('log_adam_256.txt','w') as f:
    	f.write(str(hist.history))
    plt.plot(len(hist.history['acc']),hist.history['acc'],label='acc')
    plt.plot(len(hist.history['val_acc']),hist.history['val_acc'],label='val_acc')
    plt.legend(loc='upper left',frameon=True)
    #savefig('/my_model_sgd_exp_256.png',dpi=72)
    plt.show()

def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 20.0
    lrate = initial_lrate * math.pow(drop,math.floor((1+epoch)/epochs_drop))
    return lrate

seed=7
np.random.seed(seed)
nb_classes=1595
nb_epoch=100
#learning_rate=0.1

dataset = load_data_split_pickle(('./test_set', './train_set'))
train_set_x, train_set_y = dataset[0]
# print train_set_y
train_set_y = np_utils.to_categorical(train_set_y,nb_classes)
train_set_x = train_set_x.reshape((train_set_x.shape[0], 3, 47, 55))
train_set_x = deprocess_image(train_set_x)
# print train_set_x.shape,train_set_y.shape
valid_set_x, valid_set_y = dataset[1]
valid_set_y = np_utils.to_categorical(valid_set_y, nb_classes)
valid_set_x = valid_set_x.reshape((valid_set_x.shape[0], 3, 47, 55))
valid_set_x = deprocess_image(valid_set_x)
# print valid_set_x.shape,valid_set_y.shape
# model = Sequential()
main_input = Input(shape=(3, 47, 55))
C1 = Convolution2D(20, 4, 4, border_mode='valid', activation='relu')(main_input)
#D1 = Dropout(0.1)(C1)
B1 = BatchNormalization(mode=1)(C1)
M1 = MaxPooling2D(pool_size=(2, 2), border_mode='valid')(B1)
C2 = Convolution2D(40, 3, 3, border_mode='valid', activation='relu')(M1)
#D2 = Dropout(0.1)(C2)
B2 = BatchNormalization(mode=1)(C2)
M2 = MaxPooling2D(pool_size=(2, 2), border_mode='valid')(B2)
C3 = Convolution2D(60, 3, 3, border_mode='valid', activation='relu')(M2)
#D3 = Dropout(0.1)(C3)
B3 = BatchNormalization(mode=1)(C3)
M3 = MaxPooling2D(pool_size=(2, 2), border_mode='valid')(B3)
M3_flatten = Flatten()(M3)
C4 = Convolution2D(80, 2, 2, border_mode='valid', activation='relu')(M3)
B4 = BatchNormalization(mode=1)(C4)
C4_flatten = Flatten()(B4)
Merge_layer = merge([M3_flatten, C4_flatten], mode='concat',concat_axis=1)
Deepid_layer = Dense(160)(Merge_layer) #feature extract layer's activation function should be null 
D5 = Dropout(0.5)(Deepid_layer)
B5 = BatchNormalization(mode=1)(D5)
Softmax_layer = Dense(1595, activation='softmax')(B5)
model = Model(input=main_input, output=Softmax_layer)
#sgd = SGD(lr=learning_rate, decay=learning_rate/nb_epoch, momentum=0.9, nesterov=True)
sgd = SGD(lr=0.0, decay=0.0, momentum=0.9, nesterov=True)
adam = Adam()
# adade = Adadelta()
# model.load_weights('my_model_weights_adam_1.h5')
# middle_output = K.function([main_input,K.learning_phase()],[C4_flatten])
# lay_output = middle_output([valid_set_x,1])[0]
# print lay_output.shape
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
earlystop = EarlyStopping(monitor = 'val_acc', patience = 20, mode='max')
lrate = LearningRateScheduler(step_decay)
hist = model.fit(train_set_x, train_set_y, batch_size=256, shuffle=True, nb_epoch=nb_epoch, validation_data=[valid_set_x,valid_set_y], callbacks=[earlystop])
model.save_weights('my_model_weights_adam_exp_256.h5')
save_and_print()
