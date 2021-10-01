# coding: utf-8
import tensorflow as tf
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras import backend as K


def r_square(y_true, y_pred):
    SSR = K.mean(K.square(y_pred - K.mean(y_true)), axis=-1)
    SST = K.mean(K.square(y_true - K.mean(y_true)), axis=-1)
    return SSR / SST


# fix random seed for reproducibility
np.random.seed(1)
tf.random.set_seed(1)

data_dir = "/home/lab210/test/data/"
log_dir = "/home/lab210/test/log/"
result_dir = "/home/lab210/test/result/"
prefix = 'test_inverse_model'

# Parameter configuration
num_node_ly1 = 2048
num_node_ly2 = 2048
num_node_ly3 = 1024
num_epochs = 5000
batch_size_const = 500

# dataset processing
x_trans = pd.read_csv(data_dir + 'trans.txt', header=None, encoding='utf-8')
y_para = pd.read_csv(data_dir + 'para.txt', header=None, encoding='utf-8')
X_old = x_trans.iloc[:, :].values
Y_old = y_para.iloc[:, :].values
X, Y = shuffle(X_old, Y_old)
X_train_old, X_test, Y_train_old, Y_test = train_test_split(X, Y, test_size=0.1, random_state=1)
X_train_old, Y_train_old = shuffle(X_train_old, Y_train_old)
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train_old, Y_train_old, test_size=0.1, random_state=1)

# model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=num_node_ly1, input_shape=(100,), activation='relu'))
model.add(tf.keras.layers.Dense(units=num_node_ly2, kernel_regularizer=tf.keras.regularizers.l2(0.00000008),
                                activation='relu'))
model.add(tf.keras.layers.Dense(units=num_node_ly3, kernel_regularizer=tf.keras.regularizers.l2(0.000000001),
                                activation='relu'))
model.add(tf.keras.layers.Dense(units=5, activation='tanh'))
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.MeanAbsoluteError(),
              metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError(), r_square])

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=200,
        verbose=1,
    ),
    tf.keras.callbacks.CSVLogger(
        filename=log_dir + prefix + "-train_early.csv",
        append=True
    )
]

history = model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), shuffle=True, epochs=num_epochs,
                    callbacks=callbacks,
                    batch_size=batch_size_const, verbose=2)

hist = history.history

# ### Evaluating the trained model on the test dataset
results = model.evaluate(X_test, Y_test, batch_size=batch_size_const, verbose=2)

# ### Saving and reloading the trained model
info_model = '_loss_{:.8f}_mse_{:.8f}_mae_{:.8f}'.format(*results)
name_model = result_dir + prefix + info_model + '.h5'

model.save(name_model,
           overwrite=True,
           include_optimizer=True,
           save_format='h5')








