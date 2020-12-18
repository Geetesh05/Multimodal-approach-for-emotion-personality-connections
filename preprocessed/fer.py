import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM, Dropout, Input, BatchNormalization
from keras.models import Model, Sequential
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import imgaug as ia
import imgaug.augmenters as iaa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import tensorflow as tf
# loading fer2013
# df = pd.read_csv('fer2013.csv')
import sys
emotion_list = ['neutral', 'happiness', 'sadness', 'anger', 'disgust', 'fear', 'others']

def best_emotion_matching(match):
    match = match
    size = len(match)
    emotion_unknown = [0.0] * (size)
    emotion_unknown[-2] = 1.0
    # remove emotions with a single vote (outlier removal)
    for i in range(size):
        if match[i] < 1.0 + sys.float_info.epsilon:
            match[i] = 0.0

    sum_list = sum(match)
    emotion = [0.0] * (size)
    # find the peak value of the emo_raw list
    maxval = max(match)
    if maxval > 0.5 * sum_list:
        emotion[np.argmax(np.array(match))] = maxval

    else:
        emotion = emotion_unknown
    final = [0.0] * 7
    for i in range(len(emotion)):
        if float(emotion[i]) / sum(emotion) == 1.0:
            if i == 2 or i == 7 or i == 8 or i == 9:
                final[6] = 1.0
            elif i > 2 and i < 7:
                final[i - 1] = 1.0
            else:
                final[i] = 1.0
    check = [float(i) / sum(emotion) for i in emotion]
    # print(final)
    return final


def string_to_array(string):
    pixels = string.split(' ')
    # if len(pixels) != 2304:
    #  return False
    pixels = [int(x) for x in pixels]
    pixels = np.array(pixels)
    pixels = pixels.reshape((48, 48))
    return pixels


def normalize(array):
    return array / 255


def make_xy(df, new):
    x = []
    y = []

    for row, row1 in zip(df.iterrows(), new.iterrows()):
        # print(row1[1][2:])
        # emotion = row[1]['emotion']
        emotions = best_emotion_matching(row1[1][2:])
        array = string_to_array(row[1]['pixels'])
        # if isinstance(array, bool): # bypass for wrong shaped images
        #  continue
        x.append(array)
        y.append(emotions)

    x = np.array(x)
    y = np.array(y)
    # one hot encoding y
    # y = keras.utils.np_utils.to_categorical(y)
    #x = normalize(x)
    # add dimension, keras requirement
    x = np.expand_dims(x, axis=-1)

    return x, y

def model(x1,y1,x2,y2):
    num_features = 64
    width, height = 48, 48
    model_dude = Sequential([
        Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1)),  # 1
        Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1)),  # 1
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),  # 2
        Dropout(0.5),

        Conv2D(2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'),  # 3
        BatchNormalization(),
        Conv2D(2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'),  # 4
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),  # 5
        Dropout(0.5),

        Conv2D(2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'),  # 3
        BatchNormalization(),
        Conv2D(2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'),  # 4
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),  # 5
        Dropout(0.5),

        Conv2D(2 * 2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'),  # 6
        BatchNormalization(),
        Conv2D(2 * 2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'),  # 7
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),  # 8
        Dropout(0.5),

        Flatten(),

        Dense(2 * 2 * 2 * num_features, activation='relu'),  # 9
        Dropout(0.4),
        Dense(2 * 2 * num_features, activation='relu'),  # 10
        Dropout(0.4),
        Dense(2 * num_features, activation='relu'),  # 10
        Dropout(0.5),

        Dense(7, activation='softmax')

    ])

    model_dude.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model_dude.fit(x1, y1, validation_data=(x2, y2), batch_size=256, epochs=100)
    model_dude.save_weights("thesis_fer_weights")

# pixels = string_to_array(df.iloc[0,1])
if __name__ == '__main__':
    df = pd.read_csv('/Users/currentwire/Desktop/fer2013.csv')
    new= pd.read_csv('/Users/currentwire/Desktop/fer2013new.csv')
    df_training = df.loc[df['Usage'] == 'Training']
    new_training = new.loc[new['Usage'] == 'Training']
    df_test = df.loc[df['Usage'] != 'Training']
    new_test = new.loc[new['Usage'] != 'Training']
    x_train, y_train = make_xy(df_training, new_training)
    x_test, y_test = make_xy(df_test, new_test)
    #model(x_train,y_train,x_test,y_test)
    model = keras.models.load_model("/Users/currentwire/Documents/GitHub/CREMA-D/model/model_crema_10k")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.evaluate(np.array(x_train), np.array(y_train)))