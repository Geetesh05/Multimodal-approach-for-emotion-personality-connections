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
from sklearn.model_selection import train_test_split
import cv2
# loading fer2013
# df = pd.read_csv('fer2013.csv')
import sys
emotion_list = ['neutral', 'happiness', 'sadness', 'anger', 'disgust', 'fear', 'others']
os.chdir("/home/xx652230/")
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
    x = normalize(x)
    # add dimension, keras requirement
    x = np.expand_dims(x, axis=-1)

    return x, y

def model():
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
    return model_dude
df = pd.read_csv('./fer2013.csv')
new= pd.read_csv('./fer2013new.csv')
df_training = df.loc[df['Usage'] == 'Training']
new_training = new.loc[new['Usage'] == 'Training']
df_test = df.loc[df['Usage'] != 'Training']
new_test = new.loc[new['Usage'] != 'Training']
x_train1, y_train1 = make_xy(df_training, new_training)
x_test1, y_test1 = make_xy(df_test, new_test)
x1 = []
y1 = []
model_classes = ["neutral", "happiness", "sadness", "angry", "disgust", "fear", "other"]
for i in os.listdir("/home/xx652230/Downloads/crema"):
    if not i.startswith("."):
        count = 0
        print(str(i))
        for j in os.listdir("/home/xx652230/Downloads/crema/" + str(i)):
            if count < 5000 and not j.startswith("."):
                image = cv2.imread("/home/xx652230/Downloads/crema/" + str(i) + "/" + str(j))
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.3,
                    minNeighbors=3,
                    minSize=(30, 30)
                )

                crops = []
                if len(faces) == 1:
                    yy = []
                    if str(i) == "Anger":
                        yy = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
                    elif str(i) == "Happiness":
                        yy = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    elif str(i) == "Disgust":
                        yy = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
                    elif str(i) == "Fear":
                        yy = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
                    elif str(i) == "Sadness":
                        yy = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
                    elif str(i) == "Neutral":
                        yy = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    for (x, y, w, h) in faces:
                        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        face_crop = gray[y:y + h, x:x + w]
                        face_crop = cv2.resize(face_crop, (48, 48))
                        face_crop = normalize(np.array(face_crop))
                        face_crops = np.expand_dims(face_crop, axis=-1)
                        x1.append(np.array(face_crops))
                        y1.append(np.array(yy))
                        count = count + 1

x_train2, x_test2, y_train2, y_test2 = train_test_split(np.array(x1), np.array(y1), test_size=0.20, random_state=42)

xtrain=np.concatenate([x_train1,x_train2])
xtest=np.concatenate([x_test1,x_test2])
ytrain=np.concatenate([y_train1,y_train2])
ytest=np.concatenate([y_test1,y_test2])
print("total samples crema for train and test",len(x_train2),len(x_test2))
print("total samples combined for train and test",len(xtrain),len(xtest))
m1=model()
hist=m1.fit(np.array(xtrain),np.array(ytrain),validation_data=(np.array(xtest),np.array(ytest)),batch_size= 500,epochs=100)
m1.save("combined")
m1.save_weights("combined.hdf5")

hist_df = pd.DataFrame(hist.history)
hist_csv_file = 'combined.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)