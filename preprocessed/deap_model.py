import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM, Dropout, Input, BatchNormalization
from keras.models import Model, Sequential
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import imgaug as ia
import imgaug.augmenters as iaa
from albumentations import (
    RandomRotate90,Flip
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
import cv2
from keras.preprocessing.image import ImageDataGenerator
"""
train_datagen = ImageDataGenerator(rescale=1./255,
    #shear_range=0.2,
    #zoom_range=0.2,
    #horizontal_flip=True,
    #rotation_range=90,
    validation_split=0.2) # set validation split

train_generator = train_datagen.flow_from_directory(
    "/Users/currentwire/Desktop/thesis/dataset/deap_heatmaps/",
    target_size=(224, 224),
    batch_size=50,
    color_mode="grayscale",
    class_mode='categorical',
    subset='training') # set as training data
validation_generator = train_datagen.flow_from_directory(
    "/Users/currentwire/Desktop/thesis/dataset/deap_heatmaps/", # same directory as training data
    target_size=(224, 224),
    batch_size=50,
    color_mode="grayscale",
    class_mode='categorical',
    subset='validation')
"""
def normalize(mat):
    return mat/255


def sequence():
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # horizontal flips
        iaa.Crop(percent=(0, 0.1)),  # random crops

        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
        )
    ], random_order=True)  # apply augmenters in random order

    return seq


# Flip and Random Rotate

def arguments():
    augs = [Flip(p=1), RandomRotate90(p=1)]
    return augs


# Augment an input image with n_affine transforms and n_flip's

def augment_image(image, n_affine=2, n_flip=0):
    seq = sequence()
    augs = arguments()

    extra_images = []

    for i in range(n_affine):
        result = seq(images=image)

        extra_images.append(result)

    for i in range(n_flip):

        for aug in augs:
            result_ = aug(image=image)['image']
            extra_images.append(result_)

    return np.array(extra_images)


def append_augmented_images(x_data, y_data):
    list_x = []
    list_y = []

    i = 0
    total_length = len(x_data)

    for image, y in zip(x_data, y_data):

        image = image.reshape((224, 224))
        extra_images = augment_image(image)

        extra_images = np.expand_dims(extra_images, axis=-1)

        list_x.append(extra_images)
        list_y.append(np.array([y for i in range(len(extra_images))]))

        i += 1

        # print('Creating Image Augmentation...')
        if i % 2000 == 0:
            print(i, '/', total_length)

    print('Done with adding images, now doing reshaping')

    x_images = np.array(list_x)
    y_images = np.array(list_y)
    x_images = x_images.reshape(-1, 224, 224, 1)
    y_images = y_images.reshape(-1, 5)

    print('Done.')

    return np.concatenate((x_data, x_images)), np.concatenate((y_data, y_images))
num_features=64
model_dude = Sequential([
    Conv2D(filters=64, kernel_size=3, activation='relu', input_shape=(224, 224, 1)),
    Dropout(0.2),
    BatchNormalization(),
    MaxPooling2D(pool_size=2),
    Conv2D(filters=32, kernel_size=3, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    MaxPooling2D(pool_size=2),
    Conv2D(filters=16, kernel_size=3, activation='relu'),
    BatchNormalization(),
    #Dropout(0.2),
    MaxPooling2D(pool_size=2),
    Flatten(),
    Dense(100, activation='relu'),
    #Dropout(0.2),
    Dense(5, activation='softmax')
])
#model_dude.load_weights("./all/thesis_fer_weights")
model_dude.compile(optimizer='adam', loss='categorical_crossentropy' , metrics=['accuracy'])
total_x=[]
total_y=[]
c=[0]*5
for i in os.listdir("/Users/currentwire/Documents/deap_heatmaps"):
    if not i.startswith("."):
        y=[0.0]*5
        if str(i)=="HAHV":
            y[0]=1.0
        elif str(i)=="HALV":
            y[1]=1.0
        elif str(i)=="LAHV":
            y[2]=1.0
        elif str(i)=="LALV":
            y[3]=1.0
        else:
            y[4]=1.0
        temp = 0
        for j in os.listdir("/Users/currentwire/Documents/deap_heatmaps/"+str(i)):
            if not j.startswith(".") and temp<1000 :
                image = cv2.imread("/Users/currentwire/Documents/deap_heatmaps/" + str(i) + "/" + str(j))
                gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
                image1 = cv2.resize(gray, (224, 224))
                image1=normalize(np.array(image1))
                x = np.expand_dims(image1, axis=-1)
                total_x.append(x)
                total_y.append(y)
                temp=temp+1
print(len(total_x),len(total_y))
x,y=append_augmented_images(np.array(total_x),np.array(total_y))
print(len(x),len(y))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
history = model_dude.fit(np.array(x_train), np.array(y_train), validation_data=(np.array(x_test),np.array(y_test)),batch_size=100,epochs = 50)
model_dude.save_weights("deap1again.hdf5")
hist_df = pd.DataFrame(history.history)
hist_csv_file = 'deap1.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)
predicted = model_dude.predict(np.array(x_train))
emotion_list = ['HAHV', 'HALV', 'LAHV', 'LALV', 'Neutral']
y_pred_classes = model_dude.predict_classes(np.array(x_train))
y_pred_classes = keras.utils.to_categorical(y_pred_classes,num_classes=5)
conf_matrix = confusion_matrix(np.argmax(np.array(y_train), axis=1), np.argmax(np.array(y_pred_classes),axis=1), normalize = 'true')
df_conf = pd.DataFrame(conf_matrix, index = [i for i in emotion_list],
                  columns = [i for i in emotion_list])

plt.figure(figsize = (10,10))
sns.heatmap(df_conf, annot=True, cmap = 'Blues', fmt = '.2', cbar = False)
plt.savefig("deap1.png")
