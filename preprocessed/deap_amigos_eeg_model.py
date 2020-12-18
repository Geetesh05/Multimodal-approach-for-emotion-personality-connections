import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM, Dropout, Input, BatchNormalization
from keras.models import Model, Sequential
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import cv2
import tensorflow as tf
import imgaug as ia
import imgaug.augmenters as iaa
from albumentations import (
    RandomRotate90, Flip
)

print('This should list a GPU: ', tf.test.gpu_device_name())
os.chdir('/home/xx652230')


def normalize(array):
    return array / 255


num_features = 64
model_dude = Sequential([
    Conv2D(filters=64, kernel_size=5, activation='relu', input_shape=(120, 120, 1)),
    Dropout(0.2),
    BatchNormalization(),
    MaxPooling2D(pool_size=2),
    Conv2D(filters=32, kernel_size=3, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    MaxPooling2D(pool_size=2),
    Conv2D(filters=16, kernel_size=5, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    MaxPooling2D(pool_size=2),
    Flatten(),
    Dense(100, activation='relu'),
    Dropout(0.2),
    Dense(5, activation='softmax')
])
# model_dude.load_weights("./all/thesis_fer_weights")
model_dude.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

q1 = []
q2 = []
y1 = []
y2 = []
c = [0] * 5
for i in os.listdir("./Downloads/heatmaps"):
    if not i.startswith("."):
        y = [0.0] * 5
        f = 0
        if str(i) == "HAHV":
            y[0] = 1.0
            temp = 0
            for j in os.listdir("./Downloads/heatmaps/" + str(i)):
                if not j.startswith("."):
                    image = cv2.imread("./Downloads/heatmaps/" + str(i) + "/" + str(j))
                    gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
                    image1 = cv2.resize(gray, (120, 120))
                    image1 = normalize(np.array(image1))
                    x = np.expand_dims(image1, axis=-1)
                    q1.append(x)
                    y1.append(y)
                    temp = temp + 1
        elif str(i) == "HALV":
            y[1] = 1.0
            temp = 0
            for j in os.listdir("./Downloads/heatmaps/" + str(i)):
                if not j.startswith("."):
                    image = cv2.imread("./Downloads/heatmaps/" + str(i) + "/" + str(j))
                    gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
                    image1 = cv2.resize(gray, (120, 120))
                    image1 = normalize(np.array(image1))
                    x = np.expand_dims(image1, axis=-1)
                    q1.append(x)
                    y1.append(y)
                    temp = temp + 1
        elif str(i) == "LAHV":
            y[2] = 1.0
            temp = 0
            for j in os.listdir("./Downloads/heatmaps/" + str(i)):
                if not j.startswith(".") and temp < 615:
                    image = cv2.imread("./Downloads/heatmaps/" + str(i) + "/" + str(j))
                    gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
                    image1 = cv2.resize(gray, (120, 120))
                    image1 = normalize(np.array(image1))
                    x = np.expand_dims(image1, axis=-1)
                    q1.append(x)
                    y1.append(y)
                    temp = temp + 1
        elif str(i) == "LALV":
            y[3] = 1.0
            temp = 0
            for j in os.listdir("./Downloads/heatmaps/" + str(i)):
                if not j.startswith("."):
                    image = cv2.imread("./Downloads/heatmaps/" + str(i) + "/" + str(j))
                    gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
                    image1 = cv2.resize(gray, (120, 120))
                    image1 = normalize(np.array(image1))
                    x = np.expand_dims(image1, axis=-1)
                    q1.append(x)
                    y1.append(y)
                    temp = temp + 1
        else:
            y[4] = 1.0
            temp = 0
            for j in os.listdir("./Downloads/heatmaps/" + str(i)):
                if not j.startswith(".") and temp < 500:
                    image = cv2.imread("./Downloads/heatmaps/" + str(i) + "/" + str(j))
                    gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
                    image1 = cv2.resize(gray, (120, 120))
                    image1 = normalize(np.array(image1))
                    x = np.expand_dims(image1, axis=-1)
                    q2.append(x)
                    y2.append(y)
                    temp = temp + 1

for i in os.listdir("./Downloads/amigos/maps/"):
    if not i.startswith("."):
        y = [0.0] * 5
        if str(i) == "HAHV":
            y[0] = 1.0
            for j in os.listdir("./Downloads/amigos/maps/" + str(i)):
                if not j.startswith("."):
                    image = cv2.imread("./Downloads/amigos/maps/" + str(i) + "/" + str(j))
                    gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
                    image1 = cv2.resize(gray, (120, 120))
                    image1 = normalize(np.array(image1))
                    x = np.expand_dims(image1, axis=-1)
                    q1.append(x)
                    y1.append(y)
                    temp = temp + 1
        elif str(i) == "HALV":
            y[1] = 1.0
            for j in os.listdir("./Downloads/amigos/maps/" + str(i)):
                if not j.startswith("."):
                    image = cv2.imread("./Downloads/amigos/maps/" + str(i) + "/" + str(j))
                    gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
                    image1 = cv2.resize(gray, (120, 120))
                    image1 = normalize(np.array(image1))
                    x = np.expand_dims(image1, axis=-1)
                    q1.append(x)
                    y1.append(y)
                    temp = temp + 1
        elif str(i) == "LAHV":
            y[2] = 1.0
            for j in os.listdir("./Downloads/amigos/maps/" + str(i)):
                if not j.startswith("."):
                    image = cv2.imread("./Downloads/amigos/maps/" + str(i) + "/" + str(j))
                    gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
                    image1 = cv2.resize(gray, (120, 120))
                    image1 = normalize(np.array(image1))
                    x = np.expand_dims(image1, axis=-1)
                    q1.append(x)
                    y1.append(y)
                    temp = temp + 1
        elif str(i) == "LALV":
            y[3] = 1.0
            for j in os.listdir("./Downloads/amigos/maps/" + str(i)):
                if not j.startswith("."):
                    image = cv2.imread("./Downloads/amigos/maps/" + str(i) + "/" + str(j))
                    gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
                    image1 = cv2.resize(gray, (120, 120))
                    image1 = normalize(np.array(image1))
                    x = np.expand_dims(image1, axis=-1)
                    q1.append(x)
                    y1.append(y)
                    temp = temp + 1
        else:
            y[4] = 1.0
            for j in os.listdir("./Downloads/amigos/maps/" + str(i)):
                if not j.startswith(".") and temp < 500:
                    image = cv2.imread("./Downloads/amigos/maps/" + str(i) + "/" + str(j))
                    gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
                    image1 = cv2.resize(gray, (120, 120))
                    image1 = normalize(np.array(image1))
                    x = np.expand_dims(image1, axis=-1)
                    q2.append(x)
                    y2.append(y)
                    temp = temp + 1


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

def augment_image(image, n_affine=4, n_flip=2):
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

        image = image.reshape((120, 120))
        extra_images = augment_image(image)

        extra_images = np.expand_dims(extra_images, axis=-1)

        list_x.append(extra_images)
        list_y.append(np.array([y for i in range(len(extra_images))]))

        i += 1

        # print('Creating Image Augmentation...')
        if i % 1000 == 0:
            print(i, '/', total_length)

    print('Done with adding images, now doing reshaping')

    x_images = np.array(list_x)
    y_images = np.array(list_y)
    x_images = x_images.reshape(-1, 120, 120, 1)
    y_images = y_images.reshape(-1, 5)

    print('Done.')

    return np.concatenate((x_data, x_images)), np.concatenate((y_data, y_images))


# total_x1= np.expand_dims(total_x, axis=-1)
# x=normalize(total_x1)
print(len(q1), len(q2))
x11, y11 = append_augmented_images(np.array(q1), np.array(y1))
print(len(x11))
for i in range(len(x11)):
    q2.append(x11[i])
    y2.append(y11[i])
x_train, x_test, y_train, y_test = train_test_split(q2, y2, test_size=0.20, random_state=42)
history = model_dude.fit(np.array(x_train), np.array(y_train), validation_data=(np.array(x_test), np.array(y_test)),
                         batch_size=32, epochs=100)
model_dude.save("new2")
model_dude.save_weights("new2.hdf5")
hist_df = pd.DataFrame(history.history)
hist_csv_file = 'new2.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

predicted = model_dude.predict(np.array(x_test))
emotion_list = ['HAHV', 'HALV', 'LAHV', 'LALV', 'Neutral']
y_pred_classes = model_dude.predict_classes(np.array(x_test))
final_y = []
y_pred_classes = keras.utils.to_categorical(y_pred_classes)
conf_matrix = confusion_matrix(np.argmax(np.array(y_test), axis=1), np.argmax(np.array(y_pred_classes), axis=1),
                               normalize='true')
df_conf = pd.DataFrame(conf_matrix, index=[i for i in emotion_list],
                       columns=[i for i in emotion_list])

plt.figure(figsize=(10, 10))
sns.heatmap(df_conf, annot=True, cmap='Blues', fmt='.2', cbar=False)
plt.savefig("new2.jpg")