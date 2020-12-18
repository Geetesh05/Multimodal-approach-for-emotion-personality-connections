import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import shutil
from scipy import signal
import pandas as pd
import scipy.io
import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM, Dropout, Input, BatchNormalization
from keras.optimizers import Adam,SGD
from keras.models import Model, Sequential
from sklearn.model_selection import train_test_split
from keras.layers.core import Activation
import sys
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
def normalize(array):
    return array/255
def labels(a):
    global c
    limit=0
    result=[0.0]*7
    nums=1500
    for i in range(len(a)):
        if a[i]==1.0 and c[i]<nums and limit==0 and i==0:
            result[0]=1.0
            limit=1
            c[i]=c[i]+1
        if a[i]==1.0 and c[i]<nums and limit==0 and i==1:
            limit=1
            c[i]=c[i]+1
            result[4]=1.0
        if a[i]==1.0 and c[i]<nums and limit==0 and i==2:
            limit=1
            c[i]=c[i]+1
            result[1]=1.0
        if a[i]==1.0 and c[i]<nums and limit==0 and i==3:
            limit=1
            c[i]=c[i]+1
            result[6]=1.0
        if a[i]==1.0 and c[i]<nums and limit==0 and i==4:
            limit=1
            c[i]=c[i]+1
            result[3]=1.0
        if a[i]==1.0 and c[i]<nums and limit==0 and i==5:
            limit=1
            c[i]=c[i]+1
            result[5]=1.0
        if a[i]==1.0 and c[i]<nums and limit==0 and i==6:
            limit=1
            c[i]=c[i]+1
            result[2]=1.0
    return result,limit
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
    model_dude.load_weights("/Users/currentwire/PycharmProjects/thesis/improv3.hdf5")
    model_dude.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model_dude
x1=[]
y1=[]
model_classes=["neutral","happiness","sadness","angry","disgust","fear","other"]
for i in os.listdir("/Users/currentwire/Downloads/CREMA-D/processed/"):
    if not i.startswith("."):
        count=0
        print(str(i))
        for j in os.listdir("/Users/currentwire/Downloads/CREMA-D/processed/"+str(i)):
          if count<1800 and not j.startswith("."):
            image=cv2.imread("/Users/currentwire/Downloads/CREMA-D/processed/"+str(i)+"/"+str(j))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=3,
                minSize=(30, 30)
            )

            crops = []
            if len(faces)==1:
                if str(i) == "Anger":
                        y1.append([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
                if str(i) == "Happiness":
                        y1.append([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                if str(i) == "Disgust":
                        y1.append([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
                if str(i) == "Fear":
                        y1.append([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
                if str(i) == "Sadness":
                        y1.append([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
                if str(i) == "Neutral":
                        y1.append([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                for (x, y, w, h) in faces:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    face_crop = gray[y:y + h, x:x + w]
                    face_crop = cv2.resize(face_crop, (48, 48))
                    face_crop=normalize(np.array(face_crop))
                    face_crops = np.expand_dims(face_crop, axis=-1)
                    x1.append(np.array(face_crops))
                    count = count + 1

          else:
              break

x2=[]
y2=[]
c=[0]*7
for i in os.listdir("/Users/currentwire/Desktop/thesis/dataset/experiment/2sec"):
    if not i.startswith("."):
        print(str(i))
        mat = scipy.io.loadmat('/Users/currentwire/Desktop/thesis/dataset/experiment/e/Data_Preprocessed_'+str(i)+".mat")
        for j in os.listdir("/Users/currentwire/Desktop/thesis/dataset/experiment/2sec/"+str(i)):
          if not j.startswith("."):
            d = mat["VideoIDs"][0]
            e = int(np.where(d == j)[0])
            amigos = mat["labels_selfassessment"][0][e][0][5:]
            amigos_classes = ["neutral", "disgust", "happiness", "other", "angry", "fear", "sadness"]
            model_classes = ["neutral", "happiness", "sadness", "angry", "disgust", "fear", "other"]
            max_face=np.where(amigos==max(amigos))
            count1 = 0
            for k in os.listdir("/Users/currentwire/Desktop/thesis/dataset/experiment/2sec/"+str(i)+"/"+str(j)):
                    if not k.startswith(".") and count1<75:
                        image = cv2.imread("/Users/currentwire/Desktop/thesis/dataset/experiment/2sec/" + str(i) + "/" + str(j)+"/"+str(k))
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
                            for (x, y, w, h) in faces:
                                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                face_crop = gray[y:y + h, x:x + w]
                                face_crop = cv2.resize(face_crop, (48, 48))
                                face_crop1= normalize(np.array(face_crop))
                                face_crops = np.expand_dims(face_crop1, axis=-1)
                                a,b=labels(amigos)
                                if b==1:
                                    x2.append(np.array(face_crops))
                                    y2.append(a)

                                count1=count1+1


for i in range(len(x2)):
        x1.append(x2[i])
        y1.append(y2[i])

m1=model()
x_train, x_test, y_train, y_test = train_test_split(np.array(x1), np.array(y1), test_size=0.20, random_state=42)
hist=m1.fit(np.array(x_train),np.array(y_train),validation_data=(np.array(x_test),np.array(y_test)),batch_size=512,epochs=1)
m1.save("normal")
m1.save_weights("normal.hdf5")
hist_df = pd.DataFrame(hist.history)
hist_csv_file = 'history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)
predicted = m1.predict(np.array(x1))
emotion_list = ['Neutral', 'Happiness', 'Sadness', 'Anger', 'Disgust', 'Fear','other']
y_pred_classes = m1.predict_classes(np.array(x1))
final_y=[]
y_pred_classes = keras.utils.to_categorical(y_pred_classes)
conf_matrix = confusion_matrix(np.argmax(np.array(y1), axis=1), np.argmax(np.array(y_pred_classes),axis=1), normalize = 'true')
df_conf = pd.DataFrame(conf_matrix, index = [i for i in emotion_list],
                  columns = [i for i in emotion_list])

plt.figure(figsize = (10,10))
sns.heatmap(df_conf, annot=True, cmap = 'Blues', fmt = '.2', cbar = False)
plt.savefig("ht.jpg")
