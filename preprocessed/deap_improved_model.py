#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM, Dropout, Input, BatchNormalization
from keras.models import Model, Sequential
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from keras.optimizers import Adam,SGD
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
    RandomRotate90,Flip
)
import scipy.io

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
print('This should list a GPU: ',tf.test.gpu_device_name())
os.chdir('/home/xx652230')


# In[2]:


num_features=64
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
#model_dude.load_weights("./all/thesis_fer_weights")
model_dude.compile(optimizer='adam', loss='categorical_crossentropy' , metrics=['accuracy'])


# In[3]:


def normalize(mat):
    return mat/255


# In[11]:


x_1=[]
x_neu=[]
y_neu=[]
y_1=[]
count=0
for i in os.listdir("./Downloads/amigos/eeg_frames/"):
  if not i.startswith(".") and count<3:  
    mat=scipy.io.loadmat('./Downloads/amigos/eeg/Data_Preprocessed_'+str(i)+".mat")
    for j in os.listdir("./Downloads/amigos/eeg_frames/"+str(i)):
      if not j.startswith("."):  
        d = mat["VideoIDs"][0]
        e=int(np.where(d == j)[0])
        amigos = mat["labels_selfassessment"][0][e][0]
        if amigos[0]>7 and amigos[1]>7:
            for k in range(len(os.listdir("./Downloads/amigos/eeg_frames/"+str(i)+"/"+str(j)))):
                for l in os.listdir("./Downloads/amigos/eeg_frames/"+str(i)+"/"+str(j)+"/"+str(k)):
                    image=cv2.imread("./Downloads/amigos/eeg_frames/"+str(i)+"/"+str(j)+"/"+str(k)+"/"+str(l))
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    gray=cv2.resize(gray,(120,120))
                    gray=normalize(np.array(gray))
                    xt = np.expand_dims(gray, axis=-1)
                    x_1.append(xt)
                    y_1.append([1,0,0,0,0])
        elif amigos[0]>7 and amigos[1]<3:
            for k in range(len(os.listdir("./Downloads/amigos/eeg_frames/"+str(i)+"/"+str(j)))):
                for l in os.listdir("./Downloads/amigos/eeg_frames/"+str(i)+"/"+str(j)+"/"+str(k)):
                    image=cv2.imread("./Downloads/amigos/eeg_frames/"+str(i)+"/"+str(j)+"/"+str(k)+"/"+str(l))
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    gray=cv2.resize(gray,(120,120))
                    gray=normalize(np.array(gray))
                    xt = np.expand_dims(gray, axis=-1)
                    x_1.append(xt)
                    y_1.append([0,1,0,0,0])
        elif amigos[0]<3 and amigos[1]>7:
            for k in range(len(os.listdir("./Downloads/amigos/eeg_frames/"+str(i)+"/"+str(j)))):
                for l in os.listdir("./Downloads/amigos/eeg_frames/"+str(i)+"/"+str(j)+"/"+str(k)):
                    image=cv2.imread("./Downloads/amigos/eeg_frames/"+str(i)+"/"+str(j)+"/"+str(k)+"/"+str(l))
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    gray=cv2.resize(gray,(120,120))
                    gray=normalize(np.array(gray))
                    xt = np.expand_dims(gray, axis=-1)
                    x_1.append(xt)
                    y_1.append([0,0,1,0,0])
        elif amigos[0]<3 and amigos[1]<3:
            for k in range(len(os.listdir("./Downloads/amigos/eeg_frames/"+str(i)+"/"+str(j)))):
                for l in os.listdir("./Downloads/amigos/eeg_frames/"+str(i)+"/"+str(j)+"/"+str(k)):
                    image=cv2.imread("./Downloads/amigos/eeg_frames/"+str(i)+"/"+str(j)+"/"+str(k)+"/"+str(l))
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    gray=cv2.resize(gray,(120,120))
                    gray=normalize(np.array(gray))
                    xt = np.expand_dims(gray, axis=-1)
                    x_1.append(xt)
                    y_1.append([0,0,0,1,0])
        else:
            for k in range(len(os.listdir("./Downloads/amigos/eeg_frames/"+str(i)+"/"+str(j)))):
                for l in os.listdir("./Downloads/amigos/eeg_frames/"+str(i)+"/"+str(j)+"/"+str(k)):
                    image=cv2.imread("./Downloads/amigos/eeg_frames/"+str(i)+"/"+str(j)+"/"+str(k)+"/"+str(l))
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    gray=cv2.resize(gray,(120,120))
                    gray=normalize(np.array(gray))
                    xt = np.expand_dims(gray, axis=-1)
                    x_neu.append(xt)
                    y_neu.append([0,0,0,0,1])
    count=count+1


                


# In[12]:


q1=[]
y1=[]
q2=[]
y2=[]

for i in os.listdir("./Downloads/deap/heatmaps"):
    if not i.startswith("."):
        y=[0.0]*5
        f=0
        if str(i)=="HAHV":
            y[0]=1.0
            temp=0
            for j in os.listdir("./Downloads/deap/heatmaps/"+str(i)):
                if not j.startswith("."):
                    image = cv2.imread("./Downloads/deap/heatmaps/" + str(i) + "/" + str(j))
                    gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
                    image1 = cv2.resize(gray, (120, 120))
                    image1=normalize(np.array(image1))
                    x = np.expand_dims(image1, axis=-1)
                    q1.append(x)
                    y1.append(y)
                    temp=temp+1
        elif str(i)=="HALV":
            y[1]=1.0
            temp=0
            for j in os.listdir("./Downloads/deap/heatmaps/"+str(i)):
                if not j.startswith("."):
                    image = cv2.imread("./Downloads/deap/heatmaps/" + str(i) + "/" + str(j))
                    gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
                    image1 = cv2.resize(gray, (120, 120))
                    image1=normalize(np.array(image1))
                    x = np.expand_dims(image1, axis=-1)
                    q1.append(x)
                    y1.append(y)
                    temp=temp+1
        elif str(i)=="LAHV":
            y[2]=1.0
            temp=0
            for j in os.listdir("./Downloads/deap/heatmaps/"+str(i)):
                if not j.startswith("."):
                    image = cv2.imread("./Downloads/deap/heatmaps/" + str(i) + "/" + str(j))
                    gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
                    image1 = cv2.resize(gray, (120, 120))
                    image1=normalize(np.array(image1))
                    x = np.expand_dims(image1, axis=-1)
                    q1.append(x)
                    y1.append(y)
                    temp=temp+1
        elif str(i)=="LALV":
            y[3]=1.0
            temp=0
            for j in os.listdir("./Downloads/deap/heatmaps/"+str(i)):
                if not j.startswith("."):
                    image = cv2.imread("./Downloads/deap/heatmaps/" + str(i) + "/" + str(j))
                    gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
                    image1 = cv2.resize(gray, (120, 120))
                    image1=normalize(np.array(image1))
                    x = np.expand_dims(image1, axis=-1)
                    q1.append(x)
                    y1.append(y)
                    temp=temp+1
        else:
            y[4]=1.0
            temp=0
            for j in os.listdir("./Downloads/deap/heatmaps/"+str(i)):
                if not j.startswith(".") and temp<500:
                    image = cv2.imread("./Downloads/deap/heatmaps/" + str(i) + "/" + str(j))
                    gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
                    image1 = cv2.resize(gray, (120, 120))
                    image1=normalize(np.array(image1))
                    x = np.expand_dims(image1, axis=-1)
                    q2.append(x)
                    y2.append(y)
                    temp=temp+1
     

for i in os.listdir("./Downloads/amigos/maps/"):
    if not i.startswith("."):
        y=[0.0]*5
        if str(i)=="HAHV":
            y[0]=1.0
            for j in os.listdir("./Downloads/amigos/maps/"+str(i)):
                if not j.startswith("."):
                    image = cv2.imread("./Downloads/amigos/maps/"+str(i) + "/" + str(j))
                    gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
                    image1 = cv2.resize(gray, (120, 120))
                    image1=normalize(np.array(image1))
                    x = np.expand_dims(image1, axis=-1)
                    q1.append(x)
                    y1.append(y)
                    temp=temp+1
        elif str(i)=="HALV":
            y[1]=1.0
            for j in os.listdir("./Downloads/amigos/maps/"+str(i)):
                if not j.startswith("."):
                    image = cv2.imread("./Downloads/amigos/maps/"+str(i) + "/" + str(j))
                    gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
                    image1 = cv2.resize(gray, (120, 120))
                    image1=normalize(np.array(image1))
                    x = np.expand_dims(image1, axis=-1)
                    q1.append(x)
                    y1.append(y)
                    temp=temp+1
        elif str(i)=="LAHV":
            y[2]=1.0
            for j in os.listdir("./Downloads/amigos/maps/"+str(i)):
                if not j.startswith("."):
                    image = cv2.imread("./Downloads/amigos/maps/"+str(i) + "/" + str(j))
                    gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
                    image1 = cv2.resize(gray, (120, 120))
                    image1=normalize(np.array(image1))
                    x = np.expand_dims(image1, axis=-1)
                    q1.append(x)
                    y1.append(y)
                    temp=temp+1
        elif str(i)=="LALV":
            y[3]=1.0
            for j in os.listdir("./Downloads/amigos/maps/"+str(i)):
                if not j.startswith("."):
                    image = cv2.imread("./Downloads/amigos/maps/"+str(i) + "/" + str(j))
                    gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
                    image1 = cv2.resize(gray, (120, 120))
                    image1=normalize(np.array(image1))
                    x = np.expand_dims(image1, axis=-1)
                    q1.append(x)
                    y1.append(y)
                    temp=temp+1
        else:
            y[4]=1.0
            for j in os.listdir("./Downloads/amigos/maps/"+str(i)):
                if not j.startswith(".") and temp<500:
                    image = cv2.imread("./Downloads/amigos/maps/"+str(i) + "/" + str(j))
                    gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
                    image1 = cv2.resize(gray, (120, 120))
                    image1=normalize(np.array(image1))
                    x = np.expand_dims(image1, axis=-1)
                    q2.append(x)
                    y2.append(y)
                    temp=temp+1


# In[13]:


for i in range(len(q1)):
    x_1.append(q1[i])
    y_1.append(y1[i])


# In[24]:


len(x_1)


# In[26]:


total_x=[]
total_x_neu=[]
total_
total_y=[]
c=[0]*5
for i in os.listdir("./Downloads/deap_heatmaps"):
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
        for j in os.listdir("./Downloads/deap_heatmaps/"+str(i)):
            if not j.startswith("."):
                image = cv2.imread("./Downloads/deap_heatmaps/" + str(i) + "/" + str(j))
                gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
                image1 = cv2.resize(gray, (120, 120))
                image1=normalize(np.array(image1))
                x = np.expand_dims(image1, axis=-1)
                total_x.append(x)
                total_y.append(y)
                temp=temp+1


# In[16]:


for i in os.listdir("./Downloads/amigos/maps/"):
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
        temp1 = 0
        for j in os.listdir("./Downloads/amigos/maps/"+str(i)):
            #print(temp1)
            if not j.startswith(".") and temp1<1000:
                image = cv2.imread("./Downloads/amigos/maps/" + str(i) + "/" + str(j))
                gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
                image1 = cv2.resize(gray, (120, 120))
                image1=normalize(np.array(image1))
                x = np.expand_dims(image1, axis=-1)
                total_x.append(x)
                total_y.append(y)
                temp1=temp1+1


# In[14]:


def sequence():

    seq = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontal flips
        iaa.Crop(percent=(0, 0.1)), # random crops

        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
        )
    ], random_order=True) # apply augmenters in random order
    
    return seq

#Flip and Random Rotate

def arguments():
    augs = [Flip(p=1),RandomRotate90(p=1)]
    return augs

#Augment an input image with n_affine transforms and n_flip's

def augment_image(image, n_affine = 2, n_flip = 0):
    
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

    for image,y in zip(x_data,y_data):

        image = image.reshape((120,120))
        extra_images = augment_image(image)

        extra_images = np.expand_dims(extra_images, axis = -1)

        list_x.append(extra_images)
        list_y.append(np.array([y for i in range(len(extra_images))]))

        i += 1

        #print('Creating Image Augmentation...')
        if i % 2000 == 0:
            print(i,'/',total_length)
    
    print('Done with adding images, now doing reshaping')
    
    x_images = np.array(list_x)
    y_images = np.array(list_y)
    x_images = x_images.reshape(-1,120,120,1)
    y_images = y_images.reshape(-1,5)

    print('Done.')

    return np.concatenate((x_data,x_images)), np.concatenate((y_data,y_images))


# In[15]:


#xt=normalize(np.array(total_x))
#total_x1= np.expand_dims(xt, axis=-1)
#print(len(total_x1),len(total_y))
xx,yy=append_augmented_images(np.array(x_1),np.array(y_1))
xf=[i for i in xx]
yf=[i for i in yy]
for i in range(500):
    xf.append(q2[i])
    yf.append(y2[i])
for i in range(2500):
    xf.append(x_neu[i])
    yf.append(y_neu[i])    
x_train, x_test, y_train, y_test = train_test_split(xf, yf, test_size=0.20, random_state=42)


# In[16]:


len(xf)


# In[20]:


history = model_dude.fit(np.array(x_train), np.array(y_train), validation_data=(np.array(x_test),np.array(y_test)),batch_size=256,epochs = 50)


# In[19]:


model_dude.save("trio1")
model_dude.save_weights("trio1.hdf5")
hist_df = pd.DataFrame(history.history)
hist_csv_file = 'trio1.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)


# In[10]:


predicted = model_dude.predict(np.array(x_train))
print(predicted)
emotion_list = ['HAHV', 'HALV', 'LAHV', 'LALV', 'Neutral']
y_pred_classes = model_dude.predict_classes(np.array(x_train))
print(y_pred_classes)
final_y=[]
y_pred_classes = keras.utils.to_categorical(y_pred_classes,num_classes=5)
conf_matrix = confusion_matrix(np.argmax(np.array(y_train), axis=1), np.argmax(np.array(y_pred_classes),axis=1), normalize = 'true')
df_conf = pd.DataFrame(conf_matrix, index = [i for i in emotion_list],
                  columns = [i for i in emotion_list])

plt.figure(figsize = (10,10))
sns.heatmap(df_conf, annot=True, cmap = 'Blues', fmt = '.2', cbar = False)
plt.plot()


# In[40]:


x_train,y_train=append_augmented_images(total_x,total_y)
x_train, x_test, y_train, y_test = train_test_split(np.array(x_train), np.array(y_train), test_size=0.20, random_state=42)


# In[16]:


x_train.shape


# In[23]:


model_dude = Sequential([
Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(60, 60, 1)), #1
Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(60, 60, 1)), #1
BatchNormalization(),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)), #2
Dropout(0.5),

Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'), #3
BatchNormalization(),
Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'), #4
BatchNormalization(),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)), #5
Dropout(0.5),

Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'), #3
BatchNormalization(),
Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'), #4
BatchNormalization(),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)), #5
Dropout(0.5),


Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'), #6
BatchNormalization(),
Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'), #7
BatchNormalization(),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)), #8
Dropout(0.5),

Flatten(),

Dense(2*2*2*num_features, activation='relu'), #9
Dropout(0.4),
Dense(2*2*num_features, activation='relu'), #10
Dropout(0.4),
Dense(2*num_features, activation='relu'), #10
Dropout(0.5),

Dense(7, activation='softmax')

])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
model.compile(optimizer='adam', loss='categorical_crossentropy' , metrics=['accuracy'])


# In[ ]:


history = model.fit(np.array(x_train),np.array(y_train),validation_data=(x_test,y_test),epochs = 100,callbacks=[es])


# In[33]:


np.array(total_x).shape


# In[ ]:




