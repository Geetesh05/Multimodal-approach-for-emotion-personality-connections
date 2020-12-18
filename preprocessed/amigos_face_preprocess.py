import cv2
import os
import pandas as pd
import numpy as np
os.chdir("/Users/currentwire/Desktop/thesis/")
#Function to extract frames per second
count = 0
counter = 1
emotions=["Neutral", "Happiness", "Sadness", "Surprise", "Fear", "Anger", "Disgust"]
file=pd.read_excel("dataset/Metadata_xlsx/SelfAsessment.xlsx")
for i in os.listdir("/Users/currentwire/Desktop/thesis/dataset/face/short/"):
  if not i.startswith("."):
    for j in os.listdir("/Users/currentwire/Desktop/thesis/dataset/face/short/"+str(i)):
        uid,vidid,_=str(j).split("_")
        userid=uid[1]
        emotion_folder=[]
        index=np.where(file["UserID"]==int(userid))
        index2=np.where(file["VideoID"][index[0]]== "'"+str(vidid)+"'")
        z= pd.DataFrame(file.loc[index2[0]])
        emotion_index=np.where( z.values[0][8:15]==1)
        for l in emotion_index[0]:

                    if l==0:
                        emotion_folder.append(emotions[0])
                    if l==1:
                        emotion_folder.append(emotions[1])
                    if l==2:
                        emotion_folder.append(emotions[2])
                    if l==3:
                        emotion_folder.append(emotions[3])
                    if l==4:
                        emotion_folder.append(emotions[4])
                    if l==5:
                        emotion_folder.append(emotions[5])
                    if l==6:
                        emotion_folder.append(emotions[6])
        print(emotion_folder)
        print("----")

        for d in emotion_folder:
            if not os.path.exists("/Users/currentwire/Desktop/thesis/dataset/experiment"+str(d)):
                os.makedirs("/Users/currentwire/Desktop/thesis/dataset/experiment/"+str(d))
            else:
                continue

        vidcap = cv2.VideoCapture(r'/Users/currentwire/Desktop/thesis/dataset/face/short/'+str(i)+"/"+str(j));
        count = 0
        success = True
        counter += 1
        temp=0
        while success:
            success,image = vidcap.read()
            print('read a new frame:',success)
            if count%25 == 0 and success==True or count%15==0:
                #image=cv2.imread(frame)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.3,
                    minNeighbors=3,
                    minSize=(30, 30)
                )

                print("[INFO] Found {0} Faces!".format(len(faces)))
                crops=[]
                for (x, y, w, h) in faces:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    face_crop = gray[y:y + h, x:x + w]
                    #face_crop = cv2.resize(face_crop, (48, 48))
                    crops.append(face_crop)
                    for t in emotion_folder:
                        cv2.imwrite('/Users/currentwire/Desktop/thesis/dataset/experiment/'+str(t)+ "/frames%d.jpg" % temp, face_crop)
            temp=temp+1
            count+=1

