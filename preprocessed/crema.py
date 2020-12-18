import os
import numpy as np
import cv2

os.chdir("/Users/currentwire/Downloads/CREMA-D/VideoFlash/")
"""
emotions=["Neutral", "Happiness", "Sadness", "Surprise", "Fear", "Anger", "Disgust"]
for d in emotions:
    if not os.path.exists("/Users/currentwire/Documents/GitHub/CREMA-D/processed/" + str(d)):
        os.makedirs("/Users/currentwire/Documents/GitHub/CREMA-D/processed/" + str(d))
"""
count = 0
count1 = 0

for i in os.listdir("/Users/currentwire/Downloads/CREMA-D/VideoFlash/"):
  if count<1000:
    _,_,key,_ =i.split("_")
    name=0
    if str(key)=="ANG":
        name="Anger"
    if str(key) == "DIS":
        name = "Disgust"
    if str(key)=="HAP":
        name="Happiness"
    if str(key)=="NEU":
        name="Neutral"
    if str(key)=="FEA":
        name="Fear"
    if str(key)=="SAD":
        name="Sadness"
    if not os.path.exists('/Users/currentwire/Downloads/CREMA-D/processed/' + str(name)):
        os.makedirs('/Users/currentwire/Downloads/CREMA-D/processed/' + str(name))
    vidcap = cv2.VideoCapture(r'/Users/currentwire/Downloads/CREMA-D/VideoFlash/' + str(i))
    success = True
    while success:
        success, image = vidcap.read()
        print('read a new frame:', success)
        if success == True:
            # image=cv2.imread(frame)

            cv2.imwrite('/Users/currentwire/Downloads/CREMA-D/processed/'+str(name)+"/"+"frames%d.jpg" % count1,
                image)
        count1=count1+1
    count=count+1
    """
        elif count>25000:
            break

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=3,
                minSize=(30, 30)
            )

            print("[INFO] Found {0} Faces!".format(len(faces)))
            crops = []
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                face_crop = gray[y:y + h, x:x + w]
                # face_crop = cv2.resize(face_crop, (48, 48))
                crops.append(face_crop)
                cv2.imwrite(
                        '/Users/currentwire/Documents/GitHub/CREMA-D/processed/' + str(name) + "/frames%d.jpg" % count,
                        face_crop)
            
        count += 1
        """
