import cv2
import os
import shutil
import numpy as np
os.chdir("/Users/currentwire/Desktop/thesis/dataset/")
emotion_list=['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear','others']
total_x=[]
total_y=[]
for i in os.listdir("/Volumes/TheJoker/amigos/rest"):
    if not i.startswith("."):
        count=0
        _,a,_b=str(i).split("_")
        if not os.path.exists("/Volumes/TheJoker/amigos/frames/2sec/" + str(a)):
            os.makedirs("/Volumes/TheJoker/amigos/frames/2sec/" + str(a))
        for j in os.listdir("/Volumes/TheJoker/amigos/rest/"+str(i)):
          if not j.startswith("."):
            _,b,_=str(j).split("_")
            vid = cv2.VideoCapture("/Volumes/TheJoker/amigos/rest/"+str(i)+"/"+str(j))
            if not os.path.exists("/Volumes/TheJoker/amigos/frames/2sec/" + str(a)+"/"+str(b)):
                os.makedirs("/Volumes/TheJoker/amigos/frames/2sec/" + str(a)+"/"+str(b))
            count = 0
            success = True
            temp = 0
            images = []
            while success:
                success, image = vid.read()
                print('read a new frame:', success)
                if success == True:
                    if count % 15==0 or count % 25 == 0:
                        cv2.imwrite(
                             "/Volumes/TheJoker/amigos/frames/2sec/" + str(a) + "/" + str(b) + "/frames%d.jpg" % temp,
                            image)
                        temp = temp + 1
                count=count+1