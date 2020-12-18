import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import cv2
import tensorflow as tf
from scipy import signal
import scipy.io
def getBand_Power(freqs, power, lower, upper):
    ''' Sum band power within desired frequency range '''
    low_idx = np.array(np.where(freqs <= lower)).flatten()
    up_idx = np.array(np.where(freqs > upper)).flatten()
    band_power = np.sum(power[low_idx[-1]:up_idx[0]])
    #return [band_power,freqs[low_idx[-1]:up_idx[0]]]
    return band_power
def getFiveBands_Power(freqs, power):
    ''' Calculate 5 bands power '''
    theta_power = getBand_Power(freqs, power, 3, 7)
    slow_alpha_power = getBand_Power(freqs, power, 8, 10)
    alpha_power = getBand_Power(freqs, power, 10, 13)
    beta_power = getBand_Power(freqs, power, 14, 29)
    gamma_power = getBand_Power(freqs, power, 30, 47)

    return theta_power, slow_alpha_power, alpha_power, beta_power, gamma_power
def category(y):
        y1=[]
        if y[0] > 7 and y[1] > 7:
                y1 = "HAHV"
        elif y[0] > 7 and y[1] < 3:
                y1 = "HALV"
        elif y[0] < 3 and y[1] > 7:
                y1 = "LAHV"
        elif y[0] < 3 and y[1] < 3:
                y1 = "LALV"
        else:
            y1 = "neutral"
        return y1

#ch_list=["AF3","F7","F3","FC5","T7","P7","O1","O2","P8","T8","FC6","F4","F8","AF4"]
#channel=pd.read_excel("/Volumes/TheJoker/deap/LIST.xlsx")
key=[]
count=0
"""
for i in range(len(channel["Fp1"])):
  for j in ch_list:
    if str(channel["Fp1"][i])==str(j):
      key.append(i+1)
"""
for ttt in os.listdir("/Volumes/TheJoker/amigos/mainfiles/"):
  if not ttt.startswith("."):
   print(str(ttt))
   mat2 = scipy.io.loadmat("/Volumes/TheJoker/amigos/mainfiles/" + str(ttt))
   for i in range(16):
        eeg=[]
        print(mat2["labels_selfassessment"][0][i][0][0:5])
        yes = category(mat2["labels_selfassessment"][0][i][0][0:5])
        if not os.path.exists("/Volumes/TheJoker/amigos/eeg_overall/" + str(yes)):
            os.makedirs("/Volumes/TheJoker/amigos/eeg_overall/" + str(yes))
        #for j in range(len(mat2["joined_data"][0][i])):
              #if j in key:
        eeg.append(mat2["joined_data"][0][i])
        theta=[]
        slow_alpha=[]
        alpha=[]
        beta=[]
        gamma=[]
        bands=[]
        corrdinates = [(1, 3), (2, 0), (2, 2), (3, 1), (4, 0), (6, 0), (8, 3), (8, 5), (6, 8), (4, 8), (3, 7), (2, 6),
                     (2, 8), (1, 5)]
        for p in range(14):
                f, Pxx_den = signal.welch(np.transpose(mat2["joined_data"][0][i])[p], fs=128.,scaling="spectrum")
                t,s,a,b,g=getFiveBands_Power(f,Pxx_den)
                theta.append(t)
                slow_alpha.append(s)
                alpha.append(a)
                beta.append(b)
                gamma.append(g)
        theta=theta/np.sum(theta)
        slow_alpha = slow_alpha / np.sum(slow_alpha)
        alpha = alpha / np.sum(alpha)
        beta = beta / np.sum(beta)
        gamma = gamma / np.sum(gamma)
        bands.append(theta)
        bands.append(slow_alpha)
        bands.append(alpha)
        bands.append(beta)
        bands.append(gamma)
        for m in range(len(bands)):
            matrix = np.zeros((9, 9))
            for n in range(len(bands[m])):
                  x, y = corrdinates[n]
                  count=count+1
                  matrix[x][y] = bands[m][n]

            plt.figure(figsize = (5,5))
            fig=sns.heatmap(matrix, annot=False, cmap = 'gray', fmt = '.2', cbar = False)
            plt.savefig("/Volumes/TheJoker/amigos/eeg_overall/"+str(yes)+"/frames%d.jpg" % count)