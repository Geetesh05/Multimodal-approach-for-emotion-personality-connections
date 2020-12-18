import keras
import cv2
import numpy as np
import scipy.io
import pandas as pd
import os
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM, Dropout, Input, BatchNormalization
from keras.models import Model, Sequential
os.chdir("/Users/currentwire/Desktop/thesis/dataset/")
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
for t in os.listdir("./experiment/e/"):
  if not t.startswith("."):
    _,_,a=t.split("_")
    b,_=a.split(".")
    _,d=b.split("P")
    eeg= scipy.io.loadmat("./experiment/e/"+str(t))
    for i in range(16):
        key=eeg["VideoIDs"][0][i]
        if not os.path.exists("/Users/currentwire/Desktop/thesis/dataset/experiment/eeg/" +str(b)+"/"+ str(key[0])):
            os.makedirs("/Users/currentwire/Desktop/thesis/dataset/experiment/eeg/"+str(b)+"/" + str(key[0]))
        per_sec = []
        vidnor=eeg["joined_data"][0][i]
        vidnormal = pd.DataFrame(vidnor)
        c=0
        for j in range(len(vidnormal)):

            time_gap = 1 / 128.

            if (int(j) * time_gap) % 1 == 0 and j != 0:
                m = j - 128
                per_sec.append(vidnormal[m:j])
                #if not os.path.exists("/Users/currentwire/Desktop/thesis/dataset/experiment/eeg/" +str(b)+"/"+ str(key[0])+"/"+str(c)):
                #    os.makedirs("/Users/currentwire/Desktop/thesis/dataset/experiment/eeg/" +str(b)+"/"+ str(key[0])+"/"+str(c))
                theta = []
                slow_alpha = []
                alpha = []
                beta = []
                gamma = []
                bands = []
                matrix = np.zeros((9, 9))
                corrdinates = [(1, 3), (2, 0), (2, 2), (3, 1), (4, 0), (6, 0), (8, 3), (8, 5), (6, 8), (4, 8), (3, 7),
                           (2, 6), (2, 8), (1, 5)]
                for n in range(14):
                    f, Pxx_den = signal.welch(vidnormal[m:j][n], fs=128., scaling="spectrum")
                    t, s, a, be, g = getFiveBands_Power(f, Pxx_den)
                    theta.append(t)
                    slow_alpha.append(s)
                    alpha.append(a)
                    beta.append(be)
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
                figs = []
                total=np.zeros((9,9))
                for p in range(len(bands)):
                    for q in range(len(bands[p])):
                        x, y = corrdinates[q]
                        matrix[x][y] = bands[p][q]
                    total=total+matrix
                total=total/5
                plt.figure(figsize=(5, 5))
                fig = sns.heatmap(total, annot=False, cmap='gray', fmt='.2', cbar=False)
                # total = cv2.addWeighted(total, 0.5, fig, 0.5, 0)

                plt.savefig("/Users/currentwire/Desktop/thesis/dataset/experiment/eeg/" +str(b)+"/"+ str(key[0])+ "/frames%d.jpg" % c)
                #countt = countt + 1
                c = c + 1
