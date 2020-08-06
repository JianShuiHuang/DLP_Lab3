# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 20:15:44 2020

@author: ivis
"""


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def show_AccuracyCurve(a1, a2, b1, b2, Epochs):
    plt.title('Accuracy(%)', fontsize = 18)
    x = []
    for i in range(Epochs):
        x.append(i)
    
    plt.plot(x, a1, 'bo')
    plt.plot(x, a2, 'ro')
    plt.plot(x, b1, 'go')
    plt.plot(x, b2, 'ko')
    plt.show()

def show_AccuracyCurve1(a1, a2, Epochs):
    plt.title('Accuracy(%)', fontsize = 18)
    x = []
    for i in range(Epochs):
        x.append(i)
        plt.plot(i, a1[i], 'bo')
        plt.plot(i, a2[i], 'ro')
    
    plt.plot(x, a1, a2)
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    pic = confusion_matrix(y_true = y_true, y_pred = y_pred)
        
    fig, ax = plt.subplots()
    ax.matshow(pic, cmap=plt.cm.Blues, norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1.0, clip=False), alpha=0.8)
    
    for i in range(pic.shape[0]):
        for j in range(pic.shape[1]):
            ax.text(x=j, y=i, s=round(pic[i, j], 4), va='center', ha='center')
    
    plt.title('Normalized confusion matrix', fontsize = 18)       
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
   

y_true = np.load("y_true.npy")

train_accuracy18 = np.load("train_accuracy_list.npy")
test_accuracy18 = np.load("test_accuracy_list.npy")
y_pred18 = np.load("test_y_pred.npy")

train_accuracypre18 = np.load("pretrain_accuracy_list.npy")
test_accuracypre18 = np.load("pretest_accuracy_list.npy")
y_predpre18 = np.load("pretest_y_pred.npy")

show_AccuracyCurve(train_accuracy18, test_accuracy18, train_accuracypre18, test_accuracypre18, 10)
#plot_confusion_matrix(y_true, y_pred18)
#plot_confusion_matrix(y_true, y_predpre18)

"""
train_accuracy50 = np.load("train_accuracy50_list.npy")
test_accuracy50 = np.load("test_accuracy50_list.npy")
y_pred50 = np.load("test_y_pred50.npy")
"""

train_accuracypre50 = np.load("pretrain_accuracy50_list.npy")
test_accuracypre50 = np.load("pretest_accuracy50_list.npy")
y_predpre50 = np.load("pretest_y_pred50.npy")


#show_AccuracyCurve(train_accuracy50, test_accuracy50, train_accuracypre50, test_accuracypre50, 5)
#plot_confusion_matrix(y_true, y_pred50)
#plot_confusion_matrix(y_true, y_predpre50)



show_AccuracyCurve1(train_accuracypre50, test_accuracypre50, 5)
