# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 04:37:58 2018

@author: Lenovo-PC
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2 
from skimage import measure
import glob
from PCA import pca
from Main import Classify

TrainingDatasetPath = 'Data set/Training'
TestingDatasetPath = 'Data set/Testing'

classes = []
tmp_x_train = np.full((25, 2500), 0)
y_train = np.full((25, 5), 0)
idx = 0

# ** TRAINING **
for filename in glob.glob(TrainingDatasetPath + '/*.png'):
    img = cv2.imread(filename, 0)
    GrayImage = cv2.resize(img, (50, 50))
    tmp_x_train[idx, :] = np.array(GrayImage).reshape((1, 2500))
    image = filename
    print(image.split("/")[2][:-4])
    if image.split("/")[2][:-4] not in classes:
        classes.append(image.split("/")[2][:-4])
    y_train[idx, classes.index(image.split("/")[2][:-4])] = 1
    idx = idx + 1


# ** TESTING **
tmp_x_test = np.full((26, 2500), 0)
y_test = np.full((26, 5), 0)
idx = 0
for filename in glob.glob(TestingDatasetPath + '/*.png'):
    img = cv2.imread(filename, 0)
    GrayImage = cv2.resize(img, (50, 50))
    tmp_x_test[idx, :] = np.array(GrayImage).reshape((1, 2500))
    image = filename
    y_test[idx, classes.index(image.split("/")[2][:-4])] = 1
    idx = idx + 1


o = pca()
w = o.fit(23,tmp_x_train,50,0.00000000000001)

