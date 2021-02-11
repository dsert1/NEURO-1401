from BackPropagation import MLP
from rbf import RBF
from PrepareDataset import Preparation
import numpy as np
import matplotlib.pyplot as plt
from PCA import pca
from sklearn.preprocessing import StandardScaler
import cv2
import pprint
pp = pprint.PrettyPrinter()
def Train():
    global MLPObj , PrepareObj , RBFObj , PCAObj , sc_x
    MLPObj = MLP()
    PrepareObj = Preparation()
    RBFObj = RBF()
    PCAObj = pca()
    x_train,y_train,x_test,y_test,Original_x_train , Original_x_test = PrepareObj.GetDataset("Data set/Training","Data set/Testing")

    PCAObj.LoadWeights()
    x_train = PCAObj.transform(Original_x_train)
    x_test = PCAObj.transform(Original_x_test)

    sc_x = StandardScaler()
    x_train = sc_x.fit_transform(x_train)
    x_test = sc_x.transform(x_test)

    # RBFObj.TrainTheModel_rbf(Neurons_Entry.get(), LearningRate_Entry.get(), MSE_Entry.get(), epochs_Entry.get(), 5,
                             # x_train, y_train, x_test, y_test)
    RBFObj.TrainTheModel_rbf(25,.01,250,1000,5, x_train,y_train,x_test,y_test)

    img = cv2.imread('Data set/Testing/left_squashed.png')
    RBFObj.Classify(img, ['left_squashed','Pear'])

if __name__ == '__main__':
    Train()