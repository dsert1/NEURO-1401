# @author: Deniz B. Sert, Shawn Shivdat
# @version: February 11, 2021
# @inspiration from: https://github.com/AbdullahMahmoud/Object-Detection-and-Recognition-using-Neural-Networks

# ** IMPORTS **
from BackPropagation import MLP
from rbf import RBF
from PrepareDataset import Preparation
import numpy as np
import matplotlib.pyplot as plt
from PCA import pca
from sklearn.preprocessing import StandardScaler
import cv2
import pprint
import time
import matplotlib.pyplot as plt
import numpy as np
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
    RBFObj.TrainTheModel_rbf(25, .01, 250, 1000,5, x_train,y_train,x_test,y_test)



    squares = {'[ORIGINAL] small_square.png': 'blue',
               '45_square.png': 'green',
               'left_82square.png': 'orange',
               '63_square.png': 'red'}
    counter = 0
    times = {}
    file_prefix = 'Data set/Training/'
    NUM_OF_TRIALS = 10


    while counter < NUM_OF_TRIALS:
        if counter % 10 == 0:
            print(counter)
        for square in squares:
            times.setdefault(square, []).append(classify_image(file_prefix+square, [square.strip('.png'), 'Pear']))
        counter += 1
    # pp.pprint(times)

    for square, times in times.items():
        plt.plot(times, label=square, color=squares[square])
        max_time = np.max(times)
        plt.annotate(str(round(np.max(times), 3)), (times.index(max_time), max_time))
        plt.legend()
    plt.title('Performance Between Squares with N = ' + str(NUM_OF_TRIALS) + ' trials')
    plt.xlabel('No. of trials')
    plt.ylabel('Time in seconds')
    plt.xticks(np.arange(0, NUM_OF_TRIALS, step=1))

    plt.savefig('12.png')
    plt.show()


def classify_image(img_file, classes_list):
    '''
    Classifying image based on training data and input params
    img_file: a png file
    classes_list: array of strings

    returns the time it took to classify
    '''
    img = cv2.imread(img_file)
    start = time.time()
    RBFObj.Classify(img, classes_list)
    # print(classes_list[0] + ' took: ', time.time() - start, ' seconds to recognize.')

    return time.time() - start





if __name__ == '__main__':
    Train()
