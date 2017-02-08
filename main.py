from utils import *
from plot_digits import *
from run_knn import run_knn
import numpy as np
import matplotlib.pyplot as py



def knn(k, dataIn, dataOut, testIn, testOut, text):
    N = range(1, k+1, 2)
    xaxis = np.zeros((len(N),1))
    yaxis = np.zeros((len(N),1))

    for count in N:  # VALID DATA
        index = N.index(count)
        kNNlabels = run_knn(count, dataIn, dataOut, testIn)
        error = testOut - kNNlabels  # get the difference between valid data and returned labels from KNN
        totalErrors = sum( error * error)  # make sure the error values are non negative (all positive) then find total number of errors
        totalTest= testOut.shape[0]  # get the size of the total valid data
        totalCorrect = (1 - (totalErrors / totalTest)) * 100
        yaxis[index] = totalCorrect
        print(text, "K:", count, "Error: ", totalCorrect)

    print("---------------------------------")
    py.plot(N, yaxis) #NLOGL/training
    py.xlabel('K Nearest Neighbours')
    py.ylabel('% Correct')
    py.title(text) #2.2.3 print for small and Big
    #py.title(text, str(N), "Nearest Neighbours")
    #py.title("{:s}, for {:1d} Nearest Neighbours").format(text, count) #2.2.3 print for small and Big
    py.show()


trainX, trainY = load_train()
trainSmallX, trainSmallY = load_train_small()
validX, validY = load_valid()
testX, testY = load_test()



k = 9
knn(k, trainX, trainY, validX, validY, "Train vs. Valid")
knn(k, trainX, trainY, testX, testY, "Train vs. Test")
knn(k, trainSmallX, trainSmallY, validX, validY, "TrainSmall vs. Valid")
knn(k, trainSmallX, trainSmallY, testX, testY, "TrainSmall vs. Test")
