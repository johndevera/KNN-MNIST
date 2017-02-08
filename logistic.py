""" Methods for doing logistic regression."""
from utils import load_train
import numpy as np
from utils import sigmoid

trainX, trainY = load_train()

def logistic_predict(weights, data):
    """
    Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to the bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
    Outputs:
        y:          :N x 1 vector of probabilities of being second class. This is the output of the classifier.
    """
    # TODO: Finish this function
   # data = np.insert(data, 0, values=1, axis=1) #make data as N x (M+1). This is the same as x0 = 1

    M = weights.shape[0]
    N = data.shape[1]
    w = weights[0:M - 1]
    w0 = weights[M - 1]
    x = data

    z = np.dot(x, w) + w0

    y = sigmoid(z)
    return y


def evaluate(targets, y):
    """
    Compute evaluation metrics.
    Inputs:
        targets : N x 1 vector of targets. TRAIN_TARGETS
        y       : N x 1 vector of probabilities.
    Outputs:
        ce           : (scalar) Cross entropy. CE(p, q) = E_p[-log q]. Here we want to compute CE(targets, y)
        frac_correct : (scalar) Fraction of inputs classified correctly.
    """
    # TODO: Finish this function

    ce = (-1.0)*(np.dot(targets.T, np.log(y)) + np.dot((1-targets).T, np.log(1.0-y)))
    #ce = -np.dot(targets.T, np.log(y)) - np.dot((1-targets).T, np.log(1-y))
    #ce = (-1)*sum(targets*np.log(y)) - sum((1-targets)*np.log(1-y))
    ce =ce[0,0]
    error = np.round(np.sqrt((targets - y)**2))
    frac_correct = 1.0-float(np.count_nonzero(error))/float(targets.shape[0]) #get number of non zero elements divided by total elements


    return ce, frac_correct


def logistic(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:       The sum of the loss over all data points. This is the objective that we want to minimize.
        df:      (M+1) x 1 vector of accumulative derivative of f w.r.t. weights, i.e. don't need to average over number of sample
        y:       N x 1 vector of probabilities.
    """

    y = logistic_predict(weights, data)

    if hyperparameters['weight_regularization'] is True:
        f, df = logistic_pen(weights, data, targets, hyperparameters)
    else:
        # TODO: compute f and df without regularization
        f, df = logistic_regression(weights, data, targets, hyperparameters)

    return f, df, y


def logistic_pen(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:             The sum of the loss over all data points. This is the objective that we want to minimize.
        df:            (M+1) x 1 vector of accumulative derivative of f w.r.t. weights, i.e. don't need to average over number of sample
    """

    # TODO: Finish this function
    f, df = logistic_regression(weights, data, targets, hyperparameters)

    return f, df

def logistic_regression(weights, data, targets, hyperparameters): #I made this to handle logistic regression. With or without Regularization

    M = weights.shape[0]

    x = data
    t = targets
    w = weights[0:M - 1]
    w0 = weights[M - 1]

    z = np.dot(x, w) + w0
    e = np.exp(-z)
    L = np.dot((1.0 - t).T, z) + np.sum(np.log(1.0 + e),axis=0)
    a = 0
    Reg = 0

    if hyperparameters['weight_regularization'] is True:
        a = hyperparameters['weight_decay']
        Reg = 0.5*a*np.dot(weights.T, weights)

    loss = Reg + L
    #print('loss shape', loss.shape)

    f = loss[0][0]

    dW = a * w + np.dot((1.0 - t).T, x).T - np.dot((e / (1.0 + e)).T, x).T
    #dW = a*w + np.dot((1 - t).T, x).T - np.dot(x.T, e / (1 + e))
    dW0 = sum(1 - t) - np.dot(e.T, 1 / e)

    df = np.zeros((dW.shape[0] + 1, 1))
    df[0:M - 1] = dW[0:M - 1]
    df[M - 1] = dW0
#
    #print("df", df)

    return f, df
"""

def logistic_regression(weights, data, targets, hyperparameters): #I made this to handle logistic regression. With or without Regularization

    x = data
    t = targets
    M = weights.shape[0]
    w = weights[0:M - 1]
    w0 = weights[M - 1]
    z = np.dot(x, w) + w0
    e = np.exp(-z)
    L = np.dot((1 - t).T, z) + sum(np.log(1 + e))
    a = 0
    Reg = 0


    if hyperparameters['weight_regularization'] is True:
        a = hyperparameters['weight_decay']
        Reg = 0.5*a*np.dot(weights.T, weights)

    loss = Reg + L
    f = loss[0][0]

    dW = np.zeros((M-1,1))

    for j in range(0, M-1):
        xj = x[:,j]

        #tj = t[:,j]
        tj = t
        z = np.dot(x, w) + w0
        e = np.exp(z)
        dW[j] = a*w[j] + np.dot((1-tj).T, xj) - np.dot(xj.T, e/(1+e))
       # print('awj', awj.shape)
        dW0 = sum(1-tj) - sum(e/(1+e))


    df = np.zeros((dW.shape[0] + 1, 1))
    df[0:M - 1] = dW[0:M - 1]
    df[M-1] = dW0

    return f, df
"""