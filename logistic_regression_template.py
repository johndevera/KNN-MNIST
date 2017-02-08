from check_grad import check_grad
from utils import *
from logistic import *
import numpy as np
import matplotlib.pyplot as py



def run_logistic_regression(hyperparameters):
    # TODO specify training data
    train_inputs, train_targets = load_train()
    valid_inputs, valid_targets = load_valid()
    N, M = train_inputs.shape
    # N is number of examples; M is the number of features per example.
    # Logistic regression weights

    # TODO:Initialize to random weights here.
    weights = np.random.randn(M+1,1)

    weights = abs(np.random.rand(M+1, 1)) # M (784+1) weight rows in 1 column
    largestWeightIndex = weights.argsort(axis=0)
    largestWeightIndexNumber = largestWeightIndex[M,0] #go to the end of the sorted array
    largestWeight = weights[largestWeightIndexNumber,0]
    weights /= largestWeight  # scale the weights by the largest random number

    weights = np.random.randn(M+1,1)

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.

    run_check_grad(hyperparameters)

    # Begin learning with gradient descent
    logging = np.zeros((hyperparameters['num_iterations'], 5))

    for t in xrange(hyperparameters['num_iterations']):

        # Find the negative log likelihood and its derivatives w.r.t. the weights.
        f, df, predictions = logistic(weights, train_inputs, train_targets, hyperparameters) #prediction=y
        
        # Evaluate the prediction.
        cross_entropy_train, frac_correct_train = evaluate(train_targets, predictions) #prediction=y

        if np.isnan(f) or np.isinf(f):
            raise ValueError("nan/inf error")

        # update parameters
        weights = weights - hyperparameters['learning_rate'] * df / N

        # Make a prediction on the valid_inputs.
        predictions_valid = logistic_predict(weights, valid_inputs)

        # Evaluate the prediction.
        cross_entropy_valid, frac_correct_valid = evaluate(valid_targets, predictions_valid)

        # print some stats
        print ("ITERATION:{:4d}  TRAIN NLOGL:{:4.2f}  TRAIN CE:{:.6f} "
               "TRAIN FRAC:{:2.2f}  VALID CE:{:.6f}  VALID FRAC:{:2.2f}").format(
                   t+1, f / N, cross_entropy_train, frac_correct_train*100,
                   cross_entropy_valid, frac_correct_valid*100)
        logging[t] = [f / N, cross_entropy_train, frac_correct_train*100, cross_entropy_valid, frac_correct_valid*100]
    return logging

def run_check_grad(hyperparameters):
    """Performs gradient check on logistic function.
    """
    # This creates small random data with 7 examples and 
    # 9 dimensions and checks the gradient on that data.
    num_examples = 7
    num_dimensions = 9

    weights = np.random.randn(num_dimensions+1, 1)
    data    = np.random.randn(num_examples, num_dimensions)
    targets = (np.random.rand(num_examples, 1) > 0.5).astype(int)

    diff = check_grad(logistic,      # function to check
                      weights,
                      0.001,         # perturbation
                      data,
                      targets,
                      hyperparameters)
    print "diff =", diff

if __name__ == '__main__':
    # TODO: Set hyperparameters
    hyperparameters = {
                    'learning_rate': 0.1,
                    'weight_regularization': False, # boolean, True for using Gaussian prior on weights
                    'num_iterations': 700,
                    'weight_decay': 1 # related to standard deviation of weight prior
                    } # 0.001, 700, 0.1
    # average over multiple runs
    num_runs = 1
    logging = np.zeros((hyperparameters['num_iterations'], 5))
    for i in xrange(num_runs):
        logging += run_logistic_regression(hyperparameters)
    logging /= num_runs

 # TODO generate plots

    #


    X_ITR = np.arange(1, logging.shape[0]+1)
    Y_NLOGL = logging[:,0]
    Y_CE_TRAIN = logging[:,1]
    Y_FRAC_TRAIN = logging[:,2]
    Y_CE_VALID = logging[:,3]
    Y_FRAC_VALID = logging[:, 4]

    ITR = 'ITERATIONS'
    TRAIN_NLOGL = 'TRAIN NLOGL'
    TRAIN_CE = 'TRAIN CE'
    TRAIN_FRAC = 'TRAIN FRAC'
    VALID_CE = 'VALID CE'
    VALID_FRAC = 'VALID FRAC'
    CE = 'CROSS ENTROPY (BITS)'
    FRAC = 'FRACTION CORRECT (%)'
    VS = ' VS. '

    py.title('CROSS ENTROPY VS. ITERATIONS') #2.2.2 print for Small and Big
    py.xlabel(ITR)
    py.ylabel(CE)
    py.plot(X_ITR,Y_CE_TRAIN, label = 'CE TRAIN')#ce/training
    py.plot(X_ITR,Y_CE_VALID, label = 'CE VALID')
    py.legend(loc='upper center', shadow=False)
    py.show()

    py.title('TRAIN_NLOGL VS. ITERATIONS') #2.2.3 print for small and Big
    py.xlabel(ITR)
    py.ylabel(TRAIN_NLOGL)
    py.plot(X_ITR, Y_NLOGL, label = TRAIN_NLOGL) #NLOGL/training
    py.legend(loc='upper center', shadow=False)
    py.show()

    py.xlabel(ITR)
    py.ylabel(FRAC)
    py.title('FRACTION CORRECT VS. ITERATIONS')
    py.plot(X_ITR,Y_FRAC_TRAIN, label = TRAIN_FRAC)#frac/training
    py.plot(X_ITR,Y_FRAC_VALID, label = 'TEST FRAC')
    py.legend(loc='lower center', shadow=False)
    py.show()
