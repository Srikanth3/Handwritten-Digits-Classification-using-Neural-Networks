import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import pickle


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer

    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    return (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return 1 / (1 + np.exp(-z))


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""

    mat = loadmat('new_mnist_all.mat')  # loads the MAT object as a Dictionary

    # Pick a reasonable size for validation data

    train_data = np.array([]).reshape(0, 784)
    train_label = []
    validation_data = np.array([]).reshape(0, 784)
    validation_label = np.array([]).reshape(0, 1)
    test_data = np.array([]).reshape(0, 784)
    test_label = []

    # Stack training data into one array
    for i in range(0, 10):
        train_data = np.vstack((train_data, mat.get("train" + str(i))))

    # # Find redundant columns. Returns True for every column where every
    # # element is 0.
    # check_columns = np.all(train_data == train_data[0, :], axis=0)
    #
    # # Get indexes of the columns where everything is 0.
    # count = 0
    # data_zero_indexes = []
    #
    # for i in np.nditer(check_columns):
    #     if i:
    #         data_zero_indexes.append(count)
    #     count += 1
    #
    # # Delete the columns from train_data to give optimized array.
    # train_data = np.delete(train_data, data_zero_indexes, 1)

    # Setting the labels for reference.
    for i in range(0, 10):
        # temp_label = np.repeat(i, mat.get('train' + str(i)).shape[0])

        # temp_label = np.matrix(temp_label).T

        temp_label = i * np.ones(mat.get('train' + str(i)).shape[0])

        train_label.extend(temp_label)

    train_label = np.asarray(train_label)
    # print train_label.shape

    # Seperating Training data from Validation data.
    perm = np.random.permutation(range(train_data.shape[0]))
    temp_train_data = train_data[perm[0:50000], :]
    validation_data = train_data[perm[50000:], :]
    train_data = temp_train_data

    temp_train_label = train_label[perm[0:50000]]
    validation_label = train_label[perm[50000:]]
    train_label = temp_train_label

    # Get Test Data
    for i in range(0, 10):
        test_data = np.vstack((test_data, mat.get("test" + str(i))))

    # # Find redundant columns. Returns True for every column where every
    # # element is 0.
    # check_columns = np.all(test_data == test_data[0, :], axis=0)
    #
    # # Get indexes of the columns where everything is 0.
    # count = 0
    # data_zero_indexes = []
    #
    # for i in np.nditer(check_columns):
    #     if i:
    #         data_zero_indexes.append(count)
    #     count += 1
    #
    # # Delete the columns from train_data to give optimized array.
    # test_data = np.delete(test_data, data_zero_indexes, 1)

    # Setting the labels for reference.
    for i in range(0, 10):
        temp_label = i * np.ones(mat.get('test' + str(i)).shape[0])

        test_label.extend(temp_label)

    # Normalization
    test_data *= 1.0
    test_data /= 255.0
    train_data *= 1.0
    train_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log
    %   likelihood error function with regularization) given the parameters
    %   of Neural Networks, thetraining data, their corresponding training
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.

    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden
    %     layer to unit i in output layer."""
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    obj_val = 0
    error = 0.0
    grad_w1 = 0.0
    grad_w2 = 0.0

    for i in range(0, training_data.shape[0]):
        train_sample = training_data[i]
        train_sample = np.append(train_sample, 1.0)
        hidden_layer_out = sigmoid(np.dot(w1, train_sample))
        hidden_layer_out = np.append(hidden_layer_out, 1.0)
        output_layer_out = sigmoid(np.dot(w2, hidden_layer_out))

        # hidden_layer_output = hidden_layer_output.T

        # output_layer_out = output_layer_out.T

        target_output = np.zeros(n_class)
        target_output[training_label[i]] = 1.0

        error += (1 / 2) * (np.sum(np.square(target_output - output_layer_out)))

        delvalue = (target_output - output_layer_out) * \
            (np.ones(10) - output_layer_out) * output_layer_out

        grad_w2 += (-1 * (delvalue.reshape((n_class, 1)))) * \
            hidden_layer_out
        grad_w2_inner = (np.dot(delvalue, w2))[:-1]
        hidden_layer_out = hidden_layer_out[:-1]
        hidden_delta = (1 - hidden_layer_out) * \
            hidden_layer_out * grad_w2_inner
        grad_w1 += -1 * (hidden_delta.reshape((n_hidden,1))) * train_sample

    regulation = (1 / 2) * (lambdaval / training_data.shape[0]) * (
        (np.sum(np.square(w1))) + (np.sum(np.square(w2))))
    obj_val = ((1 / training_data.shape[0]) * error) + regulation

    """ gradient matrices """
    grad_w1 = (1 / training_data.shape[0]) * (grad_w1 + (lambdaval * w1))
    grad_w2 = (1 / training_data.shape[0]) * (grad_w2 + lambdaval * w2)
    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)

    flat_w1 = grad_w1.flatten()
    flat_w2 = grad_w2.flatten()
    obj_grad = np.concatenate((flat_w1, flat_w2), 0)

    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature
    %       vector of a particular image

    % Output:
    % label: a column vector of predicted labels"""

    labels = []
    #Your code here

    for i in range(0, data.shape[0]):

        data_sample = data[i]
        data_sample = np.append(data_sample, 1.0)
        hidden_layer_out = sigmoid(np.dot(w1, data_sample))
        hidden_layer_out = np.append(hidden_layer_out, 1.0)
        output_layer_out = sigmoid(np.dot(w2, hidden_layer_out))
        labels.append(np.argmax(output_layer_out))


    return np.asarray(labels)



"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 20

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate(
    (initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 0.2

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize
# module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True,
                     args=args, method='CG', options=opts)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1))                 :].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# Find the accuracy on Training Dataset

print('\n Training set Accuracy:' +
      str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# Find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 *
                                          np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# Find the accuracy on Validation Dataset

print('\n Test set Accuracy:' +
      str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')


pickleFile = open("params.pickle", 'wb')
pickle.dump(n_hidden, pickleFile)
pickle.dump(w1, pickleFile)
pickle.dump(w2, pickleFile)
pickle.dump(lambdaval, pickleFile)