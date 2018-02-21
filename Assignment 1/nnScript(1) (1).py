# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import matplotlib.pyplot as plt
import random
import time


g_count = 0

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
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return  1.0/ ( 1.0 + np.exp ( -(z.astype(np.float64) ) ) )


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
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary
    
    # Pick a reasonable size for validation data

    # ------------Initialize preprocess arrays----------------------#
    train_preprocess = np.zeros(shape=(50000, 784))
    validation_preprocess = np.zeros(shape=(10000, 784))
    test_preprocess = np.zeros(shape=(10000, 784))
    train_label_preprocess = np.zeros(shape=(50000,))
    validation_label_preprocess = np.zeros(shape=(10000,))
    test_label_preprocess = np.zeros(shape=(10000,))
    # ------------Initialize flag variables----------------------#
    train_len = 0
    validation_len = 0
    test_len = 0
    train_label_len = 0
    validation_label_len = 0

    # ------------Start to split the data set into 6 arrays-----------#
    for key in mat:
        # -----------when the set is training set--------------------#
        if "train" in key:
            label = key[-1]  # record the corresponding label            
            tup = mat.get(key)
            #print type(tup)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)  # get the length of current training set
            tag_len = tup_len - 1000  # defines the number of examples which will be added into the training set

            # ---------------------adding data to training set-------------------------#
            train_preprocess[train_len:train_len + tag_len] = tup[tup_perm[1000:], :]
            train_len += tag_len

            train_label_preprocess[train_label_len:train_label_len + tag_len] = label
            train_label_len += tag_len

            # ---------------------adding data to validation set-------------------------#
            validation_preprocess[validation_len:validation_len + 1000] = tup[tup_perm[0:1000], :]
            validation_len += 1000

            validation_label_preprocess[validation_label_len:validation_label_len + 1000] = label
            validation_label_len += 1000

            # ---------------------adding data to test set-------------------------#
        elif "test" in key:
            label = key[-1]
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)
            test_label_preprocess[test_len:test_len + tup_len] = label
            test_preprocess[test_len:test_len + tup_len] = tup[tup_perm]
            test_len += tup_len
            # ---------------------Shuffle,double and normalize-------------------------#
    train_size = range(train_preprocess.shape[0])
    train_perm = np.random.permutation(train_size)
    train_data = train_preprocess[train_perm]
    train_data = np.double(train_data)
    train_data = train_data / 255.0
    train_label = train_label_preprocess[train_perm]

    validation_size = range(validation_preprocess.shape[0])
    vali_perm = np.random.permutation(validation_size)
    validation_data = validation_preprocess[vali_perm]
    validation_data = np.double(validation_data)
    validation_data = validation_data / 255.0
    validation_label = validation_label_preprocess[vali_perm]

    test_size = range(test_preprocess.shape[0])
    test_perm = np.random.permutation(test_size)
    test_data = test_preprocess[test_perm]
    test_data = np.double(test_data)
    test_data = test_data / 255.0
    test_label = test_label_preprocess[test_perm]

    # Feature selection
    # Your code here.
    #ff = train_data
    
    Total_Train_data = np.concatenate((train_data,validation_data), axis=0)
    # remove feature which is same across all the inputs, as its of no use for training.
    # np.all will return a bool array indicating if the coloumn (axis 0), has all element being same.
    # use mask to remove the coloumn and store the result. ~ is required as false means del, true mean keep.
    mask_toDel = ~(np.all(Total_Train_data == Total_Train_data[0,:],axis = 0))
    train_data = train_data[:,mask_toDel]
    validation_data = validation_data[:,mask_toDel]
    test_data = test_data[:,mask_toDel]

    
    #print to_del, train_data.shape
    
    #to_del = np.empty(0)    
    #for i in xrange(28*28):
    #    #print np.all(train_data[:,i] == train_data[0,i])
    #    if((Total_Train_data[:,i] == Total_Train_data[0,i]).all()):
    #        #print to_del
    #        to_del = np.append(to_del,i)   
    #ff = np.delete(ff, to_del,1)
    #print np.array_equal(ff,train_data)
    
    print('preprocess done')

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
    
    global g_count

    #print len(args)
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    obj_val = 0
    #print w1.shape, w2.shape
    
    
    # 1-of-K coding scheme
    total_inputs = training_label.shape[0]
    y = np.zeros([n_class,total_inputs])    
    y[training_label.astype(int),range(total_inputs)]=1
    
    
    # forward pass to find o
    temp1 = np.ones((training_data.shape[0],training_data.shape[1]+1));
    temp1[:,1:] = training_data
    #print temp.shape
    a1 = np.dot(w1,temp1.transpose())
    z = sigmoid(a1)
    #del temp
    temp = np.ones((z.shape[0]+1,z.shape[1]));
    temp[1:,:] = z
    a2 = np.dot(w2,temp)
    o = sigmoid(a2) 
    
    #δl   
    s = o-y
    
    #∂Ji / ∂w(2)lj = δlzj
    grad_w2 = (np.dot(s,temp.transpose()) + (lambdaval*w2))/total_inputs

    # Your code here   
    temp2 =np.dot( ( ( ( (1-temp) * temp ).transpose() ) * ( np.dot( s.transpose(), w2 ) ) ).transpose() , temp1)
    grad_w1 = ( temp2[1:,:] + (lambdaval*w1) )/total_inputs
    #print grad_w1.shape, ().shape
    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array

    obj_grad = np.array([])
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    
    
    #obj_val
    obj_val = ( np.sum( (y*np.log(o))+( (1-y) * ( np.log( (1-o) ) ) ) ) ) * (-1/total_inputs)
    obj_val = obj_val + (lambdaval/(2*total_inputs))*( np.sum(w1*w1) + np.sum(w2*w2) )
    g_count = g_count+1
    print g_count, obj_val
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

    #labels = np.array([])
    # Your code here
    temp = np.ones((data.shape[0],data.shape[1]+1));
    temp[:,1:] = data
    #print temp.shape
    a1 = np.dot(w1,temp.transpose())
    z1 = sigmoid(a1)
    #del temp
    temp = np.ones((z1.shape[0]+1,z1.shape[1]));
    temp[1:,:] = z1
    a2 = np.dot(w2,temp)
    a2 = sigmoid(a2)    
    
    return a2.argmax(axis = 0)


"""**************Neural Network Script Starts here********************************"""

g_count = 0


train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 40



args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.

#t1, t2 = nnObjFunction(initialWeights,n_input, n_hidden, n_class, train_data, train_label, lambdaval)

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')


"""
 when 20:
 Training set Accuracy:94.364%

 Validation set Accuracy:93.76%

 Test set Accuracy:93.92%
 
 
 
 when 0:
 Training set Accuracy:94.998%

 Validation set Accuracy:93.86%

 Test set Accuracy:94.62%
 
 
 when 80:
 
 Training set Accuracy:93.096%

 Validation set Accuracy:92.48%

 Test set Accuracy:92.93%
 
 when 40:
     
  Training set Accuracy:94.266%

 Validation set Accuracy:93.51%

 Test set Accuracy:94.15%
 
 
 """
