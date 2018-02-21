'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import numpy as np
import pickle
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import matplotlib.pyplot as plt
import random
import time

max_training = 0
max_validation = 0
max_test = 0
count = 0


# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer

    # Output:
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W



# Replace this with your sigmoid implementation
def sigmoid(z):
    return  1/ (1+np.exp(np.array(z)*(-1)))
# Replace this with your nnObjFunction implementation
def nnObjFunction(params, *args):
    global count
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    obj_val = 0

    xp = np.append(training_data, np.ones((training_data.shape[0],1)), 1)

    aj = np.dot(xp, np.transpose(w1))

    zj = sigmoid(aj)

    zj = np.append(zj, np.ones((zj.shape[0], 1)),1)
    
    total_inputs = training_label.shape[0]

    bl = np.dot(zj, np.transpose(w2))

    ol = sigmoid(bl)

    yl = np.zeros((xp.shape[0],n_class))

    yl[range(total_inputs),training_label.astype(int)]=1
    
    # print("Y and Log")
    #print(yl.shape)
    #print(np.log(ol).shape)

    parta = np.multiply(yl, np.log(ol))
    sum_1 = np.add(parta, np.multiply(np.subtract(1,yl), np.log(np.subtract(1, ol))))

    sum_1 = np.divide(np.sum(sum_1),(-1)*xp.shape[0])

    # print("NEXT")
    partsum = np.sum(np.square(w1)) + np.sum(np.square(w2))

    sum_2 = (partsum) * np.divide(lambdaval,(2*xp.shape[0]))

    obj_val = sum_1 + sum_2

    #Calculate grad descent
    #Calculate w2
    delta = np.subtract(ol,yl)

    derivate2 = np.dot(np.transpose(delta), zj)

    grad_w2 = np.divide(np.add(derivate2,np.multiply(lambdaval,w2)),xp.shape[0])

    #calculate grad_w1
    w2mod = w2[:,0:w2.shape[1]-1]
    #print("modified_w2",modified_w2.shape)
    a1 = np.dot(delta,w2mod)
    #print("t1",t1.shape)
    # t2 = np.multiply(t1,training_data)
    zj = zj[:,0:zj.shape[1]-1]

    a2 = np.multiply(np.subtract(1,zj),zj)
    # print("t2" , t2.shape)
    #print("t3", t3.shape)
    w1_1 = np.dot(np.transpose(np.multiply(a2,a1)),xp)
    # print("partasum",partasum.shape)
    grad_w1 = np.add(w1_1,np.multiply(lambdaval,w1))/xp.shape[0]
    #print("grad_w1",grad_w1.shape)
    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    #obj_grad = np.array([])
    #print(obj_val)
    # print("Shape", obj_grad.shape)
    count = count +1
    
    print(count,"   ", obj_val)
    return (obj_val, obj_grad)
# Replace this with your nnPredict implementation
def nnPredict(w1,w2,data):
    data_with_bias = np.append(data, np.ones((data.shape[0],1)),1)

    zj = np.dot(data_with_bias, np.transpose(w1))

    zj = sigmoid(zj)

    zj = np.append(zj, np.ones((zj.shape[0],1)),1)

    ol = np.dot(zj, np.transpose(w2))

    ol = sigmoid(ol)

    labels = np.argmax(ol, axis = 1)
    return labels
    
    """**************Neural Network Script Starts here********************************"""
train_data, train_label, validation_data, validation_label, test_data, test_label,  = preprocess()
#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
#print n_input, train_data.size, train_label[1:10]

#print sigmoid([[0,-6],[-100,1],[2,0]])


#print initialWeights.shape
#trial = nnPredict(initial_w1, initial_w2, train_data)
ff = open('output_check_facenn.txt','w')
#
#for i in range(4,21,4) :
#    for j in range(0,61,5):
#    
count=0
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50
        
# set the number of nodes in output unit
n_class = 10

#print type(n_class), type(n_hidden)

# initialize the weights into some random matrices
w1 = initial_w1 = initializeWeights(n_input, n_hidden)
w2 = initial_w2 = initializeWeights(n_hidden, n_class)
#print initial_w1.shape,   initial_w2.shape

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)   


# set the regularization hyper-parameter
lambdaval = 25

print ("hiden: ", n_hidden , "  lamda: " , lambdaval)

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value. 

start_time = time.time()    

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

end_time = time.time()
#
## In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
## and nnObjGradient. Check documentation for this function before you proceed.
## nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)
#
#
## Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
#
## Test the computed parameters
#
predicted_label = nnPredict(w1, w2, train_data)
#
## find the accuracy on Training Dataset
#
#print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')
comp_training = (100 * np.mean((predicted_label == train_label).astype(float)))

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

#print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')
comp_validation = (100 * np.mean((predicted_label == validation_label).astype(float)))

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

#print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
comp_test = (100 * np.mean((predicted_label == test_label).astype(float)))

if  max_training < comp_training:
    max_training = comp_training
    # selected_features is a list of feature indices that you use after removing unwanted features in feature selection step
    
    #[w1,n_hidden, w_1, w_2, lambda]
    obj = [ n_hidden, w1, w2, lambdaval] #n_hidden,  w_1, lambda ]
    pickle.dump(obj, open('params_training_facenn.pickle', 'wb'))
    
if  max_validation < comp_validation:
    max_validation = comp_validation
    # selected_features is a list of feature indices that you use after removing unwanted features in feature selection step
    
    #[w1,n_hidden, w_1, w_2, lambda]
    obj = [n_hidden, w1, w2, lambdaval] #n_hidden,  w_1, lambda ]
    pickle.dump(obj, open('params_validation_facenn.pickle', 'wb'))
    
if  max_test < comp_test:
    max_test = comp_test
    # selected_features is a list of feature indices that you use after removing unwanted features in feature selection step
    
    #[w1,n_hidden, w_1, w_2, lambda]
    obj = [n_hidden, w1, w2, lambdaval] #n_hidden,  w_1, lambda ]
    pickle.dump(obj, open('params_test_facenn.pickle', 'wb'))           

    
ff.write("lambdaval    :"+str(lambdaval)+"\n"\
    +"number of hidden layers     :"+str(n_hidden)+"\n"\
    +"comp_training    :"+str(comp_training)+"\n"\
    +"comp_validation     :"+str(comp_validation)+"\n"\
    +"comp_test    :"+str(comp_test)+"\n"\
    +" no.of itr   :"+str(count)+"\n"\
    +" time taken  :"+str(end_time - start_time)+ "\n"\
    +"\n\n\n\n\n\n\n")
    
        
        
ff.close

