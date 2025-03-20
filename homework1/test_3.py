from problem3 import *
import sys
import pytest 
import importlib.metadata 

import numpy as np
'''
    Unit test 3  (Total Points: 20)
    This file includes unit tests for problem3.py.
'''

def test_python_environment():
    assert sys.version_info[0]==3 # require python 3.11 or above 
    assert sys.version_info[1]>=11
    


# ============= Class: Linear_Regression ===========================
# Total Points: 0.0


# ============= Class: Linear_Regression_SE ===========================
# Total Points: 10.0
# ***********  Method: train  (Class: Linear_Regression_SE) **************** 
# Total Points: 10.0

# ---- Test Case: test_3instances (Method: train, Class: Linear_Regression_SE) 
# Goal: test when there are 3 data samples and 2 features. After training, the parameter w of the model should be set as the optimal value based on the training samples
# Points: 5.0
@pytest.mark.Linear_Regression_SE_train
def test_Linear_Regression_SE_train_test_3instances():
    X = np.array([[ 1.,-1.],  # the first instance,
                  [ 1., 0.],  # the second instance
                  [ 1., 1.]])
    y = np.array([1.5,2.5,3.5])
    m = Linear_Regression_SE(p=2) # create the regression model
    m.train(X,y)
    assert type(m.w) == np.ndarray
    assert m.w.shape == (2,) 
    assert np.allclose(m.w, [2.5,1.], atol = 1e-2) 
# --------------------------------

# ---- Test Case: test_random (Method: train, Class: Linear_Regression_SE) 
# Goal: test when there are random data samples. After training, the parameter w of the model should be set as the optimal value based on the training samples
# Points: 5.0
@pytest.mark.Linear_Regression_SE_train
def test_Linear_Regression_SE_train_test_random():
    for _ in range(20):
        p = np.random.randint(2,8) # number of features
        n = np.random.randint(200,400) # number of samples
        w_true = np.random.random(p) # true weights of the linear model
        X = np.random.random((n,p))*10 # generate random feature values
        e = np.random.randn(n)*0.01 # random errors on the labels
        y = np.dot(X,w_true) + e # compute the labels with true linear parameters and error terms
        m = Linear_Regression_SE(p) # create the regression model
        m.train(X,y)
        assert np.allclose(m.w,w_true, atol = 0.1)
# --------------------------------

# ============= Class: Linear_Regression_Ridge ===========================
# Total Points: 10.0

# ***********  Method: train  (Class: Linear_Regression_Ridge) **************** 
# Total Points: 10.0

# ---- Test Case: test_3instances (Method: train, Class: Linear_Regression_Ridge) 
# Goal: test when there are 3 data samples and 2 features. After training, the parameter b of the model should be set as the optimal value based on the training samples
# Points: 5.0
@pytest.mark.Linear_Regression_Ridge_train
def test_Linear_Regression_Ridge_train_test_3instances():
    X = np.array([[ 1.,-1.],  # the first instance,
                  [ 1., 0.],  # the second instance
                  [ 1., 1.]])
    y = np.array([1.5,2.5,3.5])
    m = Linear_Regression_Ridge(p=2,alpha=1000.) # if the L2 regularization term has a large weight
    m.train(X,y)
    assert np.allclose(m.w, [0.,0.], atol = 1e-2) 
    m = Linear_Regression_Ridge(p=2,alpha=0.) # if the L2 regularization term has zero weight
    m.train(X,y)
    assert np.allclose(m.w, [2.5,1.], atol = 1e-1) 
    m = Linear_Regression_Ridge(p=2,alpha=2.) # if the L2 regularization term has a weight of 2
    m.train(X,y)
    assert np.allclose(m.w, [1.5,0.5], atol = 1e-1) 
# --------------------------------

# ---- Test Case: test_random (Method: train, Class: Linear_Regression_Ridge) 
# Goal: test when there are random data samples. After training, the parameter w of the model should be set as the optimal value based on the training samples
# Points: 5.0
@pytest.mark.Linear_Regression_Ridge_train
def test_Linear_Regression_Ridge_train_test_random():
    for _ in range(20):
        p = np.random.randint(2,8) # number of features
        n = np.random.randint(200,400) # number of samples
        w_true = np.random.random(p) # true weights of the linear model
        X = np.random.random((n,p))*10 # generate random feature values
        e = np.random.randn(n)*0.01 # random errors on the labels
        y = np.dot(X,w_true) + e # compute the labels with true linear parameters and error terms
        m = Linear_Regression_Ridge(p) # create the regression model
        m.train(X,y)
        assert np.allclose(m.w,w_true, atol = 0.1)
# --------------------------------


