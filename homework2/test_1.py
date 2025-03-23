from problem1 import *
import sys
import pytest 
import importlib.metadata 

import numpy as np
'''
    Unit test 1  (Total Points: 40)
    This file includes unit tests for problem1.py.
'''

def test_python_environment():
    assert sys.version_info[0]==3 # require python 3.11 or above 
    assert sys.version_info[1]>=11
    installed_pkg = {pkg.metadata['Name'] for pkg in importlib.metadata.distributions()}
    assert len(set(['pytest', 'numpy']) - installed_pkg)==0 # Check if any required package is missing. If missing, please follow the instructions at the begining of problem1.py to install the packages


# ============= Class: Linear_Classification ===========================
# Total Points: 0.0



# ============= Class: SGD ===========================
# Total Points: 0.0



# ============= Class: Linear_SVM ===========================
# Total Points: 40.0

# ***********  Method: compute_gradient  (Class: Linear_SVM) **************** 
# Total Points: 20.0

# ---- Test Case: test_2d (Method: compute_gradient, Class: Linear_SVM) 
# Goal: test a simple example with 2 dimensional features, when f(x)*y < 1 
# Points: 2.0
@pytest.mark.Linear_SVM_compute_gradient
def test_Linear_SVM_compute_gradient_test_2d():
    x = np.array([1.,1.])
    y = -1.
    m = Linear_SVM(p=2)
    m.w = np.array([1.,1.])
    m.b = 1.
    dL_dw, dL_db = m.compute_gradient(x,y,l=1.)
    assert type(dL_dw) == np.ndarray
    assert dL_dw.shape == (2,)
    assert np.allclose(dL_dw, [2,2], atol = 1e-3) 
    assert dL_db == 1.
# --------------------------------

# ---- Test Case: test_3d (Method: compute_gradient, Class: Linear_SVM) 
# Goal: test a simple example with 3 dimensional features, when f(x)*y < 1 
# Points: 2.0
@pytest.mark.Linear_SVM_compute_gradient
def test_Linear_SVM_compute_gradient_test_3d():
    x = np.array([1.,1.,1.])
    y = -1.
    m = Linear_SVM(p=3)
    m.w = np.array([1.,1.,1.])
    m.b = 1.
    dL_dw, dL_db = m.compute_gradient(x,y,l=1.)
    assert type(dL_dw) == np.ndarray
    assert dL_dw.shape == (3,)
    assert np.allclose(dL_dw, [2,2,2], atol = 1e-3) 
    assert dL_db == 1.
# --------------------------------

# ---- Test Case: test_b1 (Method: compute_gradient, Class: Linear_SVM) 
# Goal: test a simple example with 2 dimensional features, when f(x)*y < 1 and a different label y
# Points: 2.0
@pytest.mark.Linear_SVM_compute_gradient
def test_Linear_SVM_compute_gradient_test_b1():
    x = np.array([1.,1.])
    y = 1.
    m = Linear_SVM(p=2)
    m.w = np.array([-1.,-1.])
    m.b = 1.
    dL_dw, dL_db = m.compute_gradient(x,y,l=1.)
    assert type(dL_dw) == np.ndarray
    assert dL_dw.shape == (2,)
    assert np.allclose(dL_dw, [-2,-2], atol = 1e-3) 
    assert dL_db == -1.
# --------------------------------

# ---- Test Case: test_w1 (Method: compute_gradient, Class: Linear_SVM) 
# Goal: test a simple example with 2 dimensional features, when f(x)*y < 1 and a different weight and feature x
# Points: 4.0
@pytest.mark.Linear_SVM_compute_gradient
def test_Linear_SVM_compute_gradient_test_w1():
    x = np.array([4.,1.])
    y = -1.
    m = Linear_SVM(p=2)
    m.w = np.array([1.,2.])
    m.b = 1.
    dL_dw, dL_db = m.compute_gradient(x,y,l=1.)
    assert type(dL_dw) == np.ndarray
    assert dL_dw.shape == (2,)
    assert np.allclose(dL_dw, [5,3], atol = 1e-3) 
    assert dL_db == 1.
# --------------------------------

# ---- Test Case: test_l1 (Method: compute_gradient, Class: Linear_SVM) 
# Goal: test a simple example with 2 dimensional features, when f(x)*y < 1 and a different l
# Points: 2.0
@pytest.mark.Linear_SVM_compute_gradient
def test_Linear_SVM_compute_gradient_test_l1():
    x = np.array([4.,1.])
    y = -1.
    m = Linear_SVM(p=2)
    m.w = np.array([1.,2.])
    m.b = 1.
    dL_dw, dL_db = m.compute_gradient(x,y,l=5.)
    assert type(dL_dw) == np.ndarray
    assert dL_dw.shape == (2,)
    assert np.allclose(dL_dw, [9,11], atol = 1e-3) 
    assert dL_db == 1.
# --------------------------------

# ---- Test Case: test_2d2 (Method: compute_gradient, Class: Linear_SVM) 
# Goal: test a simple example with 2 dimensional features, when f(x)*y > 1 
# Points: 4.0
@pytest.mark.Linear_SVM_compute_gradient
def test_Linear_SVM_compute_gradient_test_2d2():
    x = np.array([1.,2.])
    y = 1.
    m = Linear_SVM(p=2)
    m.w = np.array([3.,4.])
    m.b = 1.
    dL_dw, dL_db = m.compute_gradient(x,y,l=1.)
    assert np.allclose(dL_dw, [3,4], atol = 1e-3) 
    assert dL_db == 0.
# --------------------------------

# ---- Test Case: test_l2 (Method: compute_gradient, Class: Linear_SVM) 
# Goal: test a simple example with 2 dimensional features, when f(x)*y > 1 with a different l value 
# Points: 4.0
@pytest.mark.Linear_SVM_compute_gradient
def test_Linear_SVM_compute_gradient_test_l2():
    x = np.array([1.,2.])
    y = 1.
    m = Linear_SVM(p=2)
    m.w = np.array([3.,4.])
    m.b = 1.
    dL_dw, dL_db = m.compute_gradient(x,y,l=5.)
    assert np.allclose(dL_dw, [15,20], atol = 1e-3) 
    assert dL_db == 0.
# --------------------------------


# ***********  Method: train  (Class: Linear_SVM) **************** 
# Total Points: 20.0

# ---- Test Case: test_2d2s (Method: train, Class: Linear_SVM) 
# Goal: test when there are 2 data samples and 2 features. After training, the parameters w and b of the model should be set as the optimal parameter values based on the training samples
# Points: 4.0
@pytest.mark.Linear_SVM_train
def test_Linear_SVM_train_test_2d2s():
    m = Linear_SVM(p=2,lr = 0.01,n_epoch=1000)
    X  = np.array([[0., 0.], # first training sample x1
                   [1., 1.]])# second training sample x2
    y = np.array([-1., 1.])
    m.train(X, y)
    assert np.allclose(m.w[0]+m.w[1]+ m.b, 1.,atol = 0.1)  # x2 is a positive support vector 
    assert np.allclose(m.b, -1.,atol =0.1)  # x1 is a negative support vector 
# --------------------------------

# ---- Test Case: test_largeC (Method: train, Class: Linear_SVM) 
# Goal: test when there are 4 data samples, 2 features and a large C value. After training, the parameters w and b of the model should be set as the optimal parameter values based on the training samples
# Points: 4.0
@pytest.mark.Linear_SVM_train
def test_Linear_SVM_train_test_largeC():
    m = Linear_SVM(p=2,C=10000., lr = 0.01,n_epoch=1000)
    X  = np.array([[0., 1.],
                   [1., 0.],
                   [2., 0.],
                   [0., 2.]])
    y = np.array([-1., -1., 1., 1.])
    m.train(X, y)
    assert np.allclose(m.w[0]+m.b, -1, atol = 0.1)
    assert np.allclose(m.w[1]+m.b, -1, atol = 0.1)
    assert np.allclose(m.w[0]+m.w[1]+m.b, 1, atol = 0.1)
# --------------------------------

# ---- Test Case: test_smallC (Method: train, Class: Linear_SVM) 
# Goal: test when there are 4 data samples, 2 features and a small C value. After training, the parameters w and b of the model should be set as the optimal parameter values based on the training samples
# Points: 4.0
@pytest.mark.Linear_SVM_train
def test_Linear_SVM_train_test_smallC():
    m = Linear_SVM(p=2,C=0.01, lr = 0.01,n_epoch=1000)
    m.w=np.array([1.,2.])
    X  = np.array([[0., 1.],
                   [1., 0.],
                   [2., 0.],
                   [0., 2.]])
    y = np.array([-1., -1., 1., 1.])
    m.train(X, y)
    assert np.allclose(m.w, [0,0], atol = 0.1)
# --------------------------------

# ---- Test Case: test_dataset (Method: train, Class: Linear_SVM) 
# Goal: test with a dataset from files X.csv, Y.csv. After training, the parameters w and b of the model should be set as the optimal parameter values based on the training samples
# Points: 8.0
@pytest.mark.Linear_SVM_train
def test_Linear_SVM_train_test_dataset():
    m = Linear_SVM(p=2,C=1000., lr = 0.001,n_epoch=500)
    # load a binary classification dataset
    X=np.loadtxt("X.csv",dtype=float, delimiter=",")
    y=np.loadtxt("y.csv",dtype=int, delimiter=",")
    # split the dataset into a training set (100 samples) and a test set (100 samples)
    Xtrain, Ytrain, Xtest, Ytest = X[::2], y[::2], X[1::2], y[1::2]
    # train SVM 
    m.train(Xtrain, Ytrain)
    # accuracy on training samples
    Y = m.predict(Xtrain)
    accuracy = (Y == Ytrain).sum()/100.
    print("Training accuracy:", accuracy)
    assert accuracy > 0.9
    # accuracy samples
    Y = m.predict(Xtest)
    accuracy = (Y == Ytest).sum()/100.
    print("Test accuracy:", accuracy)
    assert accuracy > 0.9
# --------------------------------


