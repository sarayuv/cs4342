from problem4 import *
import sys
import pytest 
import importlib.metadata 

import numpy as np
'''
    Unit test 4  (Total Points: 40)
    This file includes unit tests for problem4.py.
'''

def test_python_environment():
    assert sys.version_info[0]==3 # require python 3.11 or above 
    assert sys.version_info[1]>=11
    


# ============= Class: Lasso_Regression ===========================
# Total Points: 40.0

# ***********  Method: compute_gradient  (Class: Lasso_Regression) **************** 
# Total Points: 20.0

# ---- Test Case: test_2d (Method: compute_gradient, Class: Lasso_Regression) 
# Goal: test an simple example with 2 dimensional features
# Points: 4.0
@pytest.mark.Lasso_Regression_compute_gradient
def test_Lasso_Regression_compute_gradient_test_2d():
    x = np.array([ 1.,2.]) 
    y = 1.
    m = Lasso_Regression(p=2,alpha=0.5)
    m.w = np.array([1.,1.])
    dL_dw = m.compute_gradient(x,y)
    assert np.allclose(dL_dw,[2.5, 4.5])
# --------------------------------

# ---- Test Case: test_3d (Method: compute_gradient, Class: Lasso_Regression) 
# Goal: test an simple example with 3 dimensional features
# Points: 4.0
@pytest.mark.Lasso_Regression_compute_gradient
def test_Lasso_Regression_compute_gradient_test_3d():
    x = np.array([ 1.,2.,3.]) 
    y = 4.
    m = Lasso_Regression(p=3,alpha=0.5)
    m.w = np.array([1.,1.,1.])
    dL_dw = m.compute_gradient(x,y)
    assert np.allclose(dL_dw,[2.5, 4.5, 6.5])
# --------------------------------

# ---- Test Case: test_alpha (Method: compute_gradient, Class: Lasso_Regression) 
# Goal: test an simple example with a different alpha value
# Points: 4.0
@pytest.mark.Lasso_Regression_compute_gradient
def test_Lasso_Regression_compute_gradient_test_alpha():
    x = np.array([ 1.,2.,3.]) 
    y = 4.
    m = Lasso_Regression(p=3,alpha=1.)
    m.w = np.array([1.,1.,1.])
    dL_dw = m.compute_gradient(x,y)
    assert np.allclose(dL_dw,[3., 5, 7])
# --------------------------------

# ---- Test Case: test_weight (Method: compute_gradient, Class: Lasso_Regression) 
# Goal: test an simple example with a different weight assignment
# Points: 4.0
@pytest.mark.Lasso_Regression_compute_gradient
def test_Lasso_Regression_compute_gradient_test_weight():
    x = np.array([ 1.,2.,3.]) 
    y = -1.
    m = Lasso_Regression(p=3,alpha=1.)
    m.w = np.array([-2.,0.,1.])
    dL_dw = m.compute_gradient(x,y)
    assert np.allclose(dL_dw,[1., 4, 7])
# --------------------------------

# ---- Test Case: test_label (Method: compute_gradient, Class: Lasso_Regression) 
# Goal: test an simple example with a different label
# Points: 4.0
@pytest.mark.Lasso_Regression_compute_gradient
def test_Lasso_Regression_compute_gradient_test_label():
    x = np.array([ 1.,2.,3.]) 
    y = 2.
    m = Lasso_Regression(p=3,alpha=1.)
    m.w = np.array([-2.,0.,1.])
    dL_dw = m.compute_gradient(x,y)
    assert np.allclose(dL_dw,[-2, -2, -2])
# --------------------------------


# ***********  Method: update_w  (Class: Lasso_Regression) **************** 
# Total Points: 8.0

# ---- Test Case: test_2d (Method: update_w, Class: Lasso_Regression) 
# Goal: test an simple example with 2 dimensional features
# Points: 2.4
@pytest.mark.Lasso_Regression_update_w
def test_Lasso_Regression_update_w_test_2d():
    m = Lasso_Regression(p=2,lr = 0.1)
    m.w = np.array([4.,3.])
    dL_dw = np.array([ 1.,2.]) 
    m.update_w(dL_dw)
    assert np.allclose(m.w,[3.9, 2.8])
# --------------------------------

# ---- Test Case: test_3d (Method: update_w, Class: Lasso_Regression) 
# Goal: test an simple example with 3 dimensional features
# Points: 2.4
@pytest.mark.Lasso_Regression_update_w
def test_Lasso_Regression_update_w_test_3d():
    m = Lasso_Regression(p=3,lr = 0.1)
    m.w = np.array([4.,3.,2.])
    dL_dw = np.array([ 1.,2.,3.]) 
    m.update_w(dL_dw)
    assert np.allclose(m.w,[3.9, 2.8, 1.7])
# --------------------------------

# ---- Test Case: test_lr (Method: update_w, Class: Lasso_Regression) 
# Goal: test an simple example with a different learning rate
# Points: 3.2
@pytest.mark.Lasso_Regression_update_w
def test_Lasso_Regression_update_w_test_lr():
    m = Lasso_Regression(p=3,lr = 0.2)
    m.w = np.array([4.,3.,2.])
    dL_dw = np.array([ 1.,2.,3.]) 
    m.update_w(dL_dw)
    assert np.allclose(m.w,[3.8, 2.6, 1.4])
# --------------------------------


# ***********  Method: train  (Class: Lasso_Regression) **************** 
# Total Points: 12.0

# ---- Test Case: test_alpha0 (Method: train, Class: Lasso_Regression) 
# Goal: test when alpha = 0, the model is reduced to a least square regression model
# Points: 3.6
@pytest.mark.Lasso_Regression_train
def test_Lasso_Regression_train_test_alpha0():
    X = np.array([[ 1.,-1.],  # the first instance,
                  [ 1., 0.],  # the second instance
                  [ 1., 1.]])
    y = np.array([1.5,2.5,3.5])
    m = Lasso_Regression(p=2,alpha=0.,n_epoch=300) # create the regression model
    m.train(X,y)
    assert type(m.w) == np.ndarray
    assert m.w.shape == (2,) 
    assert np.allclose(m.w, [2.5,1.], atol = 1e-2) 
# --------------------------------

# ---- Test Case: test_large_alpha (Method: train, Class: Lasso_Regression) 
# Goal: test when alpha is a large number, the model weights will all shrink to 0
# Points: 4.8
@pytest.mark.Lasso_Regression_train
def test_Lasso_Regression_train_test_large_alpha():
    X = np.array([[ 1.,-1.],  # the first instance,
                  [ 1., 0.],  # the second instance
                  [ 1., 1.]])
    y = np.array([1.5,2.5,3.5])
    m = Lasso_Regression(p=2,alpha=10.,n_epoch=300) # create the regression model
    m.w = np.array([1.,2.]) 
    m.train(X,y)
    assert np.allclose(m.w, [0.,0.], atol = 0.1) 
# --------------------------------

# ---- Test Case: test_medium_alpha (Method: train, Class: Lasso_Regression) 
# Goal: test when alpha is a medium number, only one weight shrinks to 0
# Points: 3.6
@pytest.mark.Lasso_Regression_train
def test_Lasso_Regression_train_test_medium_alpha():
    X = np.array([[ 1.,-1.],  # the first instance,
                  [ 1., 0.],  # the second instance
                  [ 1., 1.]])
    y = np.array([1.5,2.5,3.5])
    m = Lasso_Regression(p=2, alpha=0.8, n_epoch=300) # create the regression model
    m.train(X,y)
    assert m.w[0]>1.5 
    assert m.w[1]<0.1 
# --------------------------------


