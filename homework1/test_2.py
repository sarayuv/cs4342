from problem2 import *
import sys
import pytest 
import importlib.metadata 

import numpy as np
'''
    Unit test 2  (Total Points: 20)
    This file includes unit tests for problem2.py.
'''

def test_python_environment():
    assert sys.version_info[0]==3 # require python 3.11 or above 
    assert sys.version_info[1]>=11
    


# ============= Class: Simple_Regression ===========================
# Total Points: 0.0


# ============= Class: Simple_Regression_Abs ===========================
# Total Points: 10.0
# ***********  Method: train  (Class: Simple_Regression_Abs) **************** 
# Total Points: 10.0

# ---- Test Case: test_3instances (Method: train, Class: Simple_Regression_Abs) 
# Goal: test when there are 3 data samples and 2 features. After training, the parameter b of the model should be set as the optimal value based on the training samples
# Points: 3.0
@pytest.mark.Simple_Regression_Abs_train
def test_Simple_Regression_Abs_train_test_3instances():
    X = np.array([[ 1.,-1.],  # the first instance,
                  [ 1., 0.],  # the second instance
                  [ 1., 1.]])
    y = np.array([1.2,5.1,100])
    m = Simple_Regression_Abs() # create the regression model
    m.train(X,y)
    assert m.b==5.1
# --------------------------------

# ---- Test Case: test_5instances (Method: train, Class: Simple_Regression_Abs) 
# Goal: test when there are 5 data samples and 3 features. After training, the parameter b of the model should be set as the optimal value based on the training samples
# Points: 3.0
@pytest.mark.Simple_Regression_Abs_train
def test_Simple_Regression_Abs_train_test_5instances():
    X = np.random.rand(5,3)
    y = np.array([1.2,6.2,-2., 2.5,3.8])
    m = Simple_Regression_Abs() # create the regression model
    m.train(X,y)
    assert m.b==2.5
# --------------------------------

# ---- Test Case: test_7instances (Method: train, Class: Simple_Regression_Abs) 
# Goal: test when there are 7 data samples and 5 features. After training, the parameter b of the model should be set as the optimal value based on the training samples
# Points: 4.0
@pytest.mark.Simple_Regression_Abs_train
def test_Simple_Regression_Abs_train_test_7instances():
    X = np.random.rand(7,5)
    y = np.array([1.2,6.2,-2., 2.9,3.8,1000,-100])
    m = Simple_Regression_Abs() # create the regression model
    m.train(X,y)
    assert m.b==2.9
# --------------------------------

# ============= Class: Simple_Regression_SE ===========================
# Total Points: 10.0
# ***********  Method: train  (Class: Simple_Regression_SE) **************** 
# Total Points: 10.0

# ---- Test Case: test_3instances (Method: train, Class: Simple_Regression_SE) 
# Goal: test when there are 3 data samples and 2 features. After training, the parameter b of the model should be set as the optimal value based on the training samples
# Points: 3.0
@pytest.mark.Simple_Regression_SE_train
def test_Simple_Regression_SE_train_test_3instances():
    X = np.array([[ 1.,-1.],  # the first instance,
                  [ 1., 0.],  # the second instance
                  [ 1., 1.]])
    y = np.array([1.,2.,6.])
    m = Simple_Regression_SE() # create the regression model
    m.train(X,y)
    assert m.b==3.
# --------------------------------

# ---- Test Case: test_5instances (Method: train, Class: Simple_Regression_SE) 
# Goal: test when there are 5 data samples and 3 features. After training, the parameter b of the model should be set as the optimal value based on the training samples
# Points: 3.0
@pytest.mark.Simple_Regression_SE_train
def test_Simple_Regression_SE_train_test_5instances():
    X = np.random.rand(5,3)
    y = np.array([1.2,6.2,-2., 2.5,3.8])
    m = Simple_Regression_SE() # create the regression model
    m.train(X,y)
    assert m.b==2.34
# --------------------------------

# ---- Test Case: test_7instances (Method: train, Class: Simple_Regression_SE) 
# Goal: test when there are 7 data samples and 5 features. After training, the parameter b of the model should be set as the optimal value based on the training samples
# Points: 4.0
@pytest.mark.Simple_Regression_SE_train
def test_Simple_Regression_SE_train_test_7instances():
    X = np.random.rand(7,5)
    y = np.array([1.2,6.2,-2., 2.9,3.8,1000,-100])
    m = Simple_Regression_SE() # create the regression model
    m.train(X,y)
    assert m.b==130.3
# --------------------------------


