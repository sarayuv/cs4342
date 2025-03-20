from problem1 import *
import sys
import pytest 
import importlib.metadata 

import numpy as np
'''
    Unit test 1  (Total Points: 20)
    This file includes unit tests for problem1.py.
'''

def test_python_environment():
    assert sys.version_info[0]==3 # require python 3.11 or above 
    assert sys.version_info[1]>=11
    installed_pkg = {pkg.metadata['Name'] for pkg in importlib.metadata.distributions()}
    assert len(set(['pytest', 'numpy']) - installed_pkg)==0 # Check if any required package is missing. If missing, please follow the instructions at the begining of problem1.py to install the packages


# ============= Class: Nearest_Neighbor ===========================
# Total Points: 20.0

# ***********  Method: find_neighbor  (Class: Nearest_Neighbor) **************** 
# Total Points: 10.0

# ---- Test Case: test_2d (Method: find_neighbor, Class: Nearest_Neighbor) 
# Goal: test when there are 3 training samples and 2 features
# Points: 5.0
@pytest.mark.Nearest_Neighbor_find_neighbor
def test_Nearest_Neighbor_find_neighbor_test_2d():
    X = np.array([[ 1.,  1.],  # the first training sample,
                  [ 1., -1.],  # the second training sample
                  [-1.,  1.]])
    y = np.array([10.,20.,30.]) # the labels of the 3 training samples
    m = Nearest_Neighbor() # create the regression model
    m.train(X,y)
    x_test = np.array([2.,1.])
    y_test = m.find_neighbor(x_test)
    assert y_test ==10.
    x_test = np.array([-0.1,1.])
    y_test = m.find_neighbor(x_test)
    assert y_test ==30.
    x_test = np.array([-0.1,-0.2])
    y_test = m.find_neighbor(x_test)
    assert y_test ==20.
# --------------------------------

# ---- Test Case: test_3d (Method: find_neighbor, Class: Nearest_Neighbor) 
# Goal: test when there are 2 data samples and 3 features
# Points: 5.0
@pytest.mark.Nearest_Neighbor_find_neighbor
def test_Nearest_Neighbor_find_neighbor_test_3d():
    X = np.array([[ 1.,  1., 0.],  # the first instance,
                  [ 1., -1., 0.]]) # the second instance
    y = np.array([10.,20.])
    m = Nearest_Neighbor() # create the regression model
    m.train(X,y)
    x_test = np.array([2.,1.,0.])
    y_test = m.find_neighbor(x_test)
    assert y_test ==10.
    x_test = np.array([-0.1,-0.2,0.])
    y_test = m.find_neighbor(x_test)
    assert y_test ==20.
# --------------------------------


# ***********  Method: predict  (Class: Nearest_Neighbor) **************** 
# Total Points: 10.0

# ---- Test Case: test_2d (Method: predict, Class: Nearest_Neighbor) 
# Goal: test when there are 3 training samples and 2 features
# Points: 5.0
@pytest.mark.Nearest_Neighbor_predict
def test_Nearest_Neighbor_predict_test_2d():
    X = np.array([[ 1.,  1.],  # the first training sample,
                  [ 1., -1.],  # the second training sample
                  [-1.,  1.]])
    y = np.array([10.,20.,30.]) # the labels of the 3 training samples
    m = Nearest_Neighbor() # create the regression model
    m.train(X,y)
    Xt = np.array([[  2.,  1.],  # the first test sample,
                   [-0.1,  1.],  # the second test sample
                   [-0.2,-0.1],  # the third test sample
                   [-0.1,-0.2]]) # the fourth test sample
    yt = m.predict(Xt)
    assert np.allclose(yt, [10,30, 30,20])
# --------------------------------

# ---- Test Case: test_3d (Method: predict, Class: Nearest_Neighbor) 
# Goal: test when there are 3 training samples and 3 features
# Points: 5.0
@pytest.mark.Nearest_Neighbor_predict
def test_Nearest_Neighbor_predict_test_3d():
    X = np.array([[0., 1.,  1.],  # the first training sample,
                  [0., 1., -1.],  # the second training sample
                  [0.,-1.,  1.]])
    y = np.array([10.,20.,30.]) # the labels of the 3 training samples
    m = Nearest_Neighbor() # create the regression model
    m.train(X,y)
    Xt = np.array([[0.,  2.,  1.],  # the first test sample,
                   [0.,-0.1,-0.2]]) # the second test sample
    yt = m.predict(Xt)
    assert np.allclose(yt, [10,20])
# --------------------------------


