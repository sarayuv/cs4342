from problem3 import *
import sys
import pytest 
import importlib.metadata 

import numpy as np
import warnings
'''
    Unit test 3  (Total Points: 30)
    This file includes unit tests for problem3.py.
'''

def test_python_environment():
    assert sys.version_info[0]==3 # require python 3.11 or above 
    assert sys.version_info[1]>=11
    


# ============= Class: Mini_Batch_SGD ===========================
# Total Points: 0.0


# ============= Class: Logistic_Regression_Batch ===========================
# Total Points: 30.0

# ***********  Method: compute_z  (Class: Logistic_Regression_Batch) **************** 
# Total Points: 1.5

# ---- Test Case: test_2d (Method: compute_z, Class: Logistic_Regression_Batch) 
# Goal: test a simple example with 2 dimensional features
# Points: 1.5
@pytest.mark.Logistic_Regression_Batch_compute_z
def test_Logistic_Regression_Batch_compute_z_test_2d():
    m = Logistic_Regression_Batch(p=2,batch_size=3)
    m.w = np.array([0.5, -0.6])
    m.b = 0.2
    x = np.array([[1., 2.], # the first training samples
                  [3., 4.], # the second training sample
                  [5., 6.]])# the third training sample
    z = m.compute_z(x)
    assert np.allclose(z, [-0.5, -0.7,-0.9], atol = 1e-3) 
    m.w = np.array([-0.5, 0.6])
    z = m.compute_z(x)
    assert np.allclose(z, [0.9,1.1,1.3], atol = 1e-3) 
# --------------------------------


# ***********  Method: compute_dz_db  (Class: Logistic_Regression_Batch) **************** 
# Total Points: 1.5

# ---- Test Case: test_1 (Method: compute_dz_db, Class: Logistic_Regression_Batch) 
# Goal: test with a simple example
# Points: 1.5
@pytest.mark.Logistic_Regression_Batch_compute_dz_db
def test_Logistic_Regression_Batch_compute_dz_db_test_1():
    m = Logistic_Regression_Batch(p=2,batch_size = 2)
    dz_db = m.compute_dz_db()
    assert np.allclose(dz_db, [1,1], atol=1e-2) 
    m = Logistic_Regression_Batch(p=2,batch_size = 3)
    dz_db = m.compute_dz_db()
    assert np.allclose(dz_db, [1,1,1], atol=1e-2) 
# --------------------------------


# ***********  Method: compute_dz_dw  (Class: Logistic_Regression_Batch) **************** 
# Total Points: 1.5

# ---- Test Case: test_1 (Method: compute_dz_dw, Class: Logistic_Regression_Batch) 
# Goal: test with a simple example
# Points: 1.5
@pytest.mark.Logistic_Regression_Batch_compute_dz_dw
def test_Logistic_Regression_Batch_compute_dz_dw_test_1():
    m = Logistic_Regression_Batch(p=2,batch_size=3)
    x=np.array([[1.,2.],[3,4],[5,6]])
    dz_dw=m.compute_dz_dw(x)
    assert np.allclose(dz_dw, x, atol=1e-2) 
# --------------------------------


# ***********  Method: compute_a  (Class: Logistic_Regression_Batch) **************** 
# Total Points: 3.0

# ---- Test Case: test_toy (Method: compute_a, Class: Logistic_Regression_Batch) 
# Goal: test with simple inputs
# Points: 3.0
@pytest.mark.Logistic_Regression_Batch_compute_a
def test_Logistic_Regression_Batch_compute_a_test_toy():
    m = Logistic_Regression_Batch(p=2,batch_size=8)
    z =np.array([0.,1,-1,-2,-50,50,-710,710])
    a =m.compute_a(z)
    assert np.allclose(a, [0.5,0.731,.269,.119,0,1,0,1], atol = 1e-2) 
# --------------------------------


# ***********  Method: compute_da_dz  (Class: Logistic_Regression_Batch) **************** 
# Total Points: 1.5

# ---- Test Case: test_toy (Method: compute_da_dz, Class: Logistic_Regression_Batch) 
# Goal: test with sample examples
# Points: 1.5
@pytest.mark.Logistic_Regression_Batch_compute_da_dz
def test_Logistic_Regression_Batch_compute_da_dz_test_toy():
    m = Logistic_Regression_Batch(p=2,batch_size = 5)
    a  = np.array([0.5,0.3,0.9,0,1])
    da_dz = m.compute_da_dz(a)
    assert np.allclose(da_dz, [0.25,0.21,0.09,0,0], atol= 1e-3)
# --------------------------------


# ***********  Method: compute_L  (Class: Logistic_Regression_Batch) **************** 
# Total Points: 3.0

# ---- Test Case: test_toy (Method: compute_L, Class: Logistic_Regression_Batch) 
# Goal: test with simple inputs
# Points: 3.0
@pytest.mark.Logistic_Regression_Batch_compute_L
def test_Logistic_Regression_Batch_compute_L_test_toy():
    m = Logistic_Regression_Batch(p=2, batch_size=10)
    z= np.array([0.,0,1000,2000,1000,2000,-1000,-2000,-1000,-2000])
    y= np.array([0,1,0,0,1,1,0,0,1,1])
    L= m.compute_L(z,y)
    L_true = [np.log(2), np.log(2),1000,2000,0,0,0,0,1000,2000]
    assert np.allclose(L, L_true, atol = 1e-3) 
    warnings.filterwarnings("error")
# --------------------------------


# ***********  Method: compute_dL_dz  (Class: Logistic_Regression_Batch) **************** 
# Total Points: 3.0

# ---- Test Case: test_toy (Method: compute_dL_dz, Class: Logistic_Regression_Batch) 
# Goal: test with sample examples
# Points: 3.0
@pytest.mark.Logistic_Regression_Batch_compute_dL_dz
def test_Logistic_Regression_Batch_compute_dL_dz_test_toy():
    m = Logistic_Regression_Batch(p=2,batch_size=6)
    z= np.array([0.,0,1000,1000,-1000,-1000])
    y= np.array([0,1,1,0,0,1])
    dL_dz = m.compute_dL_dz(z,y)
    dL_dz_true = [0.5,-0.5,0,1,0,-1]
    assert np.allclose(dL_dz, dL_dz_true, atol= 1e-3)
# --------------------------------


# ***********  Method: compute_dL_db  (Class: Logistic_Regression_Batch) **************** 
# Total Points: 1.5

# ---- Test Case: test_toy (Method: compute_dL_db, Class: Logistic_Regression_Batch) 
# Goal: test with sample examples
# Points: 1.5
@pytest.mark.Logistic_Regression_Batch_compute_dL_db
def test_Logistic_Regression_Batch_compute_dL_db_test_toy():
    m = Logistic_Regression_Batch(p=2,batch_size=3)
    dL_dz = np.array([1.,2.,3.]) 
    dz_db = np.array([1.,1,1]) 
    dL_db = m.compute_dL_db(dL_dz,dz_db)
    assert np.allclose(dL_db, [1,2,3], atol = 1e-3)
# --------------------------------


# ***********  Method: compute_dL_dw  (Class: Logistic_Regression_Batch) **************** 
# Total Points: 1.5

# ---- Test Case: test_toy (Method: compute_dL_dw, Class: Logistic_Regression_Batch) 
# Goal: test with sample examples
# Points: 1.5
@pytest.mark.Logistic_Regression_Batch_compute_dL_dw
def test_Logistic_Regression_Batch_compute_dL_dw_test_toy():
    m = Logistic_Regression_Batch(p=2,batch_size = 2)
    dL_dz = np.array([-1.0, 0.5])
    dz_dw = np.array([[1., 2.,3.],[4,5,6]])
    dL_dw = m.compute_dL_dw(dL_dz, dz_dw)
    dL_dw_true =np.array([[-1., -2.,-3],[2.,2.5,3]])
    assert np.allclose(dL_dw, dL_dw_true, atol = 1e-3)
# --------------------------------


# ***********  Method: backward  (Class: Logistic_Regression_Batch) **************** 
# Total Points: 3.0

# ---- Test Case: test_toy (Method: backward, Class: Logistic_Regression_Batch) 
# Goal: test with simple inputs
# Points: 3.0
@pytest.mark.Logistic_Regression_Batch_backward
def test_Logistic_Regression_Batch_backward_test_toy():
    m = Logistic_Regression_Batch(p=2,batch_size = 3)
    x = np.array([[1., 2.],[3,4],[5,6]])
    y = np.array([1,0,1]) 
    z = np.array([0.,1000,-1000])
    dL_dw, dL_db = m.backward(x,y,z)
    dL_db_true = [-0.5,1,-1]
    dL_dw_true = [[-0.5,-1. ],[ 3. , 4. ],[-5., -6. ]]
    assert np.allclose(dL_db,dL_db_true, atol=1e-3)
    assert np.allclose(dL_dw,dL_dw_true, atol=1e-3)
# --------------------------------


# ***********  Method: train  (Class: Logistic_Regression_Batch) **************** 
# Total Points: 9.0

# ---- Test Case: test_2d4s (Method: train, Class: Logistic_Regression_Batch) 
# Goal: test when there are 4 data samples and 2 features. After training, the parameters w and b of the model should be set as the optimal parameter values based on the training samples
# Points: 2.7
@pytest.mark.Logistic_Regression_Batch_train
def test_Logistic_Regression_Batch_train_test_2d4s():
    m = Logistic_Regression_Batch(p=2,lr = 1.,n_epoch=100,batch_size=2)
    m.w = np.array([1.,2.])
    X = np.array([[0., 1.], # an example feature matrix (4 instances, 2 features)
                  [1., 0.],
                  [0., 0.],
                  [1., 1.]])
    y = np.array([0, 1, 0, 1])
    m.train(X, y)
    assert m.w[1] + m.b <= 0 # x1 is negative 
    assert m.w[0] + m.b >= 0 # x2 is positive
    assert  m.b <= 0 # x3 is negative 
    assert m.w[0]+m.w[1] + m.b >= 0 # x4 is positive
# --------------------------------

# ---- Test Case: test_6samples (Method: train, Class: Logistic_Regression_Batch) 
# Goal: test when there are 6 data samples, 2 features and a different learning rate
# Points: 2.7
@pytest.mark.Logistic_Regression_Batch_train
def test_Logistic_Regression_Batch_train_test_6samples():
    m = Logistic_Regression_Batch(p=2,lr = 0.1,n_epoch=1000,batch_size=3)
    m.w = np.array([1.,2.])
    X = np.array([[0., 1.],
                  [1., 0.],
                  [0., 0.],
                  [2., 0.],
                  [0., 2.],
                  [1., 1.]])
    Y = np.array([0, 0, 0, 1, 1, 1])
    m.train(X, Y)
    assert m.w[0]+m.w[1] + m.b >= 0 
    assert 2*m.w[0] + m.b >= 0 
    assert 2*m.w[1] + m.b >= 0 
    assert m.w[0] + m.b <= 0 
    assert m.w[1] + m.b <= 0 
    assert  m.b <= 0 
# --------------------------------

# ---- Test Case: test_datafile (Method: train, Class: Logistic_Regression_Batch) 
# Goal: test on the dataset in files X.csv and Y.csv
# Points: 3.6
@pytest.mark.Logistic_Regression_Batch_train
def test_Logistic_Regression_Batch_train_test_datafile():
    X = np.loadtxt("X.csv",delimiter=",",dtype=float)
    y = np.loadtxt("y.csv",delimiter=",",dtype=int)
    y[y==-1] = 0
    Xtrain, Ytrain, Xtest, Ytest = X[::2], y[::2], X[1::2], y[1::2]
    m = Logistic_Regression_Batch(p=2,lr = 0.001,n_epoch=1000,batch_size=10)
    m.train(Xtrain, Ytrain)
    Y = m.predict(Xtrain)
    accuracy = sum(Y == Ytrain)/100
    print("Training accuracy:", accuracy)
    assert accuracy > 0.9
    Y = m.predict(Xtest)
    accuracy = sum(Y == Ytest)/100
    print("Test accuracy:", accuracy)
    assert accuracy > 0.9
# --------------------------------


