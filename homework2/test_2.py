from problem2 import *
import sys
import pytest 
import importlib.metadata 

import numpy as np
import warnings
'''
    Unit test 2  (Total Points: 30)
    This file includes unit tests for problem2.py.
'''

def test_python_environment():
    assert sys.version_info[0]==3 # require python 3.11 or above 
    assert sys.version_info[1]>=11
    


# ============= Class: Logistic_Regression ===========================
# Total Points: 30.0

# ***********  Method: compute_z  (Class: Logistic_Regression) **************** 
# Total Points: 1.5

# ---- Test Case: test_2d (Method: compute_z, Class: Logistic_Regression) 
# Goal: test a simple example with 2 dimensional features
# Points: 1.5
@pytest.mark.Logistic_Regression_compute_z
def test_Logistic_Regression_compute_z_test_2d():
    m = Logistic_Regression(p=2)
    m.w = np.array([0.5, -0.6])
    m.b = 0.2
    x = np.array([1., 2.])
    z = m.compute_z(x)
    assert np.allclose(z, -0.5, atol = 1e-3) 
    m.w = np.array([-0.5, 0.6])
    z = m.compute_z(x)
    assert np.allclose(z, .9, atol = 1e-3) 
    m.w = np.array([0.5,-0.6])
    x = np.array([ 2., 5. ])
    z = m.compute_z(x)
    assert np.allclose(z, -1.8, atol = 1e-3) 
    m.b = 0.5
    z = m.compute_z(x)
    assert np.allclose(z, -1.5, atol = 1e-3) 
# --------------------------------


# ***********  Method: compute_dz_db  (Class: Logistic_Regression) **************** 
# Total Points: 1.5

# ---- Test Case: test_random (Method: compute_dz_db, Class: Logistic_Regression) 
# Goal: test with random data using gradient checking
# Points: 1.5
@pytest.mark.Logistic_Regression_compute_dz_db
def test_Logistic_Regression_compute_dz_db_test_random():
    for _ in range(20):
        p = np.random.randint(2,20)
        x = np.random.random(p)
        m = Logistic_Regression(p=2)
        m.w = np.random.random(p)
        m.b = np.random.random(1)
        # analytical gradients
        db = m.compute_dz_db()
        # numerical gradients
        db_true = m.check_dz_db(x)
        assert np.allclose(db, db_true, atol=1e-2) 
# --------------------------------



# ***********  Method: compute_dz_dw  (Class: Logistic_Regression) **************** 
# Total Points: 1.5

# ---- Test Case: test_random (Method: compute_dz_dw, Class: Logistic_Regression) 
# Goal: test with random inputs
# Points: 1.5
@pytest.mark.Logistic_Regression_compute_dz_dw
def test_Logistic_Regression_compute_dz_dw_test_random():
    for _ in range(10):
        p = np.random.randint(2,20)
        m = Logistic_Regression(p=p)
        m.w = 2*np.random.random(p)-1
        m.b = 2*np.random.random(1)[0]-1
        x = 2*np.random.random(p)-1
        # analytical gradients
        dw = m.compute_dz_dw(x)
        # numerical gradients
        dw_true = m.check_dz_dw(x)
        assert np.allclose(dw, dw_true, atol=1e-2) 
# --------------------------------



# ***********  Method: compute_a  (Class: Logistic_Regression) **************** 
# Total Points: 3.0

# ---- Test Case: test_toy (Method: compute_a, Class: Logistic_Regression) 
# Goal: test with simple inputs
# Points: 1.5
@pytest.mark.Logistic_Regression_compute_a
def test_Logistic_Regression_compute_a_test_toy():
    m = Logistic_Regression(p=2)
    a =m.compute_a(0.)
    assert np.allclose(a, 0.5, atol = 1e-2) 
    a =m.compute_a(1.)
    assert np.allclose(a, 0.73105857863, atol = 1e-2) 
    a = m.compute_a(-1.)
    assert np.allclose(a, 0.26894142137, atol = 1e-2) 
    a = m.compute_a(-2.)
    assert np.allclose(a, 0.1192029, atol = 1e-2) 
    a =m.compute_a(-50.)
    assert np.allclose(a, 0, atol = 1e-2) 
    a =m.compute_a(50.)
    assert np.allclose(a, 1, atol = 1e-2) 
# --------------------------------

# ---- Test Case: test_small_z (Method: compute_a, Class: Logistic_Regression) 
# Goal: test with small z values, such as z =-1000, try to avoid computing exp(z)
# Points: 1.5
@pytest.mark.Logistic_Regression_compute_a
def test_Logistic_Regression_compute_a_test_small_z():
    m = Logistic_Regression(p=2)
    z = -710.
    a =m.compute_a(z)
    assert np.allclose(a, 0, atol = 1e-2) 
    z = 710.
    a =m.compute_a(z)
    assert np.allclose(a, 1, atol = 1e-2) 
# --------------------------------


# ***********  Method: compute_da_dz  (Class: Logistic_Regression) **************** 
# Total Points: 1.5

# ---- Test Case: test_toy (Method: compute_da_dz, Class: Logistic_Regression) 
# Goal: test with sample examples
# Points: 0.75
@pytest.mark.Logistic_Regression_compute_da_dz
def test_Logistic_Regression_compute_da_dz_test_toy():
    m = Logistic_Regression(p=2)
    a  = 0.5 
    da_dz = m.compute_da_dz(a)
    assert np.allclose(da_dz, 0.25, atol= 1e-3)
    a  = 0.3 
    da_dz = m.compute_da_dz(a)
    assert np.allclose(da_dz, 0.21, atol= 1e-3)
    a  = 0.9 
    da_dz = m.compute_da_dz(a)
    assert np.allclose(da_dz, 0.09, atol= 1e-3)
    a  = 0.
    da_dz = m.compute_da_dz(a)
    assert np.allclose(da_dz, 0, atol= 1e-4)
    a  = 1.
    da_dz = m.compute_da_dz(a)
    assert np.allclose(da_dz, 0, atol= 1e-4)
# --------------------------------

# ---- Test Case: test_rand (Method: compute_da_dz, Class: Logistic_Regression) 
# Goal: test with random examples
# Points: 0.75
@pytest.mark.Logistic_Regression_compute_da_dz
def test_Logistic_Regression_compute_da_dz_test_rand():
    m = Logistic_Regression(p=2)
    for _ in range(20):
        z = 2000*np.random.random(1)-1000
        a = m.compute_a(z)
        # analytical gradients
        da_dz = m.compute_da_dz(a)
        # numerical gradients
        da_dz_true = m.check_da_dz(z)
        assert np.allclose(da_dz, da_dz_true, atol=1e-4) 
# --------------------------------



# ***********  Method: compute_L  (Class: Logistic_Regression) **************** 
# Total Points: 3.0

# ---- Test Case: test_toy (Method: compute_L, Class: Logistic_Regression) 
# Goal: test with simple inputs
# Points: 1.2
@pytest.mark.Logistic_Regression_compute_L
def test_Logistic_Regression_compute_L_test_toy():
    m = Logistic_Regression(p=2)
    L= m.compute_L(z=0.,y=0.)
    assert np.allclose(L, np.log(2), atol = 1e-3) 
    L= m.compute_L(z=0.,y=1)
    assert np.allclose(L, np.log(2), atol = 1e-3) 
    warnings.filterwarnings("error")
# --------------------------------

# ---- Test Case: test_large_z (Method: compute_L, Class: Logistic_Regression) 
# Goal: test with large z values as the input
# Points: 0.9
@pytest.mark.Logistic_Regression_compute_L
def test_Logistic_Regression_compute_L_test_large_z():
    m = Logistic_Regression(p=2)
    L= m.compute_L(1000.,0)
    assert np.allclose(L, 1000., atol = 1e-1) 
    L= m.compute_L(2000.,0)
    assert np.allclose(L, 2000., atol = 1e-1) 
    L= m.compute_L(1000.,1)
    assert np.allclose(L, 0., atol = 1e-1) 
    L= m.compute_L(2000.,1)
    assert np.allclose(L, 0., atol = 1e-1) 
# --------------------------------

# ---- Test Case: test_negative_z (Method: compute_L, Class: Logistic_Regression) 
# Goal: test with large negative z values as the input
# Points: 0.9
@pytest.mark.Logistic_Regression_compute_L
def test_Logistic_Regression_compute_L_test_negative_z():
    m = Logistic_Regression(p=2)
    L= m.compute_L(-1000.,0)
    assert np.allclose(L, 0., atol = 1e-1) 
    L= m.compute_L(-2000.,0)
    assert np.allclose(L, 0., atol = 1e-1) 
    L= m.compute_L(-1000.,1)
    assert np.allclose(L, 1000., atol = 1e-1) 
    L= m.compute_L(-2000.,1)
    assert np.allclose(L, 2000., atol = 1e-1) 
# --------------------------------


# ***********  Method: compute_dL_dz  (Class: Logistic_Regression) **************** 
# Total Points: 3.0

# ---- Test Case: test_toy (Method: compute_dL_dz, Class: Logistic_Regression) 
# Goal: test with sample examples
# Points: 0.6
@pytest.mark.Logistic_Regression_compute_dL_dz
def test_Logistic_Regression_compute_dL_dz_test_toy():
    m = Logistic_Regression(p=2)
    dL_dz = m.compute_dL_dz(z=0.,y=0.)
    assert np.allclose(dL_dz, 0.5, atol= 1e-3)
    dL_dz = m.compute_dL_dz(z=0,y=1)
    assert np.allclose(dL_dz, -0.5, atol= 1e-3)
# --------------------------------

# ---- Test Case: test_large_z (Method: compute_dL_dz, Class: Logistic_Regression) 
# Goal: test with large z values as the input
# Points: 0.6
@pytest.mark.Logistic_Regression_compute_dL_dz
def test_Logistic_Regression_compute_dL_dz_test_large_z():
    m = Logistic_Regression(p=2)
    dL_dz = m.compute_dL_dz(1000,1)
    assert dL_dz == dL_dz # check if dL_dz is NaN (not a number)
    assert np.allclose(dL_dz, 0., atol= 1e-3)
    dL_dz = m.compute_dL_dz(1000,0)
    assert dL_dz == dL_dz # check if dL_dz is NaN (not a number)
    assert np.allclose(dL_dz, 1., atol= 1e-3)
    warnings.filterwarnings("error")
# --------------------------------

# ---- Test Case: test_negative_z (Method: compute_dL_dz, Class: Logistic_Regression) 
# Goal: test with large negative z values as the input
# Points: 0.6
@pytest.mark.Logistic_Regression_compute_dL_dz
def test_Logistic_Regression_compute_dL_dz_test_negative_z():
    m = Logistic_Regression(p=2)
    dL_dz = m.compute_dL_dz(-1000,0)
    assert np.allclose(dL_dz, 0., atol= 1e-3)
    dL_dz = m.compute_dL_dz(-1000,1)
    assert np.allclose(dL_dz, -1., atol= 1e-3)
# --------------------------------

# ---- Test Case: test_rand (Method: compute_dL_dz, Class: Logistic_Regression) 
# Goal: test with random examples
# Points: 1.2
@pytest.mark.Logistic_Regression_compute_dL_dz
def test_Logistic_Regression_compute_dL_dz_test_rand():
    m = Logistic_Regression(p=2)
    for _ in range(20):
        z = 10*np.random.random(1)[0]-5
        y = np.random.randint(2)
        # analytical gradients
        dz = m.compute_dL_dz(z,y)
        # numerical gradients
        dz_true = m.check_dL_dz(z,y)
        assert np.allclose(dz, dz_true, atol=1e-2) 
# --------------------------------



# ***********  Method: compute_dL_db  (Class: Logistic_Regression) **************** 
# Total Points: 1.5

# ---- Test Case: test_toy (Method: compute_dL_db, Class: Logistic_Regression) 
# Goal: test with sample examples
# Points: 1.5
@pytest.mark.Logistic_Regression_compute_dL_db
def test_Logistic_Regression_compute_dL_db_test_toy():
    m = Logistic_Regression(p=2)
    dL_dz = -2.0 
    dz_db = 1.0 
    dL_db = m.compute_dL_db(dL_dz,dz_db)
    dL_db_true = -2.0
    assert np.allclose(dL_db, dL_db_true, atol = 1e-3)
# --------------------------------


# ***********  Method: compute_dL_dw  (Class: Logistic_Regression) **************** 
# Total Points: 1.5

# ---- Test Case: test_toy (Method: compute_dL_dw, Class: Logistic_Regression) 
# Goal: test with sample examples
# Points: 1.5
@pytest.mark.Logistic_Regression_compute_dL_dw
def test_Logistic_Regression_compute_dL_dw_test_toy():
    m = Logistic_Regression(p=2)
    dL_dz = -1.0
    dz_dw = np.array([1., 2.])
    dL_dw = m.compute_dL_dw(dL_dz, dz_dw)
    dL_dw_true =np.array([-1., -2.])
    assert np.allclose(dL_dw, dL_dw_true, atol = 1e-3)
    dL_dz = 0.5
    dz_dw = np.array([2., 3.])
    dL_dw = m.compute_dL_dw(dL_dz, dz_dw)
    dL_dw_true =np.array([1., 1.5])
    assert np.allclose(dL_dw, dL_dw_true, atol = 1e-3)
# --------------------------------


# ***********  Method: backward  (Class: Logistic_Regression) **************** 
# Total Points: 3.0

# ---- Test Case: test_toy (Method: backward, Class: Logistic_Regression) 
# Goal: test with simple inputs
# Points: 1.2
@pytest.mark.Logistic_Regression_backward
def test_Logistic_Regression_backward_test_toy():
    m = Logistic_Regression(p=2)
    x = np.array([1., 2.])
    y = 1 
    z = 0
    dL_dw, dL_db = m.backward(x,y,z)
    assert np.allclose(dL_dw,[-0.5,-1], atol=1e-3)
    assert np.allclose(dL_db,-0.5, atol=1e-3)
# --------------------------------

# ---- Test Case: test_large_z (Method: backward, Class: Logistic_Regression) 
# Goal: test with large z values as the input
# Points: 1.8
@pytest.mark.Logistic_Regression_backward
def test_Logistic_Regression_backward_test_large_z():
    warnings.filterwarnings("error")
    m = Logistic_Regression(p=2)
    x = np.array([2., 3., 4.])
    y = 1
    z = 1000.
    dL_dw, dL_db = m.backward(x,y,z)
    assert np.allclose(dL_dw,[0,0,0], atol=1e-3)
    assert np.allclose(dL_db,0, atol=1e-3)
    y = 1
    z = -1000.
    dL_dw, dL_db = m.backward(x,y,z)
    assert np.allclose(dL_dw,[-2,-3,-4], atol=1e-3)
    assert np.allclose(dL_db,-1, atol=1e-3)
    y = 0
    z = -1000.
    dL_dw, dL_db = m.backward(x,y,z)
    assert np.allclose(dL_dw,[0,0,0], atol=1e-3)
    assert np.allclose(dL_db,0, atol=1e-3)
    y = 0
    z = 1000.
    dL_dw, dL_db = m.backward(x,y,z)
    assert np.allclose(dL_dw,[2,3,4], atol=1e-3)
    assert np.allclose(dL_db,1, atol=1e-3)
# --------------------------------


# ***********  Method: train  (Class: Logistic_Regression) **************** 
# Total Points: 9.0

# ---- Test Case: test_2d4s (Method: train, Class: Logistic_Regression) 
# Goal: test when there are 4 data samples and 2 features. After training, the parameters w and b of the model should be set as the optimal parameter values based on the training samples
# Points: 2.7
@pytest.mark.Logistic_Regression_train
def test_Logistic_Regression_train_test_2d4s():
    m = Logistic_Regression(p=2,lr = 1.,n_epoch=100)
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

# ---- Test Case: test_6samples (Method: train, Class: Logistic_Regression) 
# Goal: test when there are 6 data samples, 2 features and a different learning rate
# Points: 2.7
@pytest.mark.Logistic_Regression_train
def test_Logistic_Regression_train_test_6samples():
    m = Logistic_Regression(p=2,lr = 0.1,n_epoch=1000)
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

# ---- Test Case: test_datafile (Method: train, Class: Logistic_Regression) 
# Goal: test on the dataset in files X.csv and Y.csv
# Points: 3.6
@pytest.mark.Logistic_Regression_train
def test_Logistic_Regression_train_test_datafile():
    X = np.loadtxt("X.csv",delimiter=",",dtype=float)
    y = np.loadtxt("y.csv",delimiter=",",dtype=int)
    y[y==-1] = 0
    Xtrain, Ytrain, Xtest, Ytest = X[::2], y[::2], X[1::2], y[1::2]
    m = Logistic_Regression(p=2,lr = 0.001,n_epoch=1000)
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


