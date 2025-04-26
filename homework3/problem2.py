




#------------ No New Package --------------
# NOTE: Please don't import any new package. You should be able to solve the problems using only the package(s) imported here.
from problem1 import Multiclass_Classification, SGD
import numpy as np
#---------------------------------------------------------


# ---------------------------------------------------------
'''
    Goal of Problem 2: Non-linear Methods for Multi-Class Classification (Fully-Connected Neural Network) (30 points)
    In this problem, you will implement a multi-class classification method using fully-connected neural network with two layers. The main goal of this problem is to extend the softmax regression method to multiple layers. Here we focus on two layered neural networks. In the first layer, we will use sigmoid function as the activation function to convert the linear logits (z1) into a non-linear activations (a1). In the second layer, we will use softmax as the activation function. We will use multi-class cross entropy as the loss function and stochastic gradient descent to train the model parameters..
    
'''
# ---------------------------------------------------------

'''------------- Class: Fully_Connected_NN (30.0 points) -------
    Two layer fully-connected neural network model.  In the first layer, we will use sigmoid function as the activation function to convert the linear logits (z1) into a non-linear activations (a1).  In the second layer, we will use softmax as the activation function.  
'''
''' ---- Class Properties ----
    * p: the number of input features, an integer scalar.
    * c: the number of classes in the classification task, an integer scalar.
    * n_epoch: an integer scalar, the number of passes to iterate through all training examples during stochastic gradient descent..
    * lr: the learning rate of the stochastic gradient descent algorithm, a float scalar, controlling the speed of gradient descent.
    * h: the number of outputs in the 1st layer (or the number of hidden neurons in the first layer).
    * W1: the weight matrix of the 1st layer, a float numpy matrix of shape (h by p).
    * b1: the bias values of the 1st layer, a float numpy vector of length h.
    * W2: the weight matrix of the 2nd layer, a float numpy matrix of shape (c by h).
    * b2: the bias values of the 2nd layer, a float numpy vector of length c.
    '''
class Fully_Connected_NN(Multiclass_Classification,SGD):
    #------------- Method: __init__  ------
    ''' Goal: Initialize the model object    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * p: the number of features/measurements in each data sample, an integer scaler
    * c: the number of classes in the classification task, an integer scalar
    * h: the number of outputs in the 1st layer (or the number of hidden neurons in the first layer)
    * n_epoch: an integer scalar, the number of passes to iterate through all training examples during stochastic gradient descent.
    * lr: also called alpha, the learning rate of the stochastic gradient descent algorithm, a float scalar, controlling the speed of gradient descent. Note in the slides we use the notation 'alpha' to refer to this value
    '''
    def __init__(self, p, c, h, n_epoch=100, lr=0.001):
        Multiclass_Classification.__init__(self,p=p,c=c)
        SGD.__init__(self,n_epoch=n_epoch,lr=lr)
        self.h = h
        self.W1 = np.random.randn(h,p) # initialize W randomly using standard normal distribution
        self.b1 = np.zeros(h) # initialize b as all zeros
        self.W2 = np.random.randn(c,h) # initialize W randomly using standard normal distribution
        self.b2 = np.zeros(c) # initialize b as all zeros
        
        
    #----------------------------------------------------------
    
    #------------- Method: compute_z1  ------
    ''' Goal: (Value Forward Function 1) Given a softmax regression model with parameters W and b, please compute the linear logit z(x) on a data sample x    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: the feature vector of one data sample, a numpy vector of length p
    ---- Outputs: --------
    * z: the linear logit values on the data sample x, i.e. z(x), a float numpy vector of length c
    ---- Hints: --------
    * You could access the model parameters W and b through self variable in the input of this function.. 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_z1(self, x):
        ##############################
        ## INSERT YOUR CODE HERE (0.9 points)
        z = np.dot(self.W1, x) + self.b1
        ##############################
        return z
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Fully_Connected_NN_compute_z1
        (Mac /Linux): python3 -m pytest -v test_2.py -m Fully_Connected_NN_compute_z1
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_dz1_db1  ------
    ''' Goal: (Local Gradient 1.1.1) Suppose we are given a fully-connected neural network with parameters W1, b1, W2 and b2. Suppose we have already computed the linear logits z1(x) in the first layer on a training sample x. Please compute the partial gradient of the linear logits z1(x) in the first layer w.r.t. the biases b1 in the first layer    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    ---- Outputs: --------
    * dz1_db1: the partial gradient of the logits z1 w.r.t. the biases b1, a float matrix of shape (h, h).  Each (i,j)-th element represents the partial gradient of the i-th logit z1[i] w.r.t. the j-th bias b1[i]:  d_z1[i] / d_b1[j]
    ---- Hints: --------
    * Why there is no input variable in this function? Maybe you don't need one, because dz_db is a constant matrix. You don't need any input to compute this gradient. 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_dz1_db1(self):
        ##############################
        ## INSERT YOUR CODE HERE (0.6 points)
        dz1_db1 = np.eye(self.h)
        ##############################
        return dz1_db1
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Fully_Connected_NN_compute_dz1_db1
        (Mac /Linux): python3 -m pytest -v test_2.py -m Fully_Connected_NN_compute_dz1_db1
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_dL_db1  ------
    ''' Goal: (Global Gradient 1.1.1) Suppose we are given a fully-connected neural network with parameters W1, b1, W2 and b2. Suppose we have already computed the global gradients of the loss L w.r.t. the linear logits z1(x) in the first layer and the local gradient of the linear logits z1(x) w.r.t. the biases b1 on a training sample x. Please compute the partial gradient of the loss L w.r.t. the biases b1 in the first layer using chain rule    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * dL_dz1: the partial gradients of the loss function L w.r.t. the linear logits z1, a float numpy vector of length c.  The i-th element dL_dz1[i] represents the partial gradient of the loss function L w.r.t. the i-th linear logit z1[i],  d_L / d_z1[i]
    * dz1_db1: the partial gradient of the logits z1 w.r.t. the biases b1, a float matrix of shape (h, h).  Each (i,j)-th element represents the partial gradient of the i-th logit z1[i] w.r.t. the j-th bias b1[i]:  d_z1[i] / d_b1[j]
    ---- Outputs: --------
    * dL_db1: the partial gradients of the loss function L w.r.t. the biases b1, a float numpy vector of length h.  The i-th element represents the partial gradient of the loss function L w.r.t. the i-th bias b2[i]:  d_L / d_b2[i]
    ---- Hints: --------
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_dL_db1(self, dL_dz1, dz1_db1):
        ##############################
        ## INSERT YOUR CODE HERE (0.6 points)
        dL_db1 = np.dot(dL_dz1, dz1_db1)
        ##############################
        return dL_db1
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Fully_Connected_NN_compute_dL_db1
        (Mac /Linux): python3 -m pytest -v test_2.py -m Fully_Connected_NN_compute_dL_db1
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_dz1_dW1  ------
    ''' Goal: (Local Gradient 1.1.2) Suppose we are given a fully-connected neural network with parameters W1, b1, W2 and b2. Suppose we have already computed the linear logits z1(x) in the first layer on a training sample x. Please compute the partial gradients of the linear logits z1(x) in the first layer w.r.t. the weights W1 in the 1st layer    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: the feature vector of a data sample, a float numpy vector of length p
    ---- Outputs: --------
    * dz1_dW1: the partial gradient of logits z1 w.r.t. the weight matrix W1, a numpy float tensor of shape (h by h by p).  The (i,j,k)-th element represents the partial gradient of the i-th logit (z1[i]) w.r.t. the weight W1[j,k]:   d_z1[i] / d_W1[j,k]
    ---- Hints: --------
    * This problem can be solved using only 3 line(s) of code. More lines are okay.'''
    def compute_dz1_dW1(self, x):
        ##############################
        ## INSERT YOUR CODE HERE (0.9 points)
        dz1_dW1 = np.zeros((self.h, self.h, self.p))
        for i in range(self.h):
            for j in range(self.h):
                if i == j:
                    dz1_dW1[i, j, :] = x
        ##############################
        return dz1_dW1
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Fully_Connected_NN_compute_dz1_dW1
        (Mac /Linux): python3 -m pytest -v test_2.py -m Fully_Connected_NN_compute_dz1_dW1
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_dL_dW1  ------
    ''' Goal: (Global Gradient 1.1.2) Suppose we are given a fully-connected neural network with parameters W1, b1, W2 and b2. Suppose we have already computed the global gradients of the loss L w.r.t. the linear logits z1(x) in the first layer and the local gradient of the linear logits z1(x) w.r.t. the weights W1 on a training sample x. Please compute the partial gradient of the loss L w.r.t. the weights W1 in the first layer using chain rule    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * dL_dz1: the partial gradients of the loss function L w.r.t. the linear logits z1, a float numpy vector of length c.  The i-th element dL_dz1[i] represents the partial gradient of the loss function L w.r.t. the i-th linear logit z1[i],  d_L / d_z1[i]
    * dz1_dW1: the partial gradient of logits z1 w.r.t. the weight matrix W1, a numpy float tensor of shape (h by h by p).  The (i,j,k)-th element represents the partial gradient of the i-th logit (z1[i]) w.r.t. the weight W1[j,k]:   d_z1[i] / d_W1[j,k]
    ---- Outputs: --------
    * dL_dW1: the partial gradients of the loss function L w.r.t. the weights W1, a float numpy matrix of shape (h by p).  The i,j-th element represents the partial gradient of the loss function L w.r.t. the i,j-th weight W1[i,j]:  d_L / d_W1[i,j]
    ---- Hints: --------
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_dL_dW1(self, dL_dz1, dz1_dW1):
        ##############################
        ## INSERT YOUR CODE HERE (0.9 points)
        dL_dW1 = np.einsum('i,ijk->jk', dL_dz1, dz1_dW1)
        ##############################
        return dL_dW1
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Fully_Connected_NN_compute_dL_dW1
        (Mac /Linux): python3 -m pytest -v test_2.py -m Fully_Connected_NN_compute_dL_dW1
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_a1  ------
    ''' Goal: (Value Forward Function 1.2) Suppose we are given a fully-connected neural network and we have already computed the linear logits z1(x) in the first layer on a data sample x. Please compute the element-wise sigmoid activations a1(x) in the first layer on the data sample. Here we use element-wise sigmoid function to transform linear logits z1(x) into activations a1(x)    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * z1: the linear logits of the 1st layer, a float numpy vector of length h
    ---- Outputs: --------
    * a1: the element-wise sigmoid activations in the 1st layer, a float numpy vector of length h.  The i-th element represents the sigmoid of the i-th logit z1[i]
    ---- Hints: --------
    * In this function, we want to compute the element-wise sigmoid: computing sigmoid on each element of z1 and put them into a vector a1, so that a1[i] = sigmoid(z1[i]). 
    * This function is slightly different from the sigmoid function in logistic regression. In logistic regression, the input z is a scalar, but here the input z1 is a vector and here we need to compute the sigmoid of every element of the vector z. 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_a1(self, z1):
        ##############################
        ## INSERT YOUR CODE HERE (0.6 points)
        a1 = 1 / (1 + np.exp(-np.clip(z1, -500, 500)))
        ##############################
        return a1
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Fully_Connected_NN_compute_a1
        (Mac /Linux): python3 -m pytest -v test_2.py -m Fully_Connected_NN_compute_a1
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_da1_dz1  ------
    ''' Goal: (Local Gradient 1.2) Suppose we are given a fully-connected neural network with parameters W1, b1, W2 and b2. Suppose we have already computed the activations a1(x) in the first layer on a training sample x. Please compute the partial gradients of the activations a1(x) in the first layer w.r.t. the linear logits z1(x) in the first layer    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * a1: the element-wise sigmoid activations in the 1st layer, a float numpy vector of length h.  The i-th element represents the sigmoid of the i-th logit z1[i]
    ---- Outputs: --------
    * da1_dz1: the partial gradient of the activations a1 w.r.t. the logits z1, a float numpy matrix of shape (h, h).  The (i,j)-th element of da1_dz1 represents the partial gradient ( d_a1[i]  / d_z1[j] )
    ---- Hints: --------
    * The activations a1(x) in the first layer are computed using element-wise sigmoid function. For any element i, a1[i]= sigmoid(z1[i]). So a1(x) is NOT computed by softmax on z1(x). In this case, you cannot use the compute_da_dz() function in ether softmax regression or logistic regression. You need to implement this gradient function from scratch. 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_da1_dz1(self, a1):
        ##############################
        ## INSERT YOUR CODE HERE (0.6 points)
        da1_dz1 = np.diag(a1 * (1 - a1))
        ##############################
        return da1_dz1
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Fully_Connected_NN_compute_da1_dz1
        (Mac /Linux): python3 -m pytest -v test_2.py -m Fully_Connected_NN_compute_da1_dz1
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: check_da1_dz1  ------
    ''' Goal: Gradient Checking: compute the local gradient of the logit function using gradient check    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * z1: the input logits values of activation function, a float vector of length h
    * delta: a small number for gradient check, a float scalar
    ---- Outputs: --------
    * da1_dz1: the approximated local gradient of the activations a1 w.r.t. the logits z1, a float numpy vector of shape p by 1. The i-th element of da1_dz1 represents the partial gradient ( d_a1[i]  / d_z1[i] )
    '''
    def check_da1_dz1(self, z1, delta=1e-7):
        h = z1.shape[0]
        da1_dz1 = np.zeros((self.h,self.h))
        for i in range(self.h):
            for j in range(self.h):
                d = np.zeros(self.h)
                d[j] = delta
                da1_dz1[i,j] = (self.compute_a1(z1+d)[i] - self.compute_a1(z1)[i]) / delta
        return da1_dz1
        
    #----------------------------------------------------------
    
    #------------- Method: compute_dL_dz1  ------
    ''' Goal: (Global Gradient 1.2) Suppose we are given a fully-connected neural network with parameters W1, b1, W2 and b2. Suppose we have already computed the global gradients of the loss L w.r.t. the activation a1(x) in the first layer and the local gradient of the activations a1(x) w.r.t. the linear logits z1(x) on a training sample x. Please compute the partial gradients of the loss L w.r.t. the linear logits z1(x) in the first layer using chain rule    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * dL_da1: the element-wise sigmoid activations in the 1st layer, a float numpy vector of length h.  The i-th element represents the sigmoid of the i-th logit z1[i]
    * da1_dz1: the partial gradient of the activations a1 w.r.t. the logits z1, a float numpy matrix of shape (h, h).  The (i,j)-th element of da1_dz1 represents the partial gradient ( d_a1[i]  / d_z1[j] )
    ---- Outputs: --------
    * dL_dz1: the partial gradient ofthe loss L w.r.t. the logits z1, a float numpy vector of length h.  The i-th element of represents the partial gradient ( d_L  / d_z1[i] )
    ---- Hints: --------
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_dL_dz1(self, dL_da1, da1_dz1):
        ##############################
        ## INSERT YOUR CODE HERE (0.9 points)
        dL_dz1 = np.dot(dL_da1, da1_dz1)
        ##############################
        return dL_dz1
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Fully_Connected_NN_compute_dL_dz1
        (Mac /Linux): python3 -m pytest -v test_2.py -m Fully_Connected_NN_compute_dL_dz1
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_z2  ------
    ''' Goal: (Value Forward Function 2.1) Suppose we are given a fully-connected neural network with parameters W1, b1, W2 and b2. Suppose we have already computed the activations a1(x) in the first layer on a data sample x, please compute the linear logits z2(x) in the second layer on the data sample    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * a1: the element-wise sigmoid activations in the 1st layer, a float numpy vector of length h.  The i-th element represents the sigmoid of the i-th logit z1[i]
    ---- Outputs: --------
    * z2: the linear logits in the second layer, a float numpy vector of length c
    ---- Hints: --------
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_z2(self, a1):
        ##############################
        ## INSERT YOUR CODE HERE (0.6 points)
        z2 = np.dot(self.W2, a1) + self.b2
        ##############################
        return z2
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Fully_Connected_NN_compute_z2
        (Mac /Linux): python3 -m pytest -v test_2.py -m Fully_Connected_NN_compute_z2
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_dz2_db2  ------
    ''' Goal: (Local Gradient 2.1.1) Suppose we are given a fully-connected neural network with parameters W1, b1, W2 and b2. Suppose we have already computed the activations a1(x) in the first layer and the linear logits z2(x) in the second layer on a training sample x. Please compute the partial gradients of the linear logits z2(x) in the second layer w.r.t. the biases b2 in the second layer    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    ---- Outputs: --------
    * dz2_db2: the partial gradient of the logits z2 w.r.t. the biases b2, a float matrix of shape (c, c).  Each (i,j)-th element represents the partial gradient of the i-th logit z2[i] w.r.t. the j-th bias b2[j]:  d_z2[i] / d_b2[j]
    ---- Hints: --------
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_dz2_db2(self):
        ##############################
        ## INSERT YOUR CODE HERE (0.9 points)
        dz2_db2 = np.eye(self.c)
        ##############################
        return dz2_db2
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Fully_Connected_NN_compute_dz2_db2
        (Mac /Linux): python3 -m pytest -v test_2.py -m Fully_Connected_NN_compute_dz2_db2
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_dL_db2  ------
    ''' Goal: (Global Gradient 2.1.1) Suppose we are given a fully-connected neural network with parameters W1, b1, W2 and b2. Suppose we have already computed the global gradients of the loss L w.r.t. the linear logits z2(x) in the second layer and the local gradient of the linear logits z2(x) w.r.t. the biases b2 on a training sample x. Please compute the partial gradients of the loss L w.r.t. the biases b2 in the second layer using chain rule    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * dL_dz2: the partial gradient of the loss L w.r.t. the logits z2, a float numpy vector of length c.  The i-th element represents the partial gradient ( d_L  / d_z2[i] )
    * dz2_db2: the partial gradient of the logits z2 w.r.t. the biases b2, a float matrix of shape (c, c).  Each (i,j)-th element represents the partial gradient of the i-th logit z2[i] w.r.t. the j-th bias b2[j]:  d_z2[i] / d_b2[j]
    ---- Outputs: --------
    * dL_db2: the partial gradient of the loss L w.r.t. the biases b2, a float vector of length c.  Each i-th element represents the partial gradient of loss L w.r.t. the i-th bias b2[i]:  d_L / d_b2[i]
    ---- Hints: --------
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_dL_db2(self, dL_dz2, dz2_db2):
        ##############################
        ## INSERT YOUR CODE HERE (0.6 points)
        dL_db2 = np.dot(dL_dz2, dz2_db2)
        ##############################
        return dL_db2
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Fully_Connected_NN_compute_dL_db2
        (Mac /Linux): python3 -m pytest -v test_2.py -m Fully_Connected_NN_compute_dL_db2
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_dz2_dW2  ------
    ''' Goal: (Local Gradient 2.1.2) Suppose we are given a fully-connected neural network with parameters W1, b1, W2 and b2. Suppose we have already computed the activations a1(x) in the first layer and the linear logits z2(x) in the second layer on a training sample x. Please compute the partial gradients of the linear logits z2(x) in the second layer w.r.t. the weights W2 in the second layer    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * a1: the element-wise sigmoid activations in the 1st layer, a float numpy vector of length h.  The i-th element represents the sigmoid of the i-th logit z1[i]
    ---- Outputs: --------
    * dz2_dW2: the partial gradient of logits z2 w.r.t. the weight matrix W2, a numpy float tensor of shape (c by c by h).  The (i,j,k)-th element represents the partial gradient of the i-th logit (z2[i]) w.r.t. the weight W2[j,k]:   d_z2[i] / d_W2[j,k]
    ---- Hints: --------
    * This problem can be solved using only 3 line(s) of code. More lines are okay.'''
    def compute_dz2_dW2(self, a1):
        ##############################
        ## INSERT YOUR CODE HERE (0.9 points)
        dz2_dW2 = np.zeros((self.c, self.c, self.h))
        for i in range(self.c):
            for j in range(self.c):
                if i == j:
                    dz2_dW2[i, j, :] = a1
        ##############################
        return dz2_dW2
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Fully_Connected_NN_compute_dz2_dW2
        (Mac /Linux): python3 -m pytest -v test_2.py -m Fully_Connected_NN_compute_dz2_dW2
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_dL_dW2  ------
    ''' Goal: (Global Gradient 2.1.2) Suppose we are given a fully-connected neural network with parameters W1, b1, W2 and b2. Suppose we have already computed the global gradients of the loss L w.r.t. the linear logits z2(x) in the second layer and the local gradient of the linear logits z2(x) w.r.t. the weights w2 on a training sample x. Please compute the partial gradients of the loss L w.r.t. the weights W2 in the second layer using chain rule    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * dL_dz2: the partial gradient of the loss L w.r.t. the logits z2, a float numpy vector of length c.  The i-th element represents the partial gradient ( d_L  / d_z2[i] )
    * dz2_dW2: the partial gradient of logits z2 w.r.t. the weight matrix W2, a numpy float tensor of shape (c by c by h).  The (i,j,k)-th element represents the partial gradient of the i-th logit (z2[i]) w.r.t. the weight W2[j,k]:   d_z2[i] / d_W2[j,k]
    ---- Outputs: --------
    * dL_dW2: the partial gradient of the loss L w.r.t. the weight matrix W2, a numpy float matrix of shape (c by h).  The (i,j)-th element represents the partial gradient of the loss L w.r.t. the weight W2[i,j]:   d_L / d_W2[i,j]
    ---- Hints: --------
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_dL_dW2(self, dL_dz2, dz2_dW2):
        ##############################
        ## INSERT YOUR CODE HERE (1.5 points)
        dL_dW2 = np.einsum('i,ijk->jk', dL_dz2, dz2_dW2)
        ##############################
        return dL_dW2
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Fully_Connected_NN_compute_dL_dW2
        (Mac /Linux): python3 -m pytest -v test_2.py -m Fully_Connected_NN_compute_dL_dW2
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_dz2_da1  ------
    ''' Goal: (Local Gradient 2.1.3) Suppose we are given a fully-connected neural network with parameters W1, b1, W2 and b2. Suppose we have already computed the activations a1(x) in the first layer and the linear logits z2(x) in the second layer on a training sample x. Please compute the partial gradients of the linear logits z2(x) in the second layer w.r.t. the activations a1(x) in the first layer    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    ---- Outputs: --------
    * dz2_da1: the partial gradient of the logits z2 w.r.t. the inputs a1, a float numpy matrix of shape (c, h).  The (i,j)-th element represents the partial gradient ( d_z2[i]  / d_a1[j] )
    ---- Hints: --------
    * The activations a1(x) in the first layer is used as the input to the second layer for computing z2(x). 
    * This gradient is a constant matrix, which doesn't need any input. 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_dz2_da1(self):
        ##############################
        ## INSERT YOUR CODE HERE (1.5 points)
        dz2_da1 = self.W2
        ##############################
        return dz2_da1
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Fully_Connected_NN_compute_dz2_da1
        (Mac /Linux): python3 -m pytest -v test_2.py -m Fully_Connected_NN_compute_dz2_da1
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_dL_da1  ------
    ''' Goal: (Global Gradient 2.1.3) Suppose we are given a fully-connected neural network with parameters W1, b1, W2 and b2. Suppose we have already computed the global gradients of the loss w.r.t. the linear logits z2(x) and the local gradients of the linear logits z2(x) w.r.t. the activations a1(x) in the first layer on a training sample x. Please compute the partial gradient of the loss function L w.r.t. the activations a1(x) in the first layer using chain rule    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * dL_dz2: the partial gradient of the loss L w.r.t. the logits z2, a float numpy vector of length c.  The i-th element represents the partial gradient ( d_L  / d_z2[i] )
    * dz2_da1: the partial gradient of the logits z2 w.r.t. the inputs a1, a float numpy matrix of shape (c, h).  The (i,j)-th element represents the partial gradient ( d_z2[i]  / d_a1[j] )
    ---- Outputs: --------
    * dL_da1: the partial gradient of the loss L w.r.t. the activations a1, a float numpy vector of shape h.  The i-th element represents the partial gradient ( d_L  / d_a1[i] )
    ---- Hints: --------
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_dL_da1(self, dL_dz2, dz2_da1):
        ##############################
        ## INSERT YOUR CODE HERE (1.5 points)
        dL_da1 = np.dot(dL_dz2, dz2_da1)
        ##############################
        return dL_da1
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Fully_Connected_NN_compute_dL_da1
        (Mac /Linux): python3 -m pytest -v test_2.py -m Fully_Connected_NN_compute_dL_da1
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_a2  ------
    ''' Goal: (Value Forward Function 2.2) Suppose we are given a fully-connected neural network with parameters W1, b1, W2 and b2. Suppose we have already computed the linear logits z2(x) in the second layer on a data sample x, please compute the softmax activations a2(x) in the second layer on the data sample    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * z2: the linear logits in the second layer, a float numpy vector of length c
    ---- Outputs: --------
    * a2: the softmax activations in the 2nd layer, a float numpy vector of length c
    ---- Hints: --------
    * This problem can be solved using only 2 line(s) of code. More lines are okay.'''
    def compute_a2(self, z2):
        ##############################
        ## INSERT YOUR CODE HERE (1.5 points)
        exp_z2 = np.exp(z2 - np.max(z2))
        a2 = exp_z2 / np.sum(exp_z2)
        ##############################
        return a2
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Fully_Connected_NN_compute_a2
        (Mac /Linux): python3 -m pytest -v test_2.py -m Fully_Connected_NN_compute_a2
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_da2_dz2  ------
    ''' Goal: (Local Gradient 2.1.3) Suppose we are given a fully-connected neural network with parameters W1, b1, W2 and b2. Suppose we have already computed the activations a1(x) in the first layer and the linear logits z2(x) in the second layer on a training sample x. Please compute the partial gradients of the linear logits z2(x) in the second layer w.r.t. the activations a1(x) in the first layer    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * a2: the softmax activations in the 2nd layer, a float numpy vector of length c
    ---- Outputs: --------
    * da2_dz2: the partial gradient of the activations a2 w.r.t. the logits z2, a float numpy matrix of shape (c by c).  The (i,j)-th element represents the partial gradient ( d_a2[i]  / d_z2[j] )
    ---- Hints: --------
    * The activations a1(x) in the first layer is used as the input to the second layer for computing z2(x). 
    * This gradient is a constant matrix, which doesn't need any input. 
    * This problem can be solved using only 3 line(s) of code. More lines are okay.'''
    def compute_da2_dz2(self, a2):
        ##############################
        ## INSERT YOUR CODE HERE (1.5 points)
        da2_dz2 = np.diag(a2) - np.outer(a2, a2)
        ##############################
        return da2_dz2
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Fully_Connected_NN_compute_da2_dz2
        (Mac /Linux): python3 -m pytest -v test_2.py -m Fully_Connected_NN_compute_da2_dz2
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_dL_dz2  ------
    ''' Goal: (Global Gradient 2.2) Suppose we are given a fully-connected neural network with parameters W1, b1, W2 and b2. Suppose we have already computed the global gradients of the loss w.r.t. the activations a2(x) and the local gradients of the activations a2(x) w.r.t.  the linear logits z2(x) in the second layer on a training sample x. Please compute the partial gradient of the loss function L w.r.t. the linear logits z2(x) in the first layer using chain rule    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * dL_da2: the partial gradients of the loss function L w.r.t. the activations a2, a float numpy vector of length c.  The i-th element dL_da2[i] represents the partial gradient of the loss function L w.r.t. the i-th activation a2[i]:  d_L / d_a2[i]
    * da2_dz2: the partial gradient of the activations a2 w.r.t. the logits z2, a float numpy matrix of shape (c by c).  The (i,j)-th element represents the partial gradient ( d_a2[i]  / d_z2[j] )
    ---- Outputs: --------
    * dL_dz2: the partial gradient of the loss L w.r.t. the logits z2, a float numpy vector of length c.  The i-th element represents the partial gradient ( d_L  / d_z2[i] )
    ---- Hints: --------
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_dL_dz2(self, dL_da2, da2_dz2):
        ##############################
        ## INSERT YOUR CODE HERE (1.5 points)
        dL_dz2 = np.dot(dL_da2, da2_dz2)
        ##############################
        return dL_dz2
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Fully_Connected_NN_compute_dL_dz2
        (Mac /Linux): python3 -m pytest -v test_2.py -m Fully_Connected_NN_compute_dL_dz2
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_L  ------
    ''' Goal: (Value Forward Function 3) Suppose we are given a fully-connected neural network and we have already computed the activations a2(x) in the second layer on a training sample x. Suppose the label of the training sample is y. Please compute the loss on the training sample using multi-class cross entropy loss    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * a2: the softmax activations in the 2nd layer, a float numpy vector of length c
    * y: the label of a data sample, an integer scalar value. The values can be 0,1,2, ..., or (c-1)
    ---- Outputs: --------
    * L: the multi-class cross entropy loss, a float scalar
    ---- Hints: --------
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_L(self, a2, y):
        ##############################
        ## INSERT YOUR CODE HERE (1.5 points)
        # L = -np.log(a2[y])
        L = -np.log(a2[y]) if a2[y] > 0 else 10000000000
        ##############################
        return L
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Fully_Connected_NN_compute_L
        (Mac /Linux): python3 -m pytest -v test_2.py -m Fully_Connected_NN_compute_L
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_dL_da2  ------
    ''' Goal: (Local/Global Gradient 3) Suppose we are given a fully-connected neural network and we have already computed the activations a2(x) in the second layer. Suppose the label of the training sample is y. Please compute the partial gradients of the loss L w.r.t. the activations a2(x) in the second layer    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * a2: the softmax activations in the 2nd layer, a float numpy vector of length c
    * y: the label of a data sample, an integer scalar value. The values can be 0,1,2, ..., or (c-1)
    ---- Outputs: --------
    * dL_da2: the partial gradients of the loss function L w.r.t. the activations a2, a float numpy vector of length c.  The i-th element dL_da2[i] represents the partial gradient of the loss function L w.r.t. the i-th activation a2[i]:  d_L / d_a2[i]
    ---- Hints: --------
    * This problem can be solved using only 2 line(s) of code. More lines are okay.'''
    def compute_dL_da2(self, a2, y):
        ##############################
        ## INSERT YOUR CODE HERE (1.5 points)
        eye = np.eye(len(a2))[y]
        dL_da2 = -eye / (a2[y] + 1e-15)
        ##############################
        return dL_da2
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Fully_Connected_NN_compute_dL_da2
        (Mac /Linux): python3 -m pytest -v test_2.py -m Fully_Connected_NN_compute_dL_da2
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: forward  ------
    ''' Goal: (Forward Pass) Suppose we are given a fully-connected neural network with parameters W1, b1, W2 and b2. Given a data sample (x), please compute the activations a1(x) and a2(x) on the sample    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: the feature vector of one data sample, a numpy vector of length p
    ---- Outputs: --------
    * a1: the element-wise sigmoid activations in the 1st layer, a float numpy vector of length h.  The i-th element represents the sigmoid of the i-th logit z1[i]
    * a2: the softmax activations, a float numpy vector of length c
    ---- Hints: --------
    * This problem can be solved using only 4 line(s) of code. More lines are okay.'''
    def forward(self, x):
        ##############################
        ## INSERT YOUR CODE HERE (1.5 points)
        z1 = self.compute_z1(x)
        a1 = self.compute_a1(z1)
        z2 = self.compute_z2(a1)
        a2 = self.compute_a2(z2)
        ##############################
        return a1, a2
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Fully_Connected_NN_forward
        (Mac /Linux): python3 -m pytest -v test_2.py -m Fully_Connected_NN_forward
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: backward_layer2  ------
    ''' Goal: Back Propagation in the second layer: Suppose we are given a fully-connected neural network with parameters (W1, b1, W2 and b2) and we have a training data sample (x) with label (y).  Suppose we have already computed the activations a1(x) in the first layer and the activations a2(x) in the second layer on the data sample in the forward-pass. Please compute the global gradients of the loss L w.r.t. the parameters W2, b2 and the activation a1(x) on the data sample using back propagation    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * y: the label of one data sample, an integer scalar value. The values can be 0,1,2, ..., or (c-1)
    * a1: the element-wise sigmoid activations in the 1st layer, a float numpy vector of length h.  The i-th element represents the sigmoid of the i-th logit z1[i]
    * a2: the softmax activations, a float numpy vector of length c
    ---- Outputs: --------
    * dL_dW2: the partial gradient of the loss L w.r.t. the weight matrix W2, a numpy float matrix of shape (c by h).  The (i,j)-th element represents the partial gradient of the loss L w.r.t. the weight W2[i,j]:   d_L / d_W2[i,j]
    * dL_db2: the partial gradient of the loss L w.r.t. the biases b2, a float vector of length c.  Each i-th element represents the partial gradient of loss L w.r.t. the i-th bias b2[i]:  d_L / d_b2[i]
    * dL_da1: the partial gradient of the loss L w.r.t. the activations a1, a float numpy vector of shape h.  The i-th element represents the partial gradient ( d_L  / d_a1[i] )
    ---- Hints: --------
    * It's easier to follow a certain order to compute all the gradients: dL_da2, da2_dz2, dL_dz2, dz2_db2, dL_db2 .... 
    * This problem can be solved using only 9 line(s) of code. More lines are okay.'''
    def backward_layer2(self, y, a1, a2):
        ##############################
        ## INSERT YOUR CODE HERE (1.5 points)
        dL_da2 = self.compute_dL_da2(a2, y)
        da2_dz2 = self.compute_da2_dz2(a2)
        dL_dz2 = self.compute_dL_dz2(dL_da2, da2_dz2)
        dL_dW2 = self.compute_dL_dW2(dL_dz2, self.compute_dz2_dW2(a1))
        dL_db2 = self.compute_dL_db2(dL_dz2, self.compute_dz2_db2())
        dL_da1 = self.compute_dL_da1(dL_dz2, self.compute_dz2_da1())
        ##############################
        return dL_dW2, dL_db2, dL_da1
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Fully_Connected_NN_backward_layer2
        (Mac /Linux): python3 -m pytest -v test_2.py -m Fully_Connected_NN_backward_layer2
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: backward_layer1  ------
    ''' Goal: Back Propagation in the first layer: Suppose we are given a fully-connected neural network with parameters (W1, b1, W2 and b2) and we have a training data sample (x) with label (y).  Suppose we have already computed the activations a1(x) in the first layer on the data sample in the forward-pass. Please compute the global gradients of the loss w.r.t. the parameters W1 and b1 on the data sample using back propagation    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: the feature vector of a data sample, a float numpy vector of length p
    * a1: the element-wise sigmoid activations in the 1st layer, a float numpy vector of length h.  The i-th element represents the sigmoid of the i-th logit z1[i]
    * dL_da1: the partial gradient of the loss L w.r.t. the activations a1, a float numpy vector of shape h.  The i-th element represents the partial gradient ( d_L  / d_a1[i] )
    ---- Outputs: --------
    * dL_dW1: the partial gradients of the loss function L w.r.t. the weights W1, a float numpy matrix of shape (h by p).  The i,j-th element represents the partial gradient of the loss function L w.r.t. the i,j-th weight W1[i,j]:  d_L / d_W1[i,j]
    * dL_db1: the partial gradients of the loss function L w.r.t. the biases b1, a float numpy vector of length h.  The i-th element represents the partial gradient of the loss function L w.r.t. the i-th bias b2[i]:  d_L / d_b2[i]
    ---- Hints: --------
    * It's easier to follow a certain order to compute all the gradients: da1_dz1, dL_dz1, dz1_db1, dL_db1 .... 
    * This problem can be solved using only 6 line(s) of code. More lines are okay.'''
    def backward_layer1(self, x, a1, dL_da1):
        ##############################
        ## INSERT YOUR CODE HERE (1.5 points)
        da1_dz1 = self.compute_da1_dz1(a1)
        dL_dz1 = np.dot(dL_da1, da1_dz1)
        dz1_dW1 = self.compute_dz1_dW1(x)
        dz1_db1 = self.compute_dz1_db1()
        dL_dW1 = np.einsum('i,ijk->jk', dL_dz1, dz1_dW1)
        dL_db1 = np.dot(dL_dz1, dz1_db1)
        ##############################
        return dL_dW1, dL_db1
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Fully_Connected_NN_backward_layer1
        (Mac /Linux): python3 -m pytest -v test_2.py -m Fully_Connected_NN_backward_layer1
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: backward  ------
    ''' Goal: Back Propagation (layer 2 and 1): Suppose we are given a fully-connected neural network with parameters (W1, b1, W2 and b2) and we have a training data sample (x) with label (y).  Suppose we have already computed the activations a1(x) in the first layer and the activations a2(x) in the second layer on the data sample in the forward-pass. Please compute the global gradients of the loss w.r.t. the parameters W1, b1, W2 and b2 on the data sample using back propagation    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: the feature vector of a data sample, a float numpy vector of length p
    * y: the label of one data sample, an integer scalar value. The values can be 0,1,2, ..., or (c-1)
    * a1: the element-wise sigmoid activations in the 1st layer, a float numpy vector of length h.  The i-th element represents the sigmoid of the i-th logit z1[i]
    * a2: the softmax activations, a float numpy vector of length c
    ---- Outputs: --------
    * dL_dW2: the partial gradient of the loss L w.r.t. the weight matrix W2, a numpy float matrix of shape (c by h).  The (i,j)-th element represents the partial gradient of the loss L w.r.t. the weight W2[i,j]:   d_L / d_W2[i,j]
    * dL_db2: the partial gradient of the loss L w.r.t. the biases b2, a float vector of length c.  Each i-th element represents the partial gradient of loss L w.r.t. the i-th bias b2[i]:  d_L / d_b2[i]
    * dL_dW1: the partial gradients of the loss function L w.r.t. the weights W1, a float numpy matrix of shape (h by p).  The i,j-th element represents the partial gradient of the loss function L w.r.t. the i,j-th weight W1[i,j]:  d_L / d_W1[i,j]
    * dL_db1: the partial gradients of the loss function L w.r.t. the biases b1, a float numpy vector of length h.  The i-th element represents the partial gradient of the loss function L w.r.t. the i-th bias b2[i]:  d_L / d_b2[i]
    ---- Hints: --------
    * This problem can be solved using only 2 line(s) of code. More lines are okay.'''
    def backward(self, x, y, a1, a2):
        ##############################
        ## INSERT YOUR CODE HERE (1.5 points)
        dL_dW2, dL_db2, dL_da1 = self.backward_layer2(y, a1, a2)
        dL_dW1, dL_db1 = self.backward_layer1(x, a1, dL_da1)
        ##############################
        return dL_dW2, dL_db2, dL_dW1, dL_db1
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Fully_Connected_NN_backward
        (Mac /Linux): python3 -m pytest -v test_2.py -m Fully_Connected_NN_backward
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: check_dL_dW2  ------
    ''' Goal: Gradient Checking: Compute the gradient of weight W2 using gradient checking    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: the feature vector of a data sample, a float numpy vector of length p
    * y: the label of one data sample, an integer scalar value. The values can be 0,1,2, ..., or (c-1)
    * delta: a small number for gradient check, a float scalar
    ---- Outputs: --------
    * dL_dW2: the approximated partial gradient of the loss L w.r.t. the weight matrix W2, a numpy float matrix of shape (c by h).  The (i,j)-th element represents the partial gradient of the loss L w.r.t. the weight W2[i,j]:   d_L / d_W2[i,j]
    '''
    def check_dL_dW2(self, x, y, delta=1e-7):
        dL_dW2 = np.zeros((self.c,self.h)) 
        for i in range(self.c):
            for j in range(self.h):
                d = np.zeros((self.c,self.h))
                d[i,j] = delta
                a1,a2 = self.forward(x)
                L0 = self.compute_L(a2,y)
                self.W2+=d
                a1,a2 = self.forward(x)
                L1 = self.compute_L(a2,y)
                self.W2-=d
                dL_dW2[i,j] = (L1 - L0) / delta
        return dL_dW2
        
    #----------------------------------------------------------
    
    #------------- Method: check_dL_dW1  ------
    ''' Goal: Gradient Checking: Compute the gradient of weight W1 using gradient checking    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: the feature vector of a data sample, a float numpy vector of length p
    * y: the label of one data sample, an integer scalar value. The values can be 0,1,2, ..., or (c-1)
    * delta: a small number for gradient check, a float scalar
    ---- Outputs: --------
    * dL_dW1: the approximated partial gradient of the loss L w.r.t. the weight matrix W1, a numpy float matrix of shape (h by p).  The (i,j)-th element represents the partial gradient of the loss L w.r.t. the weight W1[i,j]:   d_L / d_W1[i,j]
    '''
    def check_dL_dW1(self, x, y, delta=1e-7):
        dL_dW1 = np.zeros((self.h,self.p)) 
        for i in range(self.h):
            for j in range(self.p):
                d = np.zeros((self.h,self.p))
                d[i,j] = delta
                a1,a2 = self.forward(x)
                L0 = self.compute_L(a2,y)
                self.W1+=d
                a1,a2 = self.forward(x)
                L1 = self.compute_L(a2,y)
                self.W1-=d
                dL_dW1[i,j] = (L1 - L0) / delta
        return dL_dW1
        
    #----------------------------------------------------------
    
    #------------- Method: train  ------
    ''' Goal: (Train a Fully-Connected Neural Network) Given a training dataset, train the Fully Connected Neural Network by iteratively updating the weights W and biases b using the gradients computed over each data instance    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * X: the feature matrix of training instances, a float numpy matrix of shape (n by p)
    * Y: the labels of training instance, a numpy integer numpy array of length n. The values can be 0, 1, ..., or c-1
    ---- Hints: --------
    * Step 1 Forward pass: compute the linear logits, activations and loss. 
    * Step 2 Back propagation: compute the gradients. 
    * Step 3 Gradient descent: update the parameters using gradient descent. 
    * This problem can be solved using only 6 line(s) of code. More lines are okay.'''
    def train(self, X, Y):
        n = X.shape[0] # n: the number of training samples
        for _ in range(self.n_epoch): # iterate through the dataset n_epoch times
            indices = np.random.permutation(n) # shuffle the indices of all samples
            for i in indices: # iterate through each random training sample (x,y)
                x=X[i] # the feature vector of the i-th random sample
                y=Y[i] # the label of the i-th random sample
                ##############################
                ## INSERT YOUR CODE HERE (1.5 points)
                a1, a2 = self.forward(x)
                dL_dW2, dL_db2, dL_dW1, dL_db1 = self.backward(x, y, a1, a2)
                self.W2 -= self.lr * dL_dW2
                self.b2 -= self.lr * dL_db2
                self.W1 -= self.lr * dL_dW1
                self.b1 -= self.lr * dL_db1
                ##############################
        
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Fully_Connected_NN_train
        (Mac /Linux): python3 -m pytest -v test_2.py -m Fully_Connected_NN_train
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: predict  ------
    ''' Goal: (Using Fully-Connected Network 2) Given a trained full-connected neural network with parameters W1, b1, W2 and b2. Suppose we have a test dataset Xtest (features). For each data sample x in Xtest, use the model to predict the label of x and the probabilities of the label being in each of the classes, i.e. the activation a2(x)    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * Xt: the feature matrix of all testing instances, a numpy matrix of shape n by p. n is the number of test samples, p is the number of features. Xt[i] represents the i-th test sample in the dataset for label prediction
    ---- Outputs: --------
    * yt: the predicted labels of the testing instances, a numpy float vector of length n, y[i] represents the predicted label of the i-th instance in the dataset
    * P: the predicted activations of test data samples to be in different classes, a float numpy matrix of shape (n_test,c).  P[i] is the vector of activations of the i-th data sample
    ---- Hints: --------
    * If we have multiple elements in the activations being the largest at the same time (for example, [0.5, 0.5,0] have two largest values), we can break the tie by choosing the element with the smallest index. 
    * This problem can be solved using only 2 line(s) of code. More lines are okay.'''
    def predict(self, Xt):
        n_test = Xt.shape[0] # the number of test samples
        yt = np.zeros(n_test) # initialize label vector as all zeros
        P = np.zeros((n_test,self.c)) # initialize activation matrix as all zeros
        for i in range(n_test): # iterate through each test sample
            x=Xt[i] # the feature vector of the i-th data sample
            ##############################
            ## INSERT YOUR CODE HERE (1.5 points)
            _, a2 = self.forward(x)
            P[i, :] = a2
            yt[i] = np.argmax(a2)
            ##############################
        return yt, P
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Fully_Connected_NN_predict
        (Mac /Linux): python3 -m pytest -v test_2.py -m Fully_Connected_NN_predict
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    

'''-------- TEST problem2.py file: (30 points) ----------
Please type the following command in your terminal to test the correctness of all the above functions in this file:
        (Windows OS): python -m pytest -v test_2.py
        (Mac /Linux): python3 -m pytest -v test_2.py
------------------------------------------------------'''






