




#------------ No New Package --------------
# NOTE: Please don't import any new package. You should be able to solve the problems using only the package(s) imported here.
import numpy as np
from problem1 import *
#---------------------------------------------------------


# ---------------------------------------------------------
'''
    Goal of Problem 2: Using the Auto-gradient System (Scalar and SGD_optimizer) (20 points)
    In this problem, we will test our auto-gradient system by building simple models with only 1-dimensional input features. We will use our automatic gradient system (Scalar and SGD_optimizer) in the previous problem to perform gradient descent algorithm on our models.
    
'''
# ---------------------------------------------------------

'''------------- Class: LinearModel (16.0 points) -------
    In this class, the goal is to build a simple linear regression model (least square regression) with only 1 dimensional input feature. In this model, we have 1 dimensional linear model z = w*x + b, here w and b are the parameters of the linear model. We will use the automatic gradient system in the previous problem to optimize the parameters of this model 
'''
''' ---- Class Properties ----
    * w: the weight paramester of the linear model, a Scalar.
    * b: the bias paramester of the linear model, a Scalar.
    '''
class LinearModel:
    #------------- Method: __init__  ------
    ''' Goal: create and initialize the paramesters of the 1d linear model    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    '''
    def __init__(self):
        self.w = Scalar(np.random.rand()) # a Scalar variable with random value
        self.b = Scalar(np.random.rand()) # a Scalar variable with random value
        
        
    #----------------------------------------------------------
    
    #------------- Method: compute_z  ------
    ''' Goal: Let's build the first half of the computational graph in the linear regression model. Here we want to compute the output of the linear model z =  w*x + b    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: the feature value of the data, a Scalar
    ---- Outputs: --------
    * z: the output of the linear model, a Scalar
    ---- Hints: --------
    * You can access to the parameters (w and b) through 'self', such as 'self.w'. 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_z(self, x):
        ##############################
        ## INSERT YOUR CODE HERE (4.8 points)
        
        ##############################
        return z
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m LinearModel_compute_z
        (Mac /Linux): python3 -m pytest -v test_2.py -m LinearModel_compute_z
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_L  ------
    ''' Goal: This is the second half of the computational graph in the linear regression model. Please compute the squared error loss on the linear model L= (z-y)^2. Here the notation '()^2' represents the square value of ()    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * z: the output of the linear model, a Scalar
    * y: the label of the data instance, a Scalar
    ---- Outputs: --------
    * L: the loss of the linear model, a Scalar
    ---- Hints: --------
    * In order to connect the gradients between variables in the computational graph, you may want to use the operators in the Scalar class. For example, x**2 is the python operator for square operation, where the value can be computed correctly in the forward pass, but the gradients are not connected. Instead, the x.square() is a Scalar operator that we implemented, where we have connected the gradient functions in the backward pass. So you may want to use this function instead. 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_L(self, z, y):
        ##############################
        ## INSERT YOUR CODE HERE (4.8 points)
        
        ##############################
        return L
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m LinearModel_compute_L
        (Mac /Linux): python3 -m pytest -v test_2.py -m LinearModel_compute_L
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: train  ------
    ''' Goal: Given a training dataset, please train the parameters of the linear regression model using Stochastic Gradient Descent algorithm    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * dataset: the training dataset, a list of training samples, where each training sample is a pair (x,y) of the feature and label
    * n_epoch: the number of epochs or passes through the whole dataset, an integer scalar
    * lr: the learning rate of the gradient descent, a float scalar
    ---- Hints: --------
    * You may want to reset the gradients of the parameters to zeros after performing gradient descient, so that in the next step, the gradient can be correctly computed in the backward pass. 
    * This problem can be solved using only 5 line(s) of code. More lines are okay.'''
    def train(self, dataset, n_epoch, lr=0.01):
        opt = SGD_optimizer([self.w, self.b], lr) # create an optimizer for the model parameters
        for _ in range(n_epoch): # iterate through the dataset n_epoch times
            for x, y in dataset: # iterate through each data sample (x,y)
                ##############################
                ## INSERT YOUR CODE HERE (6.4 points)
                pass 
                ##############################
        
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m LinearModel_train
        (Mac /Linux): python3 -m pytest -v test_2.py -m LinearModel_train
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
'''------------- Class: LogisticRegression (4.0 points) -------
    In this class, the goal is to build a simple logistic regression model with only 1 dimensional input feature. In this model, we have 1 dimensional linear model z = w*x + b, here w and b are the parameters of the linear model. As you may have noticed, this logistic regression model is actually very similar to the previous model (LinearModel) we implemented above. There is only one minor difference, where in logistic regression, we use a different loss function than LinearModel. So we can simply re-use the code in LinearModel by using the LinearModel class as the parent class of this class. So that we can inherit all the methods from the LinearModel class, and only change/add the new parts (i.e., the loss function) for the logistic regression model. Once the new loss function is in place, the model can be trained the same way as the LinearModel 
'''

class LogisticRegression(LinearModel):
    #------------- Method: compute_L  ------
    ''' Goal: Let's build the only new component in the logistic regression model. Here we want to overwrite the compute_L() function in the parent class (LinearModel), so that in this class we could use the loss of the logistic regression model during training. Please compute the loss for the logistic regression model directly from the linear logit (z) and label (y)    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * z: the linear logit or the output of the linear model, a Scalar
    * y: the label of the data sample, a Scalar of value 0 or 1
    ---- Outputs: --------
    * L: the loss of the logistic regression model, a Scalar
    ---- Hints: --------
    * In our Scalar system, we didn't implement any operator between float/int and Scalar instance, such as (1+z). So in order to connect the gradients, you may want to create a Scalar variable for the constant 1, using 'c= Scalar(1)' then compute (c+z) instead. 
    * When computing exp(z), you need to be careful about an overflowing case. When the z is a large number (say 1000),  the computer can no longer store the result of exp(z) in a floating-point number. In this case, you may want to avoid computing exp(z) and assign the final result of L(x) directly. When z is very large (say 1000), 1+exp(z) will be very close to exp(z), so we can simplify the equation "log(1+exp(z))-yz" into   "log (exp(z)) - yz". Here log() and exp() cancel each other, so we only have "z-yz" left, which is "z(1-y)". 
    * This problem can be solved using only 2 line(s) of code. More lines are okay.'''
    def compute_L(self, z, y):
        ##############################
        ## INSERT YOUR CODE HERE (4.0 points)
        
        ##############################
        return L
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m LogisticRegression_compute_L
        (Mac /Linux): python3 -m pytest -v test_2.py -m LogisticRegression_compute_L
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    

'''-------- TEST problem2.py file: (20 points) ----------
Please type the following command in your terminal to test the correctness of all the above functions in this file:
        (Windows OS): python -m pytest -v test_2.py
        (Mac /Linux): python3 -m pytest -v test_2.py
------------------------------------------------------'''






