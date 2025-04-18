




#------------ No New Package --------------
# NOTE: Please don't import any new package. You should be able to solve the problems using only the package(s) imported here.
from abc import abstractmethod
import numpy as np
#---------------------------------------------------------


# ---------------------------------------------------------
'''
    Goal of Problem 3: Linear Regression Models (20 points)
    In this problem, we will implement a linear regression model with only one set of parameters (w). In this model, we are going to assign a weight to each feature and add a constant bias. The weights and bias are combined into a vector (w), here w = (w0, w1, w2, ...). Given the featuers of a data sample x = (1, x1, x2,...), the prediction of the model is computed by the weighted sum of the input feature values, i.e.,  f(x) = w0 + w1 * x1 + w2 * x2 ...  Given a training data set of data samples, with their features and labels, we want to figure out the optimal values of the parameters in (w). We are going to use two different loss functions (squared error and ridge regression error) separately to train the model parameters (w).
    
'''
# ---------------------------------------------------------

'''------------- Class: Linear_Regression (0.0 points) -------
    This is the parent class of linear regression models, which only have one set of parameters (w). In this model, we are going to assign a weight to each feature and add a constant bias. The weights and bias are combined into a vector (w), here w = (w0, w1, w2, ...). Given the featuers of a data sample x = (1, x1, x2,...), the prediction of the model is computed by the weighted sum of the input feature values, i.e.,  f(x) = w0 + w1 * x1 + w2 * x2 ... 
'''
''' ---- Class Properties ----
    * w: the weights of the linear regression model, a numpy float vector of length p. Here w[i] is the weight of the model on the i-th feature.
    '''
class Linear_Regression:
    #------------- Method: __init__  ------
    ''' Goal: Initialize the model object    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * p: the number of features/measurements in each data sample, an integer scaler
    '''
    def __init__(self, p):
        self.w = np.zeros(p) # initialize all weight as 0s
        
        
    #----------------------------------------------------------
    
    #------------- Method: train  ------
    ''' Goal: Given a loss function (in a child class of this class) and a set of training samples, find the optimal value for the parameter w (i.e., self.w)    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * X: the feature matrix of the training samples, a numpy matrix of shape n by p, here X[i,j] is the value of the j-th feature on the i-th training sample
    * y: the labels of training samples, a numpy float vector of length n, y[i] represents the label of the i-th sample in the dataset
    '''
    @abstractmethod
    def train(self, X, y):
        pass
        
        
    #----------------------------------------------------------
    
    #------------- Method: predict  ------
    ''' Goal: Given a set of test instances, predict their labels. Suppose the parameter w of the model is given (self.w)    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * Xt: the feature matrix of all testing instances, a numpy matrix of shape n by p. n is the number of test samples, p is the number of features. Xt[i] represents the i-th test sample in the dataset for label prediction
    ---- Outputs: --------
    * yt: the predicted labels of the testing instances, a numpy float vector of length n, y[i] represents the predicted label of the i-th instance in the dataset
    '''
    def predict(self, Xt):
        yt = Xt @ self.w
        return yt
        
    #----------------------------------------------------------
    
'''------------- Class: Linear_Regression_SE (10.0 points) -------
    Least Square Regression: This is a child class of the above linear regression class, which use the sum of squared error as the loss function for finding optimal value for parameter w 
'''
''' ---- Class Properties ----
    * w: the weights of the linear regression model, a numpy float vector of length p. Here w[i] is the weight of the model on the i-th feature.
    '''
class Linear_Regression_SE(Linear_Regression):
    #------------- Method: train  ------
    ''' Goal: Given a loss function (sum of squared errors) and a set of training samples, find the optimal value for the parameter w (i.e., self.w)    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * X: the feature matrix of the training samples, a numpy matrix of shape n by p, here X[i,j] is the value of the j-th feature on the i-th training sample
    * y: the labels of the training samples, a numpy float vector of length n, y[i] represents the label of the i-th sample in the dataset
    ---- Hints: --------
    * You could use np.linalg.inv(X) to compute the inverse of a matrix X. 
    * You could use @ operator in numpy for matrix multiplication: A@B represents the matrix multiplication between matrices A and B. 
    * You could use X.T to compute the transpose of matrix X. 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def train(self, X, y):
        ##############################
        ## INSERT YOUR CODE HERE (10.0 points)
        self.w = np.linalg.inv(X.T @ X) @ X.T @ y
        ##############################
        
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_3.py -m Linear_Regression_SE_train
        (Mac /Linux): python3 -m pytest -v test_3.py -m Linear_Regression_SE_train
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
'''------------- Class: Linear_Regression_Ridge (10.0 points) -------
    Ridge Regression: This is a child class of the above linear regression class, which use the ridge regression loss (sum of squared error + alpha * |w|^2 ) 
'''
''' ---- Class Properties ----
    * w: the weights of the linear regression model, a numpy float vector of length p. Here w[i] is the weight of the model on the i-th feature.
    * alpha: the weight of the L2 regularization term in ridge regression model, a float scalar.
    '''
class Linear_Regression_Ridge(Linear_Regression):
    #------------- Method: __init__  ------
    ''' Goal: Initialize the model object    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * p: the number of features/measurements in each data sample, an integer scaler
    * alpha: the weight of the L2 regularization term in ridge regression, a float scalar
    '''
    def __init__(self, p, alpha=0.001):
        super(Linear_Regression_Ridge,self).__init__(p)
        self.alpha = alpha
        
        
    #----------------------------------------------------------
    
    #------------- Method: train  ------
    ''' Goal: Given the loss function (sum of squared error + alpha * |w|^2 ), and a training data set, find the optimal values for the parameters (w) of the linear regression model    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * X: the feature matrix of the training samples, a numpy matrix of shape n by p, here X[i,j] is the value of the j-th feature on the i-th training sample
    * y: the labels of the training samples, a numpy float vector of length n, y[i] represents the label of the i-th sample in the dataset
    ---- Hints: --------
    * You could use np.eye() to generate an identity matrix. 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def train(self, X, y):
        ##############################
        ## INSERT YOUR CODE HERE (10.0 points)
        self.w = np.linalg.inv(X.T @ X + self.alpha * np.eye(X.shape[1])) @ X.T @ y
        ##############################
        
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_3.py -m Linear_Regression_Ridge_train
        (Mac /Linux): python3 -m pytest -v test_3.py -m Linear_Regression_Ridge_train
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    

'''-------- TEST problem3.py file: (20 points) ----------
Please type the following command in your terminal to test the correctness of all the above functions in this file:
        (Windows OS): python -m pytest -v test_3.py
        (Mac /Linux): python3 -m pytest -v test_3.py
------------------------------------------------------'''






