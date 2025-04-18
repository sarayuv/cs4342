




#------------ No New Package --------------
# NOTE: Please don't import any new package. You should be able to solve the problems using only the package(s) imported here.
from abc import abstractmethod
import numpy as np
#---------------------------------------------------------


# ---------------------------------------------------------
'''
    Goal of Problem 2: Simple regression models with only one parameter (b) (20 points)
     In this problem, we will build our first regression model using a parametric model with only one parameter (bias or b). When we have only one parameter (the bias) to fit the average/median target values, we are essentially performing a simple form of linear regression. This simple regression model is the easier parametric model to build/train. We can use this model as an example to study the process of how to find optimal values of a parameter in a parametric model by learning from training samples. In this model, the input features are being ignored by the model, a constant number (bias or b) will be used as the output prediction for any test data point, i.e., for any test sample x, the predicted label f(x) = b. Here b is the only parameter of the model, and we want to know what value should we use for the parameter b. Given a training data set of data samples, with their features and labels, we want to figure out the optimal value for our parameter b. We are going to use two different loss functions, (1) sum of absolute errors and (2) sum of squared errors, separately to train the model parameter (b).
    
'''
# ---------------------------------------------------------

'''------------- Class: Simple_Regression (0.0 points) -------
    This is the parent class of simple regression models, which only have one parameter b, and will always predict the value of b during prediction 
'''
''' ---- Class Properties ----
    * b: a float scalar, the only parameter (bias) of the regression model, which is the constant value to be used for label predictions.
    '''
class Simple_Regression:
    #------------- Method: __init__  ------
    ''' Goal: Initialize the model object    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * b: the initial value of the bias parameter, a float scalar.
    '''
    def __init__(self, b=0.0):
        self.b = b
        
        
    #----------------------------------------------------------
    
    #------------- Method: train  ------
    ''' Goal: Given a loss function (in a child class of this class) and a set of training samples, find the optimal value for the parameter b (i.e., self.b)    '''
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
    ''' Goal: Given a set of test instances, predict their labels. Suppose the parameter bias b of the model is given (self.b)    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * Xt: the feature matrix of all testing instances, a numpy matrix of shape n by p. n is the number of test samples, p is the number of features. Xt[i] represents the i-th test sample in the dataset for label prediction
    ---- Outputs: --------
    * yt: the predicted labels of the testing instances, a numpy float vector of length n, y[i] represents the predicted label of the i-th instance in the dataset
    '''
    def predict(self, Xt):
        yt = np.full(Xt.shape[0],self.b) # fill all the predicted values with the bias parameter b
        return yt
        
    #----------------------------------------------------------
    
'''------------- Class: Simple_Regression_Abs (10.0 points) -------
    This is a child class of the above simple regression class, which use the sum of absolute error as the loss function for finding optimal value for parameter b 
'''
''' ---- Class Properties ----
    * b: a float scalar, the only parameter (bias) of the regression model, which is the constant value to be used for label predictions.
    '''
class Simple_Regression_Abs(Simple_Regression):
    #------------- Method: train  ------
    ''' Goal: Given a loss function (sum of absolute errors) and a set of training samples, find the optimal value for the parameter b (i.e., self.b)    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * X: the feature matrix of the training samples, a numpy matrix of shape n by p, here X[i,j] is the value of the j-th feature on the i-th training sample
    * y: the labels of the training samples, a numpy float vector of length n, y[i] represents the label of the i-th sample in the dataset
    ---- Hints: --------
    * Although the input features of training samples (X) are given, you may not need to use it when searching for optimal value of parameter b. 
    * You could use a function in numpy package to find the optimal value of b. 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def train(self, X, y):
        ##############################
        ## INSERT YOUR CODE HERE (10.0 points)
        self.b = np.median(y)
        ##############################
        
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Simple_Regression_Abs_train
        (Mac /Linux): python3 -m pytest -v test_2.py -m Simple_Regression_Abs_train
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
'''------------- Class: Simple_Regression_SE (10.0 points) -------
    This is a child class of the above simple regression class, which use the sum of squared error as the loss function for finding optimal value for parameter b 
'''
''' ---- Class Properties ----
    * b: a float scalar, the only parameter (bias) of the regression model, which is the constant value to be used for label predictions.
    '''
class Simple_Regression_SE(Simple_Regression):
    #------------- Method: train  ------
    ''' Goal: Given a loss function (sum of squared errors) and a set of training samples, find the optimal value for the parameter b (i.e., self.b)    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * X: the feature matrix of the training samples, a numpy matrix of shape n by p, here X[i,j] is the value of the j-th feature on the i-th training sample
    * y: the labels of the training samples, a numpy float vector of length n, y[i] represents the label of the i-th sample in the dataset
    ---- Hints: --------
    * Although the input features of training samples (X) are given, you may not need to use it when searching for optimal value of parameter b. 
    * You could use a function in numpy package to find the optimal value of b. 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def train(self, X, y):
        ##############################
        ## INSERT YOUR CODE HERE (10.0 points)
        self.b = np.mean(y)
        ##############################
        
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Simple_Regression_SE_train
        (Mac /Linux): python3 -m pytest -v test_2.py -m Simple_Regression_SE_train
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    

'''-------- TEST problem2.py file: (20 points) ----------
Please type the following command in your terminal to test the correctness of all the above functions in this file:
        (Windows OS): python -m pytest -v test_2.py
        (Mac /Linux): python3 -m pytest -v test_2.py
------------------------------------------------------'''






