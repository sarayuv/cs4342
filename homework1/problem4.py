




#------------ No New Package --------------
# NOTE: Please don't import any new package. You should be able to solve the problems using only the package(s) imported here.
import numpy as np
from problem3 import Linear_Regression
#---------------------------------------------------------


# ---------------------------------------------------------
'''
    Goal of Problem 4: Lasso Regression Method and Stochastic Gradient Descent (40 points)
    In this problem, we will implement a linear regression model with L1 regularization. This method is called Lasso regression, short for 'Least Absolute Shrinkage and Selection Operator' regression, is a linear regression technique that incorporates L1 regularization. Lasso regression adds a penalty term to the standard linear regression cost function, which includes the sum of squared errors. The lasso penalty is proportional to the absolute values of the regression coefficients (or weights w). Given a training data set of data samples, with their features and labels, we want to figure out the optimal values of the parameters in (w). We are going to use stochastic gradient descent to find the optimal parameter values on a training data set..
    
'''
# ---------------------------------------------------------

'''------------- Class: Lasso_Regression (40.0 points) -------
    Lasso Regression: This is a child class of the linear regression class in problem 3, which uses the sum of squared error as the loss function and lasso regularization for finding optimal value for parameter w. The overall loss of the parameters is  (sum of squared error + alpha * |w| ), here |w| stands for the sum of absolute values of all the elements in the weight vector w 
'''
''' ---- Class Properties ----
    * w: the weights of the linear regression model, a numpy float vector of length p. Here w[i] is the weight of the model on the i-th feature.
    * alpha: the weight of the L1 regularization term in lasso regression model, a float scalar.
    * n_epoch: an integer scalar, the number of passes to iterate through all training examples during stochastic gradient descent..
    * lr: the learning rate of the stochastic gradient descent algorithm, a float scalar, controlling the speed of gradient descent.
    '''
class Lasso_Regression(Linear_Regression):
    #------------- Method: __init__  ------
    ''' Goal: Initialize the model object    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * p: the number of features/measurements in each data sample, an integer scaler
    * alpha: the weight of the L1 regularization term in ridge regression, a float scalar
    * n_epoch: an integer scalar, the number of passes to iterate through all training examples during stochastic gradient descent.
    * lr: the learning rate of the stochastic gradient descent algorithm, a float scalar, controlling the speed of gradient descent
    '''
    def __init__(self, p, alpha=0.001, n_epoch=100, lr=0.01):
        super(Lasso_Regression,self).__init__(p)
        self.alpha = alpha
        self.n_epoch = n_epoch
        self.lr = lr
        
        
    #----------------------------------------------------------
    
    #------------- Method: compute_gradient  ------
    ''' Goal: Given a training data sample (x, y), compute the gradients of the parameters w    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: the feature vector of one training data sample, a numpy vector of length p
    * y: the label of the one training data sample, a float scalar
    ---- Outputs: --------
    * dL_dw: the gradient of the weights, a numpy float vector of length p. The i-th element (d L / d w[i]) is the partial gradient of loss function with respect to the i-th weight parameter w[i] 
    ---- Hints: --------
    * The weight parameters of the linear model are accessable through self.w. 
    * You could use np.sign(.) function to compute the sign of a value or the signs of the elements of a vector. 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_gradient(self, x, y):
        ##############################
        ## INSERT YOUR CODE HERE (20.0 points)
        dL_dw = - (y - np.dot(self.w, x)) * x + self.alpha * np.sign(self.w)
        ##############################
        return dL_dw
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_4.py -m Lasso_Regression_compute_gradient
        (Mac /Linux): python3 -m pytest -v test_4.py -m Lasso_Regression_compute_gradient
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: update_w  ------
    ''' Goal: Given the gradient of the parameter w, update the values of w using one step of gradient descent    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * dL_dw: the gradient of the weights, a numpy float vector of length p. The i-th element (d L / d w[i]) is the partial gradient of loss function with respect to the i-th weight parameter w[i] 
    ---- Hints: --------
    * The weight parameters of the linear model are accessable through self.w. 
    * The learning rate of the gradient descent is accessable through self.lr. 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def update_w(self, dL_dw):
        ##############################
        ## INSERT YOUR CODE HERE (8.0 points)
        self.w -= self.lr * dL_dw
        ##############################
        
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_4.py -m Lasso_Regression_update_w
        (Mac /Linux): python3 -m pytest -v test_4.py -m Lasso_Regression_update_w
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: train  ------
    ''' Goal: Given a set of training samples, find the optimal value for the parameter w (i.e., self.w) using stochastic gradient descent algorithm    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * X: the feature matrix of the training samples, a numpy matrix of shape n by p, here X[i,j] is the value of the j-th feature on the i-th training sample
    * y: the labels of the training samples, a numpy float vector of length n, y[i] represents the label of the i-th sample in the dataset
    ---- Hints: --------
    * You could use the functions that you have implemented above to build the solution of this function. 
    * This problem can be solved using only 2 line(s) of code. More lines are okay.'''
    def train(self, X, y):
        n = len(y) # n is the number of samples
        for _ in range(self.n_epoch): # iterate through the training set n_epoch times
            indices = np.random.permutation(n) # shuffle the indices of all instances
            for i in indices: # iterate through each random instance (xi,yi)
                xi=X[i] # the feature vector of the i-th random instance
                yi=y[i] # the label of the i-th random instance
                ##############################
                ## INSERT YOUR CODE HERE (12.0 points)
                dL_dw = self.compute_gradient(xi, yi)
                self.update_w(dL_dw)
                ##############################
        
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_4.py -m Lasso_Regression_train
        (Mac /Linux): python3 -m pytest -v test_4.py -m Lasso_Regression_train
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    

'''-------- TEST problem4.py file: (40 points) ----------
Please type the following command in your terminal to test the correctness of all the above functions in this file:
        (Windows OS): python -m pytest -v test_4.py
        (Mac /Linux): python3 -m pytest -v test_4.py
------------------------------------------------------'''

'''---------- TEST ALL problem files in this HW assignment (100 points) ---------
 This is the last problem file in this homework assignment. 
Please type the following command in your terminal to test the correctness of all the problem files:
        (Windows OS): python -m pytest -v
        (Mac /Linux): python3 -m pytest -v
---------------------------------------------------'''

'''-------- Automatic Grading of This HW Assignment -------
Please type the following command in your terminal to compute your score of this HW assignment:
        (Windows OS): python grading.py
        (Mac /Linux): python3 grading.py
 The grading.py will run all the unit tests of this HW assignment and compute the scores you get. 
 For example, if your code for this HW can get 95 points, you will see this message at the end in the terminal
 ****************************
 ** Total Points: 95 / 100 ** (this is just an example, you need to run the grading.py to know your grade)
 ****************************

 NOTE: Due to the randomness of the test data and/or initialization of parameters, the results of the same unit test may vary in different runs. If your code could pass a test case with more than 80% probability, you won't lose points in that test case. If you lose points after the grading by the TA due to randomness of the testing, you could contact the TA to show that your code could pass that test case with more than 80% chance, and get the lost points back.

-------------------------------------------------
***   How to submit your work?  ***

After running the above grading script, a zip file named "submission.zip" will be created in the same folder of this homework assignment. Please upload this "submission.zip" file in canvas for your final submission. 
NOTE: Please only use the "submission.zip" generated by the grading script as your submision. Don't create a zip file yourself by including all the data files and folder structures in the zip. Because the TA will use an automatic script to grade all the submissions. The script requires all submissions to have a standard format and folder structure. If you create your own zip file, you may be using a different format or name in the submission, which may cause errors in the grading of your submission. Thanks a lot for your contribution to streamlining the grading of homework assignments.

 That's all! Great job! You did it!
----------------------------------------------------'''




