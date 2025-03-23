
'''------------Turn on Word Wrap Setting in Your Editor--------------
    NOTE: For better readability of the instructions, 
          please turn on the 'Word Wrap' setting in your editor. 
    HOW: For example, in the VS Code editor, click "Settings" in the menu, 
         then type "word wrap" in the search box of the settings, 
    choose "on" in the drop-down menu.
    TEST: If you can read this long sentence without scrolling your screen from left to right, it means that your editor's word wrap setting is on and you are good to go. 
------------------------------------------------------------------'''


''' ------------ How to start? Set up Python and install packages --------
    If you are having troubles setting up your python environment, you could try the following steps to install a clean python environment.

    Step 1: Download the latest python installer from https://www.python.org/

    Step 2: Install the python environment by running the downloaded installer. During the installation, you may want to allow the program to set the path in your computer so that this python is your default python environment, which makes it easier for you to use. In windows, you may also want to allow longer path length during the setup process to avoid problems with long path strings of your folder. 

    Step 3: Open a terminal in your computer, and change the directory to the folder of this homework assignment using "cd <PATH>", here replace "<PATH>" with the path of this assignment on your computer. 

    Step 4: In the terminal, check the prompt.    If you can find a "(base)" at the beginning of your terminal prompt, you may have a conda installed in your computer. Please use the following command to disable the conda environment: 
        conda config --set auto_activate_base false
        conda deactivate

    Step 5: Update the Pip (python package manager) in the terminal. 
    Please type the following command in your terminal to update pip:
        (Windows OS): python -m pip install --upgrade pip
        (Mac /Linux): python3 -m pip install --upgrade pip

    Step 6: Install all the packages in a batch. 
    Please type the following command in your terminal to install all the required packages:
        (Windows OS): python -m pip install -r requirements.txt
        (Mac /Linux): python3 -m pip install -r requirements.txt
    
    Now your python environment is all set. 
     ------------------------------------------------'''


''' ------------Test Your Python Environment --------------
    After installing python and packages in your computer, you may want to test if your python environment meets the requirements. If your python version is incorrect or you didn't install a required package successfully, you may not be able to solve this homework assignment correctly.   
    Please type the following command in your terminal to test your python environment:
        (Windows OS): python -m pytest -v test_1.py::test_python_environment
        (Mac /Linux): python3 -m pytest -v test_1.py::test_python_environment
     ------------------------------------------------'''

#------------ No New Package --------------
# NOTE: Please don't import any new package. You should be able to solve the problems using only the package(s) imported here.
from abc import abstractmethod
import numpy as np
#---------------------------------------------------------

#--------------------------
def Terms_and_Conditions():
    ''' 
      By submitting this homework or changing this function, you agree with the following terms:
       (1) Not sharing your code/solution with any student before and after the homework due. For example, sending your code segment to another student, putting your solution online or lending your laptop (if your laptop contains your solution or your Dropbox automatically copied your solution from your desktop computer to your laptop) to another student to work on this homework will violate this term.
       (2) Not using anyone's code in this homework and building your own solution. For example, using some code segments from another student or online resources due to any reason (like too busy recently) will violate this term. Changing other's code as your solution (such as changing the variable names) will also violate this term.
       (3) When discussing with any other students about this homework, only discuss high-level ideas or use pseudo-code. Don't discuss about the solution at the code level. For example, two students discuss about the solution of a function (which needs 5 lines of code to solve) and they then work on the solution "independently", however the code of the two solutions are exactly the same, or only with minor differences (variable names are different). In this case, the two students violate this term.
      All violations of (1),(2) or (3) will be handled in accordance with the WPI Academic Honesty Policy.  For more details, please visit: https://www.wpi.edu/about/policies/academic-integrity/dishonesty
      Note: We may use the Stanford Moss system to check your code for code similarity. https://theory.stanford.edu/~aiken/moss/
      Historical Data: In one year, we ended up finding 25% of the students in that class violating one of the above terms and we handled ALL of these violations according to the WPI Academic Honesty Policy. 
    '''
    #*******************************************
    # CHANGE HERE: if you have read and agree with the term above, change "False" to "True".
    Read_and_Agree = True
    #*******************************************
    return Read_and_Agree
#--------------------------

# ---------------------------------------------------------
'''
    Goal of Problem 1: Linear Models for Binary Classification (Linear Support Vector Machine) (40 points)
     In this problem, you will implement the Support Vector Machines (SVM) method for binary classification.  We will optimize the parameters using Stochastic Gradient Descent (SGD) method. Here we focus on linear SVM, because it is the easiest to understand and implement. The linear SVM implies that the decision boundary between the two classes is a hyperplane in the input feature space. Stochastic Gradient Descent (SGD) is an optimization algorithm often employed to train SVM models. We have used SGD in the previous homework assignment for Lasso regression model. Now let's use SGD algorithm to train another method (linear SVM). Comparing with Lasso regression model, the only major difference is that the computation of parameter gradients, because we are using a different loss function (Hinge Loss + L2 regularization), because we are solving classification problems instead of regression problems..
    
'''
# ---------------------------------------------------------

'''------------- Class: Linear_Classification (0.0 points) -------
    This is the parent class for linear methods for binary classification problems, which has two parameters, weights (w) and bias (b). In this model, we are going to assign a weight to each feature and add a constant bias. The weights w = (w1, w2, ...). Given the featuers of a data sample x = (x1, x2,...), the prediction of the model is computed by the weighted sum of the input feature values, i.e.,  f(x) = b + w1 * x1 + w2 * x2 ... 
'''
''' ---- Class Properties ----
    * w: the weights of the linear model, a numpy float vector of length p. Here w[i] is the weight of the model on the i-th feature.
    * b: the bias of the linear model, a float scalar.
    * neg_code: assuming positive labels are encoded as '1's, this is the negative encoding, a float scalar, which is either '-1' or '0'.
    '''
class Linear_Classification:
    #------------- Method: __init__  ------
    ''' Goal: Initialize the model object    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * p: the number of features/measurements in each data sample, an integer scaler
    * neg_code: assuming positive labels are encoded as '1's, this is the negative encoding, a float scalar, which is either '-1' or '0'
    '''
    def __init__(self, p, neg_code=-1.0):
        self.w = np.random.randn(p) # initialize all weights randomly
        self.b = 0. # initialize the bias as 0s
        self.neg_code = neg_code # initialize the encoding of negative labels (-1 or 0)
        assert neg_code == -1. or neg_code == 0.
        
        
    #----------------------------------------------------------
    
    #------------- Method: compute_fx  ------
    ''' Goal: Compute the f(x) score of the linear model on one data instance x    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: the feature vector of one training data sample, a numpy vector of length p
    ---- Outputs: --------
    * fx: the predicted score of the data instance, a float scalar
    '''
    def compute_fx(self, x):
        fx = x @ self.w + self.b # compute score on a data sample
        return fx
        
    #----------------------------------------------------------
    
    #------------- Method: train  ------
    ''' Goal: Given a loss function (in a child class of this class) and a set of training samples, find the optimal value for the parameter w (i.e., self.w) and bias b (i.e., self.b)    '''
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
    ''' Goal: Given a set of test instances, predict their labels. Suppose the parameter w and b of the model are given (self.w, self.b)    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * Xt: the feature matrix of all testing instances, a numpy matrix of shape n by p. n is the number of test samples, p is the number of features. Xt[i] represents the i-th test sample in the dataset for label prediction
    ---- Outputs: --------
    * yt: the predicted labels of the testing instances, a numpy float vector of length n, y[i] represents the predicted label of the i-th instance in the dataset
    '''
    def predict(self, Xt):
        ft = Xt @ self.w + self.b # compute scores on testing data
        th = 0 if self.neg_code == -1. else 0.5 # compute the threshold on scores for predicting negative class
        yt = np.where(ft >= th, 1, self.neg_code) # thresholding on the scores to convert the scores into label predictions
        return yt
        
    #----------------------------------------------------------
    
'''------------- Class: SGD (0.0 points) -------
    This is the parent class for Stochastic gradient descent algorithms for optimizing parametric models with two parameters, weights (w) and bias (b). 
'''
''' ---- Class Properties ----
    * n_epoch: an integer scalar, the number of passes to iterate through all training examples during stochastic gradient descent..
    * lr: the learning rate of the stochastic gradient descent algorithm, a float scalar, controlling the speed of gradient descent.
    '''
class SGD:
    #------------- Method: __init__  ------
    ''' Goal: Initialize the model object    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * n_epoch: an integer scalar, the number of passes to iterate through all training examples during stochastic gradient descent.
    * lr: the learning rate of the stochastic gradient descent algorithm, a float scalar, controlling the speed of gradient descent
    '''
    def __init__(self, n_epoch=10, lr=0.01):
        self.n_epoch = n_epoch
        self.lr = lr
        
        
    #----------------------------------------------------------
    
    #------------- Method: train  ------
    ''' Goal: Given a set of training samples, find the optimal value for the parameters w and b    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * X: the feature matrix of the training samples, a numpy matrix of shape n by p, here X[i,j] is the value of the j-th feature on the i-th training sample
    * y: the labels of the training samples, a numpy float vector of length n, y[i] represents the label of the i-th sample in the dataset
    '''
    @abstractmethod
    def train(self, X, y):
        pass
        
        
    #----------------------------------------------------------
    
    #------------- Method: update_w  ------
    ''' Goal: Given the gradient of the parameter w, update the values of w using one step of gradient descent    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * dL_dw: the gradient of the weights, a numpy float vector of length p. The i-th element (d L / d w[i]) is the partial gradient of loss function with respect to the i-th weight parameter w[i] 
    '''
    def update_w(self, dL_dw):
        self.w -= self.lr * dL_dw
        
        
    #----------------------------------------------------------
    
    #------------- Method: update_b  ------
    ''' Goal: Given the gradient of the parameter b, update the value of b using one step of gradient descent    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * dL_db: the gradient of the bias parameter, a numpy float scalar. 
    '''
    def update_b(self, dL_db):
        self.b -= self.lr * dL_db
        
        
    #----------------------------------------------------------
    
'''------------- Class: Linear_SVM (40.0 points) -------
    Linear SVM: This is a child class of the above linear classification class. Linear SVM uses the sum of hinge loss and L2 regularization as the loss function for finding optimal value for parameters w and b 
'''
''' ---- Class Properties ----
    * w: the weights of the linear model, a numpy float vector of length p. Here w[i] is the weight of the model on the i-th feature.
    * b: the bias of the linear model, a float scalar.
    * C: the weight of the hinge loss, a positive float scalar.
    * n_epoch: an integer scalar, the number of passes to iterate through all training examples during stochastic gradient descent..
    * lr: the learning rate of the stochastic gradient descent algorithm, a float scalar, controlling the speed of gradient descent.
    '''
class Linear_SVM(Linear_Classification,SGD):
    #------------- Method: __init__  ------
    ''' Goal: Initialize the model object    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * p: the number of features/measurements in each data sample, an integer scaler
    * C: the weight of the hinge loss in the loss function, a positive float scalar
    * n_epoch: an integer scalar, the number of passes to iterate through all training examples during stochastic gradient descent.
    * lr: the learning rate of the stochastic gradient descent algorithm, a float scalar, controlling the speed of gradient descent
    '''
    def __init__(self, p, C=1.0, n_epoch=10, lr=0.01):
        Linear_Classification.__init__(self,p=p)
        SGD.__init__(self,n_epoch=n_epoch,lr=lr)
        self.C = C
        
        
    #----------------------------------------------------------
    
    #------------- Method: compute_gradient  ------
    ''' Goal: Given a training data sample (x, y), compute the gradients of the parameters w    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: the feature vector of one training data sample, a numpy vector of length p
    * y: the label of the one training data sample, a float scalar
    * l: short for lambda = 1/ (n C), which is the weight of the L2 regularization term in the loss function, a float scalar. Here n is the number of data samples in the training dataset.
    ---- Outputs: --------
    * dL_dw: the gradient of the weights, a numpy float vector of length p. The i-th element (d L / d w[i]) is the partial gradient of loss function with respect to the i-th weight parameter w[i] 
    * dL_db: the gradient of the bias, a float scalar.
    ---- Hints: --------
    * The parameters of the linear model are accessable through self.w and self.b. 
    * You could use self.compute_fx() function to compute the score f(x) of the linear model on a data sample x. This function was implemented in the parent class (Linear_Classification) . 
    * This problem can be solved using only 7 line(s) of code. More lines are okay.'''
    def compute_gradient(self, x, y, l):
        ##############################
        ## INSERT YOUR CODE HERE (20.0 points)
        fx = self.compute_fx(x)
        if y * fx >= 1:
            dL_dw = l * self.w
            dL_db = 0.0
        else:
            dL_dw = l * self.w - y * x
            dL_db = -y
        ##############################
        return dL_dw, dL_db
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m Linear_SVM_compute_gradient
        (Mac /Linux): python3 -m pytest -v test_1.py -m Linear_SVM_compute_gradient
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: train  ------
    ''' Goal: Given a set of training samples, find the optimal value for the parameters w and b    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * X: the feature matrix of the training samples, a numpy matrix of shape n by p, here X[i,j] is the value of the j-th feature on the i-th training sample
    * y: the labels of the training samples, a numpy float vector of length n, y[i] represents the label of the i-th sample in the dataset
    ---- Hints: --------
    * You could use the functions that you have implemented above to build the solution.. 
    * This problem can be solved using only 3 line(s) of code. More lines are okay.'''
    def train(self, X, y):
        n = X.shape[0] # n is the number of samples
        l = 1./(n * self.C) # l is the weight of the L2 regularization term. 
        for _ in range(self.n_epoch): # iterate through the dataset n_epoch times
            indices = np.random.permutation(n) # shuffle the indices of all instances in the training set
            for i in indices: # iterate through each random instance (x,y)
                xi = X[i] # the feature vector of the i-th random instance
                yi = y[i] # the label of the i-th random instance
                ##############################
                ## INSERT YOUR CODE HERE (20.0 points)
                dL_dw, dL_db = self.compute_gradient(xi, yi, l)
                self.update_w(dL_dw)
                self.update_b(dL_db)
                ##############################
        
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m Linear_SVM_train
        (Mac /Linux): python3 -m pytest -v test_1.py -m Linear_SVM_train
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    

'''-------- TEST problem1.py file: (40 points) ----------
Please type the following command in your terminal to test the correctness of all the above functions in this file:
        (Windows OS): python -m pytest -v test_1.py
        (Mac /Linux): python3 -m pytest -v test_1.py
------------------------------------------------------'''






