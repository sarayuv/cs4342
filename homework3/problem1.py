
'''------------Turn on Word Wrap Setting in Your Editor--------------
    NOTE: For better readability of the instructions, 
          please turn on the 'Word Wrap' setting in your editor. 
    HOW: For example, in the VS Code editor, click "Settings" in the menu, 
         then type "word wrap" in the search box of the settings, 
    choose "on" in the     git remote add origin <url>drop-down menu.
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
    Goal of Problem 1: Linear Models for Multi-Class Classification (Softmax Regression) (40 points)
     In this problem, you will implement the Softmax Regression (SR) method for multi-class classification. The main goal of this problem is to extend the logistic regression method to solving multi-class classification problems.  We will get familiar with computing gradients of vectors/matrices.  We will use multi-class cross entropy as the loss function and stochastic gradient descent to train the model parameters. For simplicity, in this model, we don't include any regularization term in the loss function..
    
'''
# ---------------------------------------------------------

'''------------- Class: Multiclass_Classification (0.0 points) -------
    This is the parent class for multiclass classification methods 
'''
''' ---- Class Properties ----
    * p: the number of input features, an integer scalar.
    * c: the number of classes in the classification task, an integer scalar.
    '''
class Multiclass_Classification:
    #------------- Method: __init__  ------
    ''' Goal: Initialize the model object    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * p: the number of features/measurements in each data sample, an integer scaler
    * c: the number of classes in the classification task, an integer scalar
    '''
    def __init__(self, p, c):
        self.p = p
        self.c = c
        
        
    #----------------------------------------------------------
    
    #------------- Method: forward  ------
    ''' Goal: (Forward Pass) Suppose we are given a multi-class classification model with parameters. Given a data sample (x), please compute the activations a(x) on the sample    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: the feature vector of one training data sample, a numpy vector of length p
    '''
    @abstractmethod
    def forward(self, x):
        pass
        
        
    #----------------------------------------------------------
    
    #------------- Method: predict  ------
    ''' Goal: Given a set of test instances, predict their labels.    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * Xt: the feature matrix of all testing instances, a numpy matrix of shape n by p. n is the number of test samples, p is the number of features. Xt[i] represents the i-th test sample in the dataset for label prediction
    '''
    @abstractmethod
    def predict(self, Xt):
        pass
        
        
    #----------------------------------------------------------
    
'''------------- Class: SGD (0.0 points) -------
    This is the parent class for Stochastic gradient descent algorithms for optimizing parametric models. 
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
    
    #------------- Method: forward  ------
    ''' Goal: (Forward Pass) Suppose we are given a parametric model. Given a data sample (x), please compute the activations a(x) on the sample    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: the feature vector of one data sample, a numpy vector of length p
    '''
    @abstractmethod
    def forward(self, x):
        pass
        
        
    #----------------------------------------------------------
    
    #------------- Method: backward  ------
    ''' Goal: (Back Propagation) Suppose we are given a softmax regression model with parameters (W and b) and we have a training data sample (x) with label (y).  Suppose we have already computed the activations a(x) on the data sample in the forward-pass. Please compute the global gradients of the loss w.r.t. the parameters on the data sample using back propagation    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: the feature vector of one data sample, a numpy vector of length p
    * y: the label of one data sample, an integer scalar value. The values can be 0,1,2, ..., or (c-1)
    * a: the softmax activations, a float numpy vector of length c
    '''
    @abstractmethod
    def backward(self, x, y, a):
        pass
        
        
    #----------------------------------------------------------
    
    #------------- Method: train  ------
    ''' Goal: Given a loss function (in a child class of this class) and a set of training samples, find the optimal value for the parameters    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * X: the feature matrix of the training samples, a numpy matrix of shape n by p, here X[i,j] is the value of the j-th feature on the i-th training sample
    * y: the labels of training samples, a numpy float vector of length n, y[i] represents the label of the i-th sample in the dataset
    '''
    @abstractmethod
    def train(self, X, y):
        pass
        
        
    #----------------------------------------------------------
    
'''------------- Class: Softmax_Regression (40.0 points) -------
    Softmax regression, also known as multinomial logistic regression or maximum entropy classifier, is a method used for multiclass classification. It is an extension of logistic regression that allows for the classification of instances into more than two classes. The softmax function, also called the normalized exponential function, is used to transform a vector of real-valued scores into a probability distribution over multiple classes. 
'''
''' ---- Class Properties ----
    * p: the number of input features, an integer scalar.
    * c: the number of classes in the classification task, an integer scalar.
    * W: the weight matrix of softmax regression, a float numpy matrix of shape (c by p).
    * b: the bias values of softmax regression, a float numpy vector of length c.
    * n_epoch: an integer scalar, the number of passes to iterate through all training examples during stochastic gradient descent..
    * lr: the learning rate of the stochastic gradient descent algorithm, a float scalar, controlling the speed of gradient descent.
    '''
class Softmax_Regression(Multiclass_Classification,SGD):
    #------------- Method: __init__  ------
    ''' Goal: Initialize the model object    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * p: the number of features/measurements in each data sample, an integer scaler
    * c: the number of classes in the classification task, an integer scalar
    * n_epoch: an integer scalar, the number of passes to iterate through all training examples during stochastic gradient descent.
    * lr: also called alpha, the learning rate of the stochastic gradient descent algorithm, a float scalar, controlling the speed of gradient descent. Note in the slides we use the notation 'alpha' to refer to this value
    '''
    def __init__(self, p, c, n_epoch=100, lr=0.001):
        Multiclass_Classification.__init__(self,p=p,c=c)
        SGD.__init__(self,n_epoch=n_epoch,lr=lr)
        self.W = np.random.randn(c,p) # initialize W randomly using standard normal distribution
        self.b= np.zeros(c) # initialize b as all zeros
        
        
    #----------------------------------------------------------
    
    #------------- Method: compute_z  ------
    ''' Goal: (Value Forward Function 1) Given a softmax regression model with parameters W and b, please compute the linear logit z(x) on a data sample x    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: the feature vector of one data sample, a numpy vector of length p
    ---- Outputs: --------
    * z: the linear logit values on the data sample x, i.e. z(x), a float numpy vector of length c
    ---- Hints: --------
    * You could access the model parameters W and b through self variable in the input of this function.. 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_z(self, x):
        ##############################
        ## INSERT YOUR CODE HERE (4.0 points)
        z = np.dot(self.W, x) + self.b
        ##############################
        return z
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m Softmax_Regression_compute_z
        (Mac /Linux): python3 -m pytest -v test_1.py -m Softmax_Regression_compute_z
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_dz_db  ------
    ''' Goal: (Local Gradient 1.1) Suppose we are given a softmax regression model with parameters W and b. Suppose we have already computed the linear logits z(x) on a training sample x.  Please compute partial gradients of the linear logits z(x) w.r.t. the biases b    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    ---- Outputs: --------
    * dz_db: the partial gradient of the logits z w.r.t. the biases b, a float matrix of shape c by c.  Each element dz_db[i,j] represents the partial gradient of the i-th logit z[i] w.r.t. the j-th bias b[j],   d_z[i] / d_b[j]
    ---- Hints: --------
    * Why there is no input variable in this function? Maybe you don't need one, because dz_db is a constant matrix. You don't need any input to compute this gradient. 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_dz_db(self):
        ##############################
        ## INSERT YOUR CODE HERE (2.0 points)
        dz_db = np.eye(self.c)
        ##############################
        return dz_db
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m Softmax_Regression_compute_dz_db
        (Mac /Linux): python3 -m pytest -v test_1.py -m Softmax_Regression_compute_dz_db
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: check_dz_db  ------
    ''' Goal: Gradient Checking: compute the local gradient of the logit function using gradient check    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: the feature vector of one data sample, a numpy vector of length p
    * delta: a small number for gradient check, a float scalar
    ---- Outputs: --------
    * dz_db: the approximated local gradient of the logits w.r.t. the biases using gradient check, a float vector of length c. Each element dz_db[i,j] represents the partial gradient of the i-th logit z[i] w.r.t. the j-th bias:  d_z[i] / d_b[j]
    '''
    def check_dz_db(self, x, delta=1e-7):
        dz_db = np.zeros((self.c,self.c))
        for i in range(self.c):
            for j in range(self.c):
                d = np.zeros(self.c) 
                d[j] = delta
                v0 = self.compute_z(x)[i]
                self.b+=d
                v1 = self.compute_z(x)[i]
                self.b-=d
                dz_db[i,j] = (v1-v0) / delta
        return dz_db
        
    #----------------------------------------------------------
    
    #------------- Method: compute_dL_db  ------
    ''' Goal: (Global Gradients 1.1) Suppose we are given a softmax regression model with parameters (W and b) and we have a training data sample (x,y).  Suppose we have already computed the local gradient 1.1 (dz_db) and the global gradients of the loss L w.r.t. the linear logits z(x) (dL_dz). Please compute the partial gradients of the loss L w.r.t. biases b using chain rule    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * dL_dz: the partial gradients of the loss function L w.r.t. the linear logits z, a float numpy vector of length c.  The i-th element dL_dz[i] represents the partial gradient of the loss function L w.r.t. the i-th linear logit z[i],  d_L / d_z[i]
    * dz_db: the partial gradient of the logits z w.r.t. the biases b, a float matrix of shape c by c.  Each element dz_db[i,j] represents the partial gradient of the i-th logit z[i] w.r.t. the j-th bias b[j],   d_z[i] / d_b[j]
    ---- Outputs: --------
    * dL_db: the partial gradient of the loss function L w.r.t. the biases b, a float numpy vector of length c.  The i-th element dL_db[i] represents the partial gradient of the loss function w.r.t. the i-th bias,  d_L / d_b[i]
    ---- Hints: --------
    * Why there is no input variable in this function? Maybe you don't need one, because dz_db is a constant matrix. You don't need any input to compute this gradient. 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_dL_db(self, dL_dz, dz_db):
        ##############################
        ## INSERT YOUR CODE HERE (2.0 points)
        dL_db = np.dot(dL_dz, dz_db)
        ##############################
        return dL_db
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m Softmax_Regression_compute_dL_db
        (Mac /Linux): python3 -m pytest -v test_1.py -m Softmax_Regression_compute_dL_db
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_dz_dW  ------
    ''' Goal: (Local Gradient 1.2) Suppose we are given a softmax regression model with parameters W and b. Suppose we have already computed the linear logits z(x) on a training sample x.  Please compute partial gradients of the linear logits z(x) w.r.t. the weights W     '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: the feature vector of one training data sample, a numpy vector of length p
    ---- Outputs: --------
    * dz_dW: the partial gradient of logits z w.r.t. the weight matrix W, a numpy float tensor of shape (c by c by p).  The (i,j,k)-th element of dz_dW represents the partial gradient of the i-th logit (z[i]) w.r.t. the weight W[j,k],   d_z[i] / d_W[j,k]
    ---- Hints: --------
    * This problem can be solved using only 3 line(s) of code. More lines are okay.'''
    def compute_dz_dW(self, x):
        ##############################
        ## INSERT YOUR CODE HERE (4.0 points)
        dz_dW = np.zeros((self.c, self.c, self.p))
        for i in range(self.c):
            for j in range(self.c):
                if i == j:
                    dz_dW[i, j, :] = x
        ##############################
        return dz_dW
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m Softmax_Regression_compute_dz_dW
        (Mac /Linux): python3 -m pytest -v test_1.py -m Softmax_Regression_compute_dz_dW
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: check_dz_dW  ------
    ''' Goal: Gradient Checking: compute the local gradient of the logit function using gradient check    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: the feature vector of one data sample, a numpy vector of length p
    * delta: a small number for gradient check, a float scalar
    ---- Outputs: --------
    * dz_dW: the approximated local gradient of the logits w.r.t. the weight matrix computed by gradient checking, a numpy float matrix of shape (c by c by p). The i,j,k -th element of dz_dW represents the partial gradient of the i-th logit (z[i]) w.r.t. the weight W[j,k]:   d_z[i] / d_W[j,k]
    '''
    def check_dz_dW(self, x, delta=1e-7):
        dz_dW = np.zeros((self.c,self.c,self.p))
        for i in range(self.c):
            for j in range(self.c):
                for k in range(self.p):
                    d = np.zeros((self.c,self.p))
                    d[j,k] = delta
                    v0 = self.compute_z(x)[i]
                    self.W+=d
                    v1 = self.compute_z(x)[i]
                    self.W-=d
                    dz_dW[i,j,k] = (v1-v0) / delta
        return dz_dW
        
    #----------------------------------------------------------
    
    #------------- Method: compute_dL_dW  ------
    ''' Goal: (Global Gradient 1.2) Suppose we are given a softmax regression model with parameters (W and b) and we have a training data sample (x,y).  Suppose we have already computed the local gradients local gradient 1.2 (dz_dW) and the global gradients of the loss L w.r.t. the linear logits z(x) (dL_dz). Please compute the partial gradient of the loss L w.r.t. the weights W using chain rule    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * dL_dz: the partial gradients of the loss function L w.r.t. the linear logits z, a float numpy
    * dz_dW: the partial gradient of logits z w.r.t. the weight matrix W, a numpy float tensor of shape (c by c by p).  The (i,j,k)-th element of dz_dW represents the partial gradient of the i-th logit (z[i]) w.r.t. the weight W[j,k],   d_z[i] / d_W[j,k]
    ---- Outputs: --------
    * dL_dW: the partial gradients of the loss function L w.r.t. the weight matrix W, a numpy float matrix of shape (c by p).  The i,j-th element dL_dW[i,j] represents the partial gradient of the loss function L w.r.t. the i,j-th weight W[i,j],  d_L / d_W[i,j]
    ---- Hints: --------
    * You could use np.tensordot(A,B) to compute the dot product of two tensors A and B. 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_dL_dW(self, dL_dz, dz_dW):
        ##############################
        ## INSERT YOUR CODE HERE (2.0 points)
        dL_dW = np.tensordot(dL_dz, dz_dW, axes=([0], [0]))
        ##############################
        return dL_dW
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m Softmax_Regression_compute_dL_dW
        (Mac /Linux): python3 -m pytest -v test_1.py -m Softmax_Regression_compute_dL_dW
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_a  ------
    ''' Goal: (Value Forward Function 2) Suppose we are given a softmax regression model and we have already computed the linear logits z(x) on a data sample x. Please compute the softmax activation on the data sample, i.e., a(x)    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * z: the linear logits, a float numpy vector of length c
    ---- Outputs: --------
    * a: the softmax activations, a float numpy vector of length c
    ---- Hints: --------
    * You could use np.exp(x) to compute the element-wise exponentials of vector x. 
    * When computing exp(z), you need to be careful about overflowing cases. When an element of z (say z[i]) is a large number (say 1000),  the computer can no longer store the result of exp(z[i]) in a floating-point number. In this case, we may want to avoid computing exp(z) directly. Instead, you could find the largest value in z (say max_z) and subtract every element with max_z and then you could compute exp() on the vector (z-max_z) directly. The result will be correct, but will no longer suffer from overflow problems. 
    * This problem can be solved using only 2 line(s) of code. More lines are okay.'''
    def compute_a(self, z):
        ##############################
        ## INSERT YOUR CODE HERE (2.0 points)
        max_z = np.max(z)
        exp_z = np.exp(z - max_z)
        a = exp_z / np.sum(exp_z)
        return a
        ##############################
        return a
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m Softmax_Regression_compute_a
        (Mac /Linux): python3 -m pytest -v test_1.py -m Softmax_Regression_compute_a
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_da_dz  ------
    ''' Goal: (Local Gradients 2) Suppose we are given a softmax regression model and we have already computed the linear logits z(x) and activations a(x) on a training sample (x). Please compute the partial gradients of the softmax activations a(x) w.r.t. the linear logits z(x)    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * a: the softmax activations, a float numpy vector of length c
    ---- Outputs: --------
    * da_dz: the partial gradient of the activations a w.r.t. the logits z, a float numpy matrix of shape (c by c).  The (i,j)-th element of da_dz represents the partial gradient ( d_a[i]  / d_z[j] )
    ---- Hints: --------
    * This problem can be solved using only 3 line(s) of code. More lines are okay.'''
    def compute_da_dz(self, a):
        ##############################
        ## INSERT YOUR CODE HERE (2.0 points)
        da_dz = np.diag(a) - np.outer(a, a)
        ##############################
        return da_dz
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m Softmax_Regression_compute_da_dz
        (Mac /Linux): python3 -m pytest -v test_1.py -m Softmax_Regression_compute_da_dz
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: check_da_dz  ------
    ''' Goal: Gradient Checking: Compute local gradient of the softmax function using gradient checking    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * z: the logit values of softmax regression, a float numpy vector of length c. Here c is the number of classes
    * delta: a small number for gradient check, a float scalar
    ---- Outputs: --------
    * da_dz: the approximated local gradient of the activations w.r.t. the logits, a float numpy matrix of shape (c by c)
    '''
    def check_da_dz(self, z, delta=1e-7):
        da_dz = np.zeros((self.c,self.c))
        for i in range(self.c):
            for j in range(self.c):
                d = np.zeros(self.c)
                d[j] = delta
                v0 = self.compute_a(z)[i]
                v1 = self.compute_a(z+d)[i]
                da_dz[i,j] = (v1-v0) / delta
        return da_dz
        
    #----------------------------------------------------------
    
    #------------- Method: compute_dL_dz  ------
    ''' Goal: (Global Gradients 2) Suppose we are given a softmax regression model with parameters (W and b) and we have a training data sample (x,y).  Suppose we have already computed the gradient of the loss L w.r.t. the activations a(x) and the partial gradients of activations a(x) w.r.t. the linear logits z(x). Please compute the partial gradients of the loss L w.r.t. the linear logits z(x) using chain rule    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * dL_da: the partial gradients of the loss function L w.r.t. the activations a, a float numpy vector of length c.  The i-th element dL_da[i] represents the partial gradient of the loss function L w.r.t. the i-th activation a[i],  d_L / d_a[i]
    * da_dz: the partial gradient of the activations a w.r.t. the logits z, a float numpy matrix of shape (c by c).  The (i,j)-th element of da_dz represents the partial gradient ( d_a[i]  / d_z[j] )
    ---- Outputs: --------
    * dL_dz: the partial gradients of the loss function L w.r.t. the linear logits z, a float numpy vector of length c.  The i-th element dL_dz[i] represents the partial gradient of the loss function L w.r.t. the i-th linear logit z[i],  d_L / d_z[i]
    ---- Hints: --------
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_dL_dz(self, dL_da, da_dz):
        ##############################
        ## INSERT YOUR CODE HERE (2.0 points)
        dL_dz = np.dot(dL_da, da_dz)
        ##############################
        return dL_dz
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m Softmax_Regression_compute_dL_dz
        (Mac /Linux): python3 -m pytest -v test_1.py -m Softmax_Regression_compute_dL_dz
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_L  ------
    ''' Goal: (Value Forward Function 3) Suppose we are given a softmax regression model and we have already computed the activations a(x) on a training sample x. Suppose the label of the training sample is y. Please compute the loss function of the softmax regression model on the training sample    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * a: the softmax activations, a float numpy vector of length c
    * y: the label of one data sample, an integer scalar value. The values can be 0,1,2, ..., or (c-1)
    ---- Outputs: --------
    * L: the multi-class cross entropy loss, a float scalar
    ---- Hints: --------
    * You could use np.log(x) to compute the natural log of x. 
    * When computing log(a[i]), you need to be careful about a corner case where log(0) is not defined in math. Now the question is how can any activation (a) become 0? It is mathematically impossible, because the output of the softmax function (activation) should be 0<a[i]<1. However, in the above function (compute_a), when some linear logit z[i] is a number much larger than all the other elements in z (say z[j]), the activation of all the other elements (a[j]) will be very very small. Then the computer can no longer store these small numbers accurately in floating-point numbers. Instead, computer will store 0 as the activation a[i]. Then we have a problem in this function. We need to handle the specially case when a[j] = 0. To solve this problem, we need to avoid computing log(0) by assigning the final result of L directly. In this case, the log(a[j]) should be a very large negative number (say -10000...000 ), though it should not be negative infinity. So the loss L = -log(a[j]) should be a very large positive number. 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_L(self, a, y):
        ##############################
        ## INSERT YOUR CODE HERE (2.0 points)
        L = -np.log(a[y]) if a[y] > 0 else 10000000000
        ##############################
        return L
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m Softmax_Regression_compute_L
        (Mac /Linux): python3 -m pytest -v test_1.py -m Softmax_Regression_compute_L
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_dL_da  ------
    ''' Goal: (Local/Global Gradient 3) Suppose we are given a softmax regression model and we have already computed the activations a(x) on a training sample x. Suppose the label of the training sample is y. Please compute the partial gradients of the loss function (L) w.r.t. the activations (a)    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * a: the softmax activations, a float numpy vector of length c
    * y: the label of one data sample, an integer scalar value. The values can be 0,1,2, ..., or (c-1)
    ---- Outputs: --------
    * dL_da: the partial gradients of the loss function L w.r.t. the activations a, a float numpy vector of length c.  The i-th element dL_da[i] represents the partial gradient of the loss function L w.r.t. the i-th activation a[i],  d_L / d_a[i]
    ---- Hints: --------
    * If you want to create an all-zero array of the same shape as x, you could use np.zeros_like(x) to create the all-zero matrix. 
    * When computing 1/a[i], you need to be careful about a corner case where 1/0 is not defined in math. Now the question is how can any of the activations (a) become 0? It is mathematically impossible, because the output of the softmax function (activations) should be 0<a[i]<1. However, in the compute_a() function, when some element of linear logits z (say z[i]) is much larger than all the other linear logits (say z[j]), the activations of all the other elements (a[j]) will be very very small. Then the computer can no longer store these small numbers accurately in floating-point numbers. Instead, computer will store 0 as the activation. Then we have a problem in this function. We need to handle the specially case when a[j] = 0.  To solve this problem, we need to avoid computing 1/a[j] by assigning the final result of 1/a[j] directly. In this case, when a[j] is a very small positive number (say exp(-900)), then 1/a[j] should be a very large positive number, though it should not be positive infinity. In this case, (-1/a[j]) will be a very large negative number (say -100000...000), though it should not be negative infinity. 
    * This problem can be solved using only 2 line(s) of code. More lines are okay.'''
    def compute_dL_da(self, a, y):
        ##############################
        ## INSERT YOUR CODE HERE (2.0 points)
        dL_da = np.zeros_like(a)
        dL_da[y] = -1 / max(a[y], 1e-10)
        ##############################
        return dL_da
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m Softmax_Regression_compute_dL_da
        (Mac /Linux): python3 -m pytest -v test_1.py -m Softmax_Regression_compute_dL_da
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: check_dL_da  ------
    ''' Goal: Gradient Checking: Compute local gradient of the softmax function using gradient checking    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * a: the softmax activations, a float numpy vector of length c
    * y: the label of a training instance, an integer scalar value. The values can be 0,1,2, ..., or (c-1)
    * delta: a small number for gradient check, a float scalar
    ---- Outputs: --------
    * dL_da: the approximated local gradients of the loss function w.r.t. the activations, a float numpy vector of length c.
    '''
    def check_dL_da(self, a, y, delta=1e-7):
        dL_da = np.zeros(self.c) # initialize the vector as all zeros
        for i in range(self.c):
            d = np.zeros(self.c)
            d[i] = delta
            dL_da[i] = ( self.compute_L(a+d,y) - self.compute_L(a,y)) / delta
        return dL_da
        
    #----------------------------------------------------------
    
    #------------- Method: forward  ------
    ''' Goal: (Forward Pass) Suppose we are given a softmax regression model with parameter W and b. Given a data sample (x), please compute the activations a(x) on the sample    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: the feature vector of one data sample, a numpy vector of length p
    ---- Outputs: --------
    * a: the softmax activations, a float numpy vector of length c
    ---- Hints: --------
    * This problem can be solved using only 2 line(s) of code. More lines are okay.'''
    def forward(self, x):
        ##############################
        ## INSERT YOUR CODE HERE (2.0 points)
        z = self.compute_z(x)
        a = self.compute_a(z)
        ##############################
        return a
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m Softmax_Regression_forward
        (Mac /Linux): python3 -m pytest -v test_1.py -m Softmax_Regression_forward
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: check_dL_dW  ------
    ''' Goal: Gradient Checking: compute the local gradient of the logit function using gradient check    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: the feature vector of one data sample, a numpy vector of length p
    * y: the label of one data sample, an integer scalar value. The values can be 0,1,2, ..., or (c-1)
    * delta: a small number for gradient check, a float scalar
    ---- Outputs: --------
    * dL_dW: the approximated gradients of the loss function w.r.t. the weight matrix, a numpy float matrix of shape (c by p)
    '''
    def check_dL_dW(self, x, y, delta=1e-7):
        dL_dW = np.zeros((self.c,self.p))
        for i in range(self.c):
            for j in range(self.p):
                d = np.zeros((self.c,self.p))
                d[i,j] = delta
                a2 = self.forward(x)
                L2 = self.compute_L(a2,y)
                self.W += d
                a1 = self.forward(x)
                L1 = self.compute_L(a1,y)
                self.W -= d
                dL_dW[i,j] = (L1 - L2 ) / delta
        return dL_dW
        
    #----------------------------------------------------------
    
    #------------- Method: check_dL_db  ------
    ''' Goal: Compute the gradient of the loss function w.r.t. the bias b using gradient checking.    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: the feature vector of one data sample, a numpy vector of length p
    * y: the label of one data sample, an integer scalar value. The values can be 0,1,2, ..., or (c-1)
    * delta: a small number for gradient check, a float scalar
    ---- Outputs: --------
    * dL_db: the approximated gradients of the loss function w.r.t. the biases, a float vector of length c.
    '''
    def check_dL_db(self, x, y, delta=1e-7):
        dL_db =np.zeros(self.c)
        for i in range(self.c):
            d = np.zeros(self.c)
            d[i] = delta
            a2 = self.forward(x) 
            L2 = self.compute_L(a2,y)
            self.b+=d
            a1 = self.forward(x)
            L1 = self.compute_L(a1,y)
            self.b-=d
            dL_db[i] = ( L1 - L2) / delta 
        return dL_db
        
    #----------------------------------------------------------
    
    #------------- Method: backward  ------
    ''' Goal: (Back Propagation) Suppose we are given a softmax regression model with parameters (W and b) and we have a training data sample (x) with label (y).  Suppose we have already computed the activations a(x) on the data sample in the forward-pass. Please compute the global gradients of the loss w.r.t. the parameters W and b on the data sample using back propagation    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: the feature vector of one data sample, a numpy vector of length p
    * y: the label of one data sample, an integer scalar value. The values can be 0,1,2, ..., or (c-1)
    * a: the softmax activations, a float numpy vector of length c
    ---- Outputs: --------
    * dL_dW: the partial gradients of the loss function L w.r.t. the weight matrix W, a numpy float matrix of shape (c by p).  The i,j-th element dL_dW[i,j] represents the partial gradient of the loss function L w.r.t. the i,j-th weight W[i,j],  d_L / d_W[i,j]
    * dL_db: the partial gradient of the loss function L w.r.t. the biases b, a float numpy vector of length c.  The i-th element dL_db[i] represents the partial gradient of the loss function w.r.t. the i-th bias,  d_L / d_b[i]
    ---- Hints: --------
    * Step 1: compute all the local gradients by re-using the above functions. 
    * Step 2: use the local gradients to build global gradients for the parameters W and b. 
    * This problem can be solved using only 7 line(s) of code. More lines are okay.'''
    def backward(self, x, y, a):
        ##############################
        ## INSERT YOUR CODE HERE (2.0 points)
        dL_da = self.compute_dL_da(a, y)
        da_dz = self.compute_da_dz(a)
        dL_dz = self.compute_dL_dz(dL_da, da_dz)
        dz_dW = self.compute_dz_dW(x)
        dz_db = self.compute_dz_db()
        dL_dW = self.compute_dL_dW(dL_dz, dz_dW)
        dL_db = self.compute_dL_db(dL_dz, dz_db)
        ##############################
        return dL_dW, dL_db
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m Softmax_Regression_backward
        (Mac /Linux): python3 -m pytest -v test_1.py -m Softmax_Regression_backward
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: update_b  ------
    ''' Goal: (Gradient Descent 1) Suppose we are given a softmax regression model with parameters (W and b) and we have a training data sample (x,y).  Suppose we have already computed the partial gradients of the loss w.r.t. the biases b on the data sample. Please update the biases b using gradient descent    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * dL_db: the partial gradient of the loss function L w.r.t. the biases b, a float numpy vector of length c.  The i-th element dL_db[i] represents the partial gradient of the loss function w.r.t. the i-th bias,  d_L / d_b[i] 
    ---- Hints: --------
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def update_b(self, dL_db):
        ##############################
        ## INSERT YOUR CODE HERE (2.0 points)
        self.b -= self.lr * dL_db
        ##############################
        
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m Softmax_Regression_update_b
        (Mac /Linux): python3 -m pytest -v test_1.py -m Softmax_Regression_update_b
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: update_W  ------
    ''' Goal: (Gradient Descent 2) Suppose we are given a softmax regression model with parameters (W and b) and we have a training data sample (x,y).  Suppose we have already computed the partial gradients of the loss w.r.t. the weights W on the data sample. Please update the weights W using gradient descent    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * dL_dW: the partial gradients of the loss function L w.r.t. the weight matrix W, a numpy float matrix of shape (c by p).  The i,j-th element dL_dW[i,j] represents the partial gradient of the loss function L w.r.t. the i,j-th weight W[i,j],  d_L / d_W[i,j] 
    ---- Hints: --------
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def update_W(self, dL_dW):
        ##############################
        ## INSERT YOUR CODE HERE (2.0 points)
        self.W -= self.lr * dL_dW
        ##############################
        
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m Softmax_Regression_update_W
        (Mac /Linux): python3 -m pytest -v test_1.py -m Softmax_Regression_update_W
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: train  ------
    ''' Goal: (Training Softmax Regression) Given a training dataset X (features), Y (labels), train the softmax regression model using stochastic gradient descent: iteratively update the weights W and biases b using the gradients on each random data sample.  We repeat n_epoch passes over all the training instances    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * X: the feature matrix of training instances, a float numpy matrix of shape (n by p)
    * Y: the labels of training instance, a numpy integer numpy array of length n. The values can be 0, 1, ..., or c-1
    ---- Hints: --------
    * Step 1 Forward pass: compute the linear logits, activations and loss. 
    * Step 2 Back propagation: compute the gradients of W and b. 
    * Step 3 Gradient descent: update the parameters W and b using gradient descent. 
    * This problem can be solved using only 4 line(s) of code. More lines are okay.'''
    def train(self, X, Y):
        n = X.shape[0] # n: the number of training samples
        for _ in range(self.n_epoch): # iterate through the dataset n_epoch times
            indices = np.random.permutation(n) # shuffle the indices of all samples
            for i in indices: # iterate through each random training sample (x,y)
                x=X[i] # the feature vector of the i-th random sample
                y=Y[i] # the label of the i-th random sample
                ##############################
                ## INSERT YOUR CODE HERE (4.0 points)
                a = self.forward(x)
                dL_dW, dL_db = self.backward(x, y, a)
                self.update_W(dL_dW)
                self.update_b(dL_db)
                ##############################
        
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m Softmax_Regression_train
        (Mac /Linux): python3 -m pytest -v test_1.py -m Softmax_Regression_train
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: predict  ------
    ''' Goal: (Using Softmax Regression) Given a trained softmax regression model with parameters W and b. Suppose we have a test dataset Xtest (features). For each data sample x in Xtest, use the softmax regression model to predict the label of x and the probabilities of the label being in each of the classes, i.e. the activation a(x)    '''
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
            ## INSERT YOUR CODE HERE (4.0 points)
            a = self.forward(x)
            P[i] = a
            yt[i] = np.argmax(a)
            ##############################
        return yt, P
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m Softmax_Regression_predict
        (Mac /Linux): python3 -m pytest -v test_1.py -m Softmax_Regression_predict
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    

'''-------- TEST problem1.py file: (40 points) ----------
Please type the following command in your terminal to test the correctness of all the above functions in this file:
        (Windows OS): python -m pytest -v test_1.py
        (Mac /Linux): python3 -m pytest -v test_1.py
------------------------------------------------------'''






