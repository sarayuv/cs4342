




#------------ No New Package --------------
# NOTE: Please don't import any new package. You should be able to solve the problems using only the package(s) imported here.
from problem1 import SGD, Linear_Classification
import numpy as np
#---------------------------------------------------------


# ---------------------------------------------------------
'''
    Goal of Problem 3: Mini-batch Stochastic Gradient Descent (Logistic Regression) (30 points)
     In this problem, you will implement the mini-batch version of Stochastic Gradient Descent (SGD) for training logistic regression models.  In mini-batch SGD, the dataset is divided into small batches of samples. The size of these batches is typically between 1 and the total number of samples in the dataset. The gradient is computed for each mini-batch, and model parameters are updated based on the average gradient over the mini-batch.  This is different from the previous SGD algorithm we implemented, where the gradient is computed on each individual sample in the dataset. Therefore, the batch size is 1, and the model parameters are updated after processing each sample. .
    
'''
# ---------------------------------------------------------

'''------------- Class: Mini_Batch_SGD (0.0 points) -------
    This is the parent class for mini-batch Stochastic gradient descent algorithms for optimizing parametric models with two parameters, weights (w) and bias (b). 
'''
''' ---- Class Properties ----
    * n_epoch: an integer scalar, the number of passes to iterate through all training examples during stochastic gradient descent..
    * lr: the learning rate of the stochastic gradient descent algorithm, a float scalar, controlling the speed of gradient descent.
    * batch_size: the number of training samples in a mini-batch, an integer scalar.
    '''
class Mini_Batch_SGD(SGD):
    #------------- Method: __init__  ------
    ''' Goal: Initialize the model object    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * n_epoch: an integer scalar, the number of passes to iterate through all training examples during stochastic gradient descent.
    * lr: the learning rate of the stochastic gradient descent algorithm, a float scalar, controlling the speed of gradient descent
    * batch_size: the number of training samples in a mini-batch, an integer scalar
    '''
    def __init__(self, n_epoch=10, lr=0.01, batch_size=20):
        SGD.__init__(self,n_epoch = n_epoch,lr=lr)
        self.batch_size= batch_size
        
        
    #----------------------------------------------------------
    
    #------------- Method: update_w  ------
    ''' Goal: Given the gradients of the parameter w on a mini-batch of training samples, update the values of w using one step of gradient descent using the average gradient in the mini-batch    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * dL_dw: the gradients of the weights, a numpy float matrix of shape (batch_size, p). 
    '''
    def update_w(self, dL_dw):
        self.w -= self.lr * np.mean(dL_dw,axis=0)
        
        
    #----------------------------------------------------------
    
    #------------- Method: update_b  ------
    ''' Goal: Given the gradients of the parameter b on a mini-batch of training samples, update the value of b using one step of gradient descent    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * dL_db: the gradients of the bias parameter, a numpy float vector of length batch_size. 
    '''
    def update_b(self, dL_db):
        self.b -= self.lr * np.mean(dL_db,axis=0)
        
        
    #----------------------------------------------------------
    
'''------------- Class: Logistic_Regression_Batch (30.0 points) -------
    Logistic Regression (LR) method for binary classification.  We will optimize the parameters using mini-batch Stochastic Gradient Descent (SGD) method.  
'''
''' ---- Class Properties ----
    * w: the weights of the linear model, a numpy float vector of length p. Here w[i] is the weight of the model on the i-th feature.
    * b: the bias of the linear model, a float scalar.
    * n_epoch: an integer scalar, the number of passes to iterate through all training examples during stochastic gradient descent..
    * lr: the learning rate of the stochastic gradient descent algorithm, a float scalar, controlling the speed of gradient descent.
    * batch_size: the number of training samples in a mini-batch, an integer scalar.
    '''
class Logistic_Regression_Batch(Linear_Classification,Mini_Batch_SGD):
    #------------- Method: __init__  ------
    ''' Goal: Initialize the model object    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * p: the number of features/measurements in each data sample, an integer scaler
    * n_epoch: an integer scalar, the number of passes to iterate through all training examples during stochastic gradient descent.
    * lr: also called alpha, the learning rate of the stochastic gradient descent algorithm, a float scalar, controlling the speed of gradient descent. Note in the slides we use the notation 'alpha' to refer to this value
    * batch_size: the number of training samples in a mini-batch, an integer scalar
    '''
    def __init__(self, p, n_epoch=100, lr=0.001, batch_size=20):
        Linear_Classification.__init__(self,p=p,neg_code=0.)
        Mini_Batch_SGD.__init__(self,n_epoch=n_epoch,lr=lr,batch_size=batch_size)
        
        
    #----------------------------------------------------------
    
    #------------- Method: compute_z  ------
    ''' Goal: (Value Forward Function 1) Given a logistic regression model with parameters w and b, compute the linear logit values z(x) on a mini-batch of data samples (x)    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: the feature vectors of one mini-batch of data samples, a numpy matrix of shape (batch_size, p), x[i] is the i-th training sample in the mini-batch
    ---- Outputs: --------
    * z: the logit values on the data samples in x, i.e. z(x), a float vector of length batch_size, z[i] is the logit value on the i-th training sample in the mini-batch
    ---- Hints: --------
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_z(self, x):
        ##############################
        ## INSERT YOUR CODE HERE (1.5 points)
        z = x @ self.w + self.b
        ##############################
        return z
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_3.py -m Logistic_Regression_Batch_compute_z
        (Mac /Linux): python3 -m pytest -v test_3.py -m Logistic_Regression_Batch_compute_z
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_dz_db  ------
    ''' Goal: (Local Gradient 1.1) Suppose we are given a logistic regression model with parameters w and b. Suppose we have already computed the linear logit z(x) on a mini-batch of training samples x. Please compute the partial gradient of the linear logit z(x) with respect to (w.r.t.) the bias b    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    ---- Outputs: --------
    * dz_db: the partial gradients of logits z(x) in a mini-batch of samples with respect to (w.r.t.) the bias parameter b, a float vector of length batch_size. dz_db[i] is the gradient of logit z(x[i]) on the i-th training sample w.r.t. the bias parameter b
    ---- Hints: --------
    * Why there is no input variable in this function? Because you don't need any input to compute these gradients, because dz_db[i] is a constant value, which doesn't require any input to compute. 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_dz_db(self):
        ##############################
        ## INSERT YOUR CODE HERE (1.5 points)
        dz_db = np.ones(self.batch_size)
        ##############################
        return dz_db
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_3.py -m Logistic_Regression_Batch_compute_dz_db
        (Mac /Linux): python3 -m pytest -v test_3.py -m Logistic_Regression_Batch_compute_dz_db
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_dz_dw  ------
    ''' Goal: (Local Gradient 1.2) Suppose we are given a logistic regression model with parameters w and b. Suppose we have already computed the linear logit z(x) on a mini-batch of training samples x. Please compute the partial gradients of the linear logit z(x)  w.r.t. the weights w.     '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: the feature vectors of one batch of data samples, a numpy matrix of shape (batch_size, p), x[i] is the i-th training sample in the mini-batch
    ---- Outputs: --------
    * dz_dw: the partial gradients of logits z(x) on a mini-batch of samples with respect to (w.r.t.) the weights w, a float matrix of shape (batch_size,p). dz_dw[i] represents (d_z / d_w) on the i-th training sample in the mini-batch
    ---- Hints: --------
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_dz_dw(self, x):
        ##############################
        ## INSERT YOUR CODE HERE (1.5 points)
        dz_dw = x
        ##############################
        return dz_dw
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_3.py -m Logistic_Regression_Batch_compute_dz_dw
        (Mac /Linux): python3 -m pytest -v test_3.py -m Logistic_Regression_Batch_compute_dz_dw
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_a  ------
    ''' Goal: (Value Forward Function 2) Suppose we are given a logistic regression model and we have already computed the linear logit z on data samples in x (i.e., z(x)). Please compute the sigmoid activation on the data samples a(x).     '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * z: the logit values on the data samples x, i.e. z(x), a float vector of length batch_size
    ---- Outputs: --------
    * a: the sigmoid activation on the data samples in x, i.e. a(x), a float vector of length batch_size
    ---- Hints: --------
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_a(self, z):
        ##############################
        ## INSERT YOUR CODE HERE (3.0 points)
        a = 1 / (1 + np.exp(-z))
        ##############################
        return a
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_3.py -m Logistic_Regression_Batch_compute_a
        (Mac /Linux): python3 -m pytest -v test_3.py -m Logistic_Regression_Batch_compute_a
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_da_dz  ------
    ''' Goal: (Local Gradient 2) Suppose we are given a logistic regression model and we have already computed the linear logits z(x) and activations a(x) on a mini-batch of training samples x. Please compute the gradients of the activation (a) w.r.t. the linear logit z     '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * a: the sigmoid activation on the data samples x, i.e. a(x), a float vector of length batch_size
    ---- Outputs: --------
    * da_dz: the partial gradient of activation (a) with respect to (w.r.t.) the linear logit (z), a float vector of length batch_size. da_dz[i] represents d_a / d_z on the i-th data sample in the mini-batch
    ---- Hints: --------
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_da_dz(self, a):
        ##############################
        ## INSERT YOUR CODE HERE (1.5 points)
        da_dz = a * (1 - a)
        ##############################
        return da_dz
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_3.py -m Logistic_Regression_Batch_compute_da_dz
        (Mac /Linux): python3 -m pytest -v test_3.py -m Logistic_Regression_Batch_compute_da_dz
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_L  ------
    ''' Goal: (Value Forward Function 3) Suppose we are given a logistic regression model and we have already computed the linear logit z(x) on a mini-batch of training samples x. Suppose the label of the training samples are y. Please compute the loss function of the logistic regression model on the training samples.    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * z: the logit values on the data samples x, i.e. z(x), a float vector of length batch_size
    * y: the labels of the data samples, an integer vector of length batch_size. Each element y[i] can be 0 or 1
    ---- Outputs: --------
    * L: the cross entropy losses on the data samples x, a float vector of length batch_size
    ---- Hints: --------
    * This problem can be solved using only 3 line(s) of code. More lines are okay.'''
    def compute_L(self, z, y):
        ##############################
        ## INSERT YOUR CODE HERE (3.0 points)
        L = np.where(z >= 0, np.log(1 + np.exp(-z)) + z*(1-y), -z*y + np.log(1 + np.exp(z)))
        L = np.where((z > 500) & (y == 0), z, np.where((z < -500) & (y == 1), -z, L))
        ##############################
        return L
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_3.py -m Logistic_Regression_Batch_compute_L
        (Mac /Linux): python3 -m pytest -v test_3.py -m Logistic_Regression_Batch_compute_L
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_dL_dz  ------
    ''' Goal: (Local Gradient 3) Suppose we are given a logistic regression model and we have already computed the logits z(x) on a batch of training samples x. Suppose the labels of the training samples are y. Please compute the gradient of the loss function (L) w.r.t. the linear logits (z).     '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * z: the logit values on the data samples x, i.e. z(x), a float vector of length batch_size
    * y: the labels of data samples, an integer vector of length batch_size. Each element can be 0 or 1
    ---- Outputs: --------
    * dL_dz: the partial gradient of loss (L) with respect to (w.r.t.) the linear logit (z), a float vector of length batch_size. Each element dL_dz[i] represents d_L / d_z on the i-th data sample in the mini-batch
    ---- Hints: --------
    * This problem can be solved using only 3 line(s) of code. More lines are okay.'''
    def compute_dL_dz(self, z, y):
        ##############################
        ## INSERT YOUR CODE HERE (3.0 points)
        a = self.compute_a(z)
        dL_dz = a - y
        ##############################
        return dL_dz
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_3.py -m Logistic_Regression_Batch_compute_dL_dz
        (Mac /Linux): python3 -m pytest -v test_3.py -m Logistic_Regression_Batch_compute_dL_dz
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_dL_db  ------
    ''' Goal: (Global Gradient 1) After computing the local gradients, now we should put them together to compute the final gradients of the parameters using the chain rule.     '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * dL_dz: the partial gradient of loss (L) with respect to (w.r.t.) the linear logit (z), a float vector of length batch_size
    * dz_db: the partial gradient of logit z(x) with respect to (w.r.t.) the bias b, a float vector of length batch_size
    ---- Outputs: --------
    * dL_db: the partial gradient of the loss function L w.r.t. the bias b, a float vector of batch_size
    ---- Hints: --------
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_dL_db(self, dL_dz, dz_db):
        ##############################
        ## INSERT YOUR CODE HERE (1.5 points)
        dL_db = dL_dz * dz_db
        ##############################
        return dL_db
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_3.py -m Logistic_Regression_Batch_compute_dL_db
        (Mac /Linux): python3 -m pytest -v test_3.py -m Logistic_Regression_Batch_compute_dL_db
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_dL_dw  ------
    ''' Goal: (Global Gradient 2) Suppose we are given a logistic regression model with parameters (w and b) and we have a batch of training data samples (x,y).  Suppose we have already computed the local gradients on the data sample. Please compute the global gradient of the loss function L w.r.t. the weight parameters w using chain rule    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * dL_dz: the partial gradient of loss (L) with respect to (w.r.t.) the linear logit (z), a float vector of length batch_size
    * dz_dw: the partial gradient of logit z(x) with respect to (w.r.t.) the weights w, a numpy matrix of shape (batch_size, p). dz_dw[i] represents the gradient on the i-th sample of the mini-batch
    ---- Outputs: --------
    * dL_dw: the partial gradient of the loss function L w.r.t. the weight vector w, a numpy float matrxi of shape (batch_size, p).
    ---- Hints: --------
    * np.newaxis is a special constant in the NumPy library. It is used to increase the dimension of the existing array by one more dimension, when used in array indexing and array manipulation operations. This can be particularly useful when you want to perform operations like matrix multiplication or broadcasting where the dimensions of the arrays need to be compatible. It's worth noting that np.newaxis doesn't actually add new elements to the array; it simply changes the way the array is interpreted in terms of its shape and dimensions.. 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_dL_dw(self, dL_dz, dz_dw):
        ##############################
        ## INSERT YOUR CODE HERE (1.5 points)
        dL_dw = dL_dz[:, np.newaxis] * dz_dw
        ##############################
        return dL_dw
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_3.py -m Logistic_Regression_Batch_compute_dL_dw
        (Mac /Linux): python3 -m pytest -v test_3.py -m Logistic_Regression_Batch_compute_dL_dw
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: backward  ------
    ''' Goal: (Back Propagation) Now let's put all the gradient computing functions together. Suppose we are given a logistic regression model with parameters (w and b) and we have a mini-batch of training data samples (x) with labels (y).  Suppose we have already computed the activations a(x) on the data samples in the forward-pass. Please compute the global gradients of the loss w.r.t. the parameters w and b on the data samples using back propagation.     '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: the feature vectors of data samples, a numpy matrix of shape (batch_size, p)
    * y: the label of data samples, an integer vector of length batch_size. The values can be 0 or 1
    * z: the logit values on the data samples x, i.e. z(x), a float vector of length batch_size
    ---- Outputs: --------
    * dL_dw: the partial gradients of the loss function L w.r.t. the weight vector w, a numpy float matrix of shape (batch_size, p)
    * dL_db: the partial gradients of the loss function L w.r.t. the bias b, a float vector of length batch_size
    ---- Hints: --------
    * This problem can be solved using only 4 line(s) of code. More lines are okay.'''
    def backward(self, x, y, z):
        ##############################
        ## INSERT YOUR CODE HERE (3.0 points)
        dL_dz = self.compute_dL_dz(z, y)
        dz_dw = self.compute_dz_dw(x)
        dz_db = self.compute_dz_db()
        dL_dw = self.compute_dL_dw(dL_dz, dz_dw)
        dL_db = self.compute_dL_db(dL_dz, dz_db)
        ##############################
        return dL_dw, dL_db
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_3.py -m Logistic_Regression_Batch_backward
        (Mac /Linux): python3 -m pytest -v test_3.py -m Logistic_Regression_Batch_backward
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: train  ------
    ''' Goal: Given a set of training samples, find the optimal value for the parameters w and b    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * X: the feature matrix of the training samples, a numpy matrix of shape n by p, here X[i,j] is the value of the j-th feature on the i-th training sample
    * y: the labels of the training samples, a numpy float vector of length n, y[i] represents the label of the i-th sample in the dataset
    ---- Hints: --------
    * This problem can be solved using only 4 line(s) of code. More lines are okay.'''
    def train(self, X, y):
        n = X.shape[0] # n is the number of samples
        for _ in range(self.n_epoch): # iterate through the dataset n_epoch times
            indices = np.random.permutation(n) # shuffle the indices of all instances in the training set
            Xs= X[indices] # shuffled feature vectors
            ys= y[indices] # shuffled labels
            for i in range(0,n,self.batch_size): 
                xi = Xs[i:i+self.batch_size] # the feature vectors of a mini-batch of random samples
                yi = ys[i:i+self.batch_size] # the labels of a mini-batch of random samples
                ##############################
                ## INSERT YOUR CODE HERE (9.0 points)
                z = self.compute_z(xi)
                dL_dw, dL_db = self.backward(xi, yi, z)
                self.update_w(dL_dw)
                self.update_b(dL_db)
                ##############################
        
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_3.py -m Logistic_Regression_Batch_train
        (Mac /Linux): python3 -m pytest -v test_3.py -m Logistic_Regression_Batch_train
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    

'''-------- TEST problem3.py file: (30 points) ----------
Please type the following command in your terminal to test the correctness of all the above functions in this file:
        (Windows OS): python -m pytest -v test_3.py
        (Mac /Linux): python3 -m pytest -v test_3.py
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




