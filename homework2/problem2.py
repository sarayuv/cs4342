




#------------ No New Package --------------
# NOTE: Please don't import any new package. You should be able to solve the problems using only the package(s) imported here.
from problem1 import Linear_Classification, SGD
import numpy as np
#---------------------------------------------------------


# ---------------------------------------------------------
'''
    Goal of Problem 2: Linear Models for Binary Classification (Logistic Regression) (30 points)
     In this problem, you will implement the Logistic Regression (LR) method for binary classification.  We will optimize the parameters using Stochastic Gradient Descent (SGD) method. The decision boundary between the two classes is a hyperplane in the input feature space. Stochastic Gradient Descent (SGD) is an optimization algorithm often employed to train LR models. We have used SGD in the previous problem for Linear SVM model. Now let's use SGD algorithm to train another method (Logistic Regression). Comparing with Linear SVM model, the only major difference is that the computation of parameter gradients, because we are using a different loss function (cross entropy loss). Note that even though the name of this method has a word 'Regression', the LR model is actually designed for solving binary classification problems, instead of regression problems. For simplicity, in this model, we don't include any regularization term in the loss function, so that we don't have weight on the regularization term..
    
'''
# ---------------------------------------------------------

'''------------- Class: Logistic_Regression (30.0 points) -------
    Logistic Regression (LR) method for binary classification.  We will optimize the parameters using Stochastic Gradient Descent (SGD) method. The decision boundary between the two classes is a hyperplane in the input feature space. Stochastic Gradient Descent (SGD) is an optimization algorithm often employed.  
'''
''' ---- Class Properties ----
    * w: the weights of the linear model, a numpy float vector of length p. Here w[i] is the weight of the model on the i-th feature.
    * b: the bias of the linear model, a float scalar.
    * n_epoch: an integer scalar, the number of passes to iterate through all training examples during stochastic gradient descent..
    * lr: the learning rate of the stochastic gradient descent algorithm, a float scalar, controlling the speed of gradient descent.
    '''
class Logistic_Regression(Linear_Classification,SGD):
    #------------- Method: __init__  ------
    ''' Goal: Initialize the model object    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * p: the number of features/measurements in each data sample, an integer scaler
    * n_epoch: an integer scalar, the number of passes to iterate through all training examples during stochastic gradient descent.
    * lr: also called alpha, the learning rate of the stochastic gradient descent algorithm, a float scalar, controlling the speed of gradient descent. Note in the slides we use the notation 'alpha' to refer to this value
    '''
    def __init__(self, p, n_epoch=100, lr=0.001):
        Linear_Classification.__init__(self,p=p,neg_code=0.)
        SGD.__init__(self,n_epoch=n_epoch,lr=lr)
        
        
    #----------------------------------------------------------
    
    #------------- Method: compute_z  ------
    ''' Goal: (Value Forward Function 1) Given a logistic regression model with parameters w and b, compute the linear logit value z(x) on a data sample x    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: the feature vector of one data sample, a numpy vector of length p
    ---- Outputs: --------
    * z: the logit value on the data sample x, i.e. z(x), a float scalar
    ---- Hints: --------
    * You could use self.compute_fx() function to compute the score of the linear model on a data sample x. This function was implemented in the parent class (Linear_Classification) . 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_z(self, x):
        ##############################
        ## INSERT YOUR CODE HERE (1.5 points)
        z = self.compute_fx(x)
        ##############################
        return z
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Logistic_Regression_compute_z
        (Mac /Linux): python3 -m pytest -v test_2.py -m Logistic_Regression_compute_z
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_dz_db  ------
    ''' Goal: (Local Gradient 1.1) Suppose we are given a logistic regression model with parameters w and b. Suppose we have already computed the linear logit z(x) on a training sample x. Please compute the partial gradient of the linear logit z(x) with respect to (w.r.t.) the bias b    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    ---- Outputs: --------
    * dz_db: the partial gradient of logit z(x) with respect to (w.r.t.) the bias b, a float scalar. It represents (d_z / d_b)
    ---- Hints: --------
    * Why there is no input variable in this function? Maybe you don't need one, because dz_db is a constant value. You don't need any input to compute this gradient. 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_dz_db(self):
        ##############################
        ## INSERT YOUR CODE HERE (1.5 points)
        dz_db = 1.0
        ##############################
        return dz_db
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Logistic_Regression_compute_dz_db
        (Mac /Linux): python3 -m pytest -v test_2.py -m Logistic_Regression_compute_dz_db
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: check_dz_db  ------
    ''' Goal: Gradient Checking: compute the local gradient of the logit function using gradient check    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: the feature vector of one data sample, a numpy vector of length p
    * delta: a small number for gradient check, a float scalar
    ---- Outputs: --------
    * dz_db: the approximated local gradient of the logits w.r.t. the biases using gradient check
    '''
    def check_dz_db(self, x, delta=1e-7):
        v0 = self.compute_z(x)
        self.b+=delta
        v1 = self.compute_z(x)
        self.b-=delta
        dz_db = (v1-v0) / delta
        return dz_db
        
    #----------------------------------------------------------
    
    #------------- Method: compute_dz_dw  ------
    ''' Goal: (Local Gradient 1.2) Suppose we are given a logistic regression model with parameters w and b. Suppose we have already computed the linear logit z(x) on a training sample x. Please compute the partial gradients of the linear logit z(x)  w.r.t. the weights w.     '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: the feature vector of one training data sample, a numpy vector of length p
    ---- Outputs: --------
    * dz_dw: the partial gradient of logit z(x) with respect to (w.r.t.) the weights w, a float vector of length p. It represents (d_z / d_w)
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
        (Windows OS): python -m pytest -v test_2.py -m Logistic_Regression_compute_dz_dw
        (Mac /Linux): python3 -m pytest -v test_2.py -m Logistic_Regression_compute_dz_dw
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: check_dz_dw  ------
    ''' Goal: Gradient Checking: compute the local gradient of the logit function using gradient check    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: the feature vector of one data sample, a numpy vector of length p
    * delta: a small number for gradient check, a float scalar
    ---- Outputs: --------
    * dz_dw: the approximated local gradient of the logits w.r.t. the weights using gradient check
    '''
    def check_dz_dw(self, x, delta=1e-7):
        dz_dw = np.zeros_like(self.w)
        for i in range(len(self.w)):
            d = np.zeros_like(self.w)
            d[i] = delta
            v0 = self.compute_z(x)
            self.w+=d
            v1 = self.compute_z(x)
            self.w-=d
            dz_dw[i] = (v1-v0) / delta
        return dz_dw
        
    #----------------------------------------------------------
    
    #------------- Method: compute_a  ------
    ''' Goal: (Value Forward Function 2) Suppose we are given a logistic regression model and we have already computed the linear logit z on a data sample x (i.e., z(x)). Please compute the sigmoid activation on the data sample a(x). Note when implementing this function, please make sure the function is numerically stable, for example if z is -1000, computing exp(-z) will result in an overflow error. You need to avoid computing exp(-z), in this case, try to use math to find out the activation value on paper, instead of trying to compute the value using computer. Hint: The activation (a) is practically 0 when z = -1000    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * z: the logit value on the data sample x, i.e. z(x), a float scalar
    ---- Outputs: --------
    * a: the sigmoid activation on the data sample x, i.e. a(x), a float scalar
    ---- Hints: --------
    * You could use np.exp(x) to compute the exponential of x. 
    * When computing exp(-z), you need to be careful about an overflowing case. When the (-z) is a large number (say 1000),  the computer can no longer store the result in a floating-point variable. In this case, we may want to avoid computing exp(-z) and assign the final result of a(x) directly. 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_a(self, z):
        ##############################
        ## INSERT YOUR CODE HERE (3.0 points)
        if z >= 0:
            a = 1 / (1 + np.exp(-z))
        else:
            a = np.exp(z) / (1 + np.exp(z))
        ##############################
        return a
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Logistic_Regression_compute_a
        (Mac /Linux): python3 -m pytest -v test_2.py -m Logistic_Regression_compute_a
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_da_dz  ------
    ''' Goal: (Local Gradient 2) Suppose we are given a logistic regression model and we have already computed the linear logit z(x) and activation a(x) on a training sample x. Please compute the gradient of the activation (a) w.r.t. the linear logit z     '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * a: the sigmoid activation on the data sample x, i.e. a(x), a float scalar
    ---- Outputs: --------
    * da_dz: the partial gradient of activation (a) with respect to (w.r.t.) the linear logit (z), a float scalar. It represents d_a / d_z
    ---- Hints: --------
    * Why the value of the linear logit z is not given? Although you could use z to compute this gradient, it is much easier to compute the gradient using the activation a instead. 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_da_dz(self, a):
        ##############################
        ## INSERT YOUR CODE HERE (1.5 points)
        da_dz = a * (1 - a)
        ##############################
        return da_dz
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Logistic_Regression_compute_da_dz
        (Mac /Linux): python3 -m pytest -v test_2.py -m Logistic_Regression_compute_da_dz
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: check_da_dz  ------
    ''' Goal: Gradient Checking: compute the local gradient of the logit function using gradient check    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * z: the logit scalar
    * delta: a small number for gradient check, a float scalar
    ---- Outputs: --------
    * da_dz: the approximated local gradient using gradient check
    '''
    def check_da_dz(self, z, delta=1e-7):
        v0 = self.compute_a(z)
        v1 = self.compute_a(z+delta)
        da_dz = (v1-v0) / delta
        return da_dz
        
    #----------------------------------------------------------
    
    #------------- Method: compute_L  ------
    ''' Goal: (Value Forward Function 3) Suppose we are given a logistic regression model and we have already computed the linear logit z(x) on a training sample x. Suppose the label of the training sample is y. Please compute the loss function of the logistic regression model on the training sample. Note when implementing this function, please make sure the function is numerically stable, for example if z is -1000, computing exp(-z) will result in an overflow error. You need to avoid computing exp(-z), in this case, try to use math to find out a better way to compute the result on paper.     '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * z: the logit value on the data sample x, i.e. z(x), a float scalar
    * y: the label of a data sample, an integer scalar value. The values can be 0 or 1
    ---- Outputs: --------
    * L: the cross entropy loss on the data sample x, a float scalar
    ---- Hints: --------
    * Why the input of this loss function is not the activation a, instead z and y are given? Although you could also compute the loss from the activation, it will be hard to build a numerically stable implementation (for example when z = -1000). A better implementation of the loss function should start with the linear logit z instead of activation a. So by combining the sigmoid function (involving exp()) and cross entropy functions (involving log()) together, you could get an implementation that is stable numerically for computers to compute. This is based upon the intuition that if the input x is large enough  log(1 + exp(x)) is close to log(exp(x)), because 1 is too small comparing with exp(x). Then the log and exp cancel each other, so we just have x. This is the trick to get a numerical stable implementation. This also means that the previous functions "compute_a()" and "compute_da_dz()" we implemented would not be used in practice. They are just to show how the values are computed in the model.. 
    * You could use np.exp(x) to compute the exponential of x. 
    * You could use np.log(x) to compute the natural log of x. 
    * When computing exp(z), you need to be careful about an overflowing case. When the z is a large number (say 1000),  the computer can no longer store the result in a floating-point number. In this case, we may want to avoid computing exp(z) and assign the final result of L(x) directly. When z is very large (say 1000), 1+exp(z) will be very close to exp(z), so we can simplify the equation "log(1+exp(z))-yz" into   "log (exp(z)) - yz". Here log() and exp() cancel out, so we only have "z-yz" left, which is "z(1-y)". 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_L(self, z, y):
        ##############################
        ## INSERT YOUR CODE HERE (3.0 points)
        if z >= 0:
            L = np.log(1 + np.exp(-z)) + z * (1 - y)
        else:
            L = (-z * y) + np.log(1 + np.exp(z))
        ##############################
        return L
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Logistic_Regression_compute_L
        (Mac /Linux): python3 -m pytest -v test_2.py -m Logistic_Regression_compute_L
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_dL_dz  ------
    ''' Goal: (Local Gradient 3) Suppose we are given a logistic regression model and we have already computed the logits z(x) on a training sample x. Suppose the label of the training sample is y. Please compute the gradient of the loss function (L) w.r.t. the linear logit (z). Similar to the previous function, please make sure this function is numerically stable (for example, when z=-1000, what would happen?)    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * z: the logit value on the data sample x, i.e. z(x), a float scalar
    * y: the label of a data sample, an integer scalar value. The values can be 0 or 1
    ---- Outputs: --------
    * dL_dz: the partial gradient of loss (L) with respect to (w.r.t.) the linear logit (z), a float scalar. It represents d_L / d_z
    ---- Hints: --------
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_dL_dz(self, z, y):
        ##############################
        ## INSERT YOUR CODE HERE (3.0 points)
        if z >= 0:
            a = 1 / (1 + np.exp(-z))
        else:
            a = np.exp(z) / (1 + np.exp(z))
        dL_dz = a - y
        ##############################
        return dL_dz
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Logistic_Regression_compute_dL_dz
        (Mac /Linux): python3 -m pytest -v test_2.py -m Logistic_Regression_compute_dL_dz
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: check_dL_dz  ------
    ''' Goal: Gradient Checking: compute the local gradient using gradient check    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * z: the logit scalar
    * y: the label scalar
    * delta: a small number for gradient check, a float scalar
    ---- Outputs: --------
    * dL_dz: the approximated local gradient using gradient check
    '''
    def check_dL_dz(self, z, y, delta=1e-7):
        v0 = self.compute_L(z,y)
        v1 = self.compute_L(z+delta,y)
        dL_dz = (v1-v0) / delta
        return dL_dz
        
    #----------------------------------------------------------
    
    #------------- Method: compute_dL_db  ------
    ''' Goal: (Global Gradient 1) After computing the local gradients, now we should put them together to compute the final gradients of the parameters using the chain rule. Let's start with the easier parameter b. Suppose we are given a logistic regression model with parameters (w and b) and we have a training data sample (x,y).  Suppose we have already computed the local gradients on the data sample. Please compute the global gradient of the loss L w.r.t. bias parameter b using chain rule    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * dL_dz: the partial gradient of loss (L) with respect to (w.r.t.) the linear logit (z), a float scalar. It represents d_L / d_z
    * dz_db: the partial gradient of logit z(x) with respect to (w.r.t.) the bias b, a float scalar. It represents (d_z / d_b)
    ---- Outputs: --------
    * dL_db: the partial gradient of the loss function L w.r.t. the bias b, a float scalar
    ---- Hints: --------
    * Do you know that dz_db always equals 1. So when using chain rule to compute dL_db, you don't really need to use dz_db. 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_dL_db(self, dL_dz, dz_db=1.0):
        ##############################
        ## INSERT YOUR CODE HERE (1.5 points)
        dL_db = dL_dz * dz_db
        ##############################
        return dL_db
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Logistic_Regression_compute_dL_db
        (Mac /Linux): python3 -m pytest -v test_2.py -m Logistic_Regression_compute_dL_db
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_dL_dw  ------
    ''' Goal: (Global Gradient 2) Suppose we are given a logistic regression model with parameters (w and b) and we have a training data sample (x,y).  Suppose we have already computed the local gradients on the data sample. Please compute the global gradient of the loss function L w.r.t. the weight parameters w using chain rule    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * dL_dz: the partial gradient of loss (L) with respect to (w.r.t.) the linear logit (z), a float scalar. It represents d_L / d_z
    * dz_dw: the partial gradient of logit z(x) with respect to (w.r.t.) the weights w, a numpy vector of length p.
    ---- Outputs: --------
    * dL_dw: the partial gradient of the loss function L w.r.t. the weight vector w, a numpy float vector of length p.  The i-th element represents ( d_L / d_w[i])
    ---- Hints: --------
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_dL_dw(self, dL_dz, dz_dw):
        ##############################
        ## INSERT YOUR CODE HERE (1.5 points)
        dL_dw = dL_dz * dz_dw
        ##############################
        return dL_dw
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Logistic_Regression_compute_dL_dw
        (Mac /Linux): python3 -m pytest -v test_2.py -m Logistic_Regression_compute_dL_dw
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: backward  ------
    ''' Goal: (Back Propagation) Now let's put all the gradient computing functions together. Suppose we are given a logistic regression model with parameters (w and b) and we have a training data sample (x) with label (y).  Suppose we have already computed the activation a(x) on the data sample in the forward-pass. Please compute the global gradients of the loss w.r.t. the parameters w and b on the data sample using back propagation.     '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: the feature vector of one data sample, a numpy vector of length p
    * y: the label of a data sample, an integer scalar value. The values can be 0 or 1
    * z: the logit value on the data sample x, i.e. z(x), a float scalar
    ---- Outputs: --------
    * dL_dw: the partial gradient of the loss function L w.r.t. the weight vector w, a numpy float vector of length p.  The i-th element represents ( d_L / d_w[i])
    * dL_db: the partial gradient of the loss function L w.r.t. the bias b, a float scalar
    ---- Hints: --------
    * Step 1: compute all the local gradients using the functions above. 
    * Step 2: use the local gradients to build global gradients for the parameters w and b. 
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
        (Windows OS): python -m pytest -v test_2.py -m Logistic_Regression_backward
        (Mac /Linux): python3 -m pytest -v test_2.py -m Logistic_Regression_backward
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
    * Step 1 compute the linear logit on the training sample x. 
    * Step 2 Back propagation to compute the gradients of w and b. 
    * Step 3 Gradient descent: update the parameters w and b using gradient descent. 
    * This problem can be solved using only 4 line(s) of code. More lines are okay.'''
    def train(self, X, y):
        n = X.shape[0] # n is the number of samples
        for _ in range(self.n_epoch): # iterate through the dataset n_epoch times
            indices = np.random.permutation(n) # shuffle the indices of all instances in the training set
            for i in indices: # iterate through each random instance (x,y)
                xi = X[i] # the feature vector of the i-th random instance
                yi = y[i] # the label of the i-th random instance
                ##############################
                ## INSERT YOUR CODE HERE (9.0 points)
                z = self.compute_z(xi)
                dL_dw, dL_db = self.backward(xi, yi, z)
                self.w -= self.lr * dL_dw
                self.b -= self.lr * dL_db
                ##############################
        
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Logistic_Regression_train
        (Mac /Linux): python3 -m pytest -v test_2.py -m Logistic_Regression_train
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    

'''-------- TEST problem2.py file: (30 points) ----------
Please type the following command in your terminal to test the correctness of all the above functions in this file:
        (Windows OS): python -m pytest -v test_2.py
        (Mac /Linux): python3 -m pytest -v test_2.py
------------------------------------------------------'''






