




#------------ No New Package --------------
# NOTE: Please don't import any new package. You should be able to solve the problems using only the package(s) imported here.
import numpy as np
from problem1 import Scalar
from problem2 import LinearModel as LinearModel1D
#---------------------------------------------------------


# ---------------------------------------------------------
'''
    Goal of Problem 3: Automatic Gradient Computation System for Vectors (30 points)
    We have implemented an auto-grad system for scalar variables in the previous questions. Now we want to extend this idea to support vector variables..
    
'''
# ---------------------------------------------------------

'''------------- Class: Vector (24.0 points) -------
    In this class, the goal is to build an automatic gradient computing system to support vector variables. We will implement the basic opporations for vector variables, such as dot product.  For each vector object, we have class properties to store the value (data) and gradient (grad). We also create a few operators (methods) to build a computational graph on the variables 
'''
''' ---- Class Properties ----
    * data: a numpy array to store the values of this vector.
    * grad: a numpy array to store the gradient of the vector, i.e., the partial derivative of the loss function on this vector x (dL_dx).
    * grad_fn: a reference to the gradient function that needs to be called in backward process.
    * grad_fn_params: a list of parameters to pass to the gradient function.
    '''
class Vector(Scalar):
    #------------- Method: __init__  ------
    ''' Goal: create and initialize an scalar object with given data    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * data: the inital values of the vector , a python list or a numpy array
    '''
    def __init__(self, data):
        assert isinstance(data,list) or isinstance(data, np.ndarray) # test if the input data is a python list
        self.data = np.array(data) # store the value in the class property
        self.grad = np.zeros_like(self.data) # initialize the gradient as an all-zero vector
        self.grad_fn = None # initialize the gradient function as unknown
        self.grad_fn_params = [] # initialize the list of parameters as an empty list 
        
        
    #----------------------------------------------------------
    
    #------------- Method: __matmul__  ------
    ''' Goal: Let's implement the dot product of two vectors (a special case of matrix multiplication). The goal is to support the computation of 'z=x@y' in the Vector class. Build the @ operator on the current variable (self or x) by overloading the '__matmul__' operator in Python. Create a new output variable (z), which is a Scalar and build up the computational graph by adding the add operator and the output variable (y). Make sure to connect the variables (x, y and z) for the forward pass (z.data) and backward pass (z.grad_fn)    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * y: a Vector instance, the other input in the '@' operator, so that we want to compute z=x@y 
    ---- Outputs: --------
    * z: the output Scalar instance, the value of z is the dot product of the current Vector (self or x) and the input Vairable (y)
    ---- Hints: --------
    * When connecting the gradient function from x to z, note that the varialbe y is also needed in the gradient function, because both x (self) and y will be needed in the backward pass. So we need put y into the parameter list of the gradient function in z (i.e., z.grad_fn_params). 
    * This problem can be solved using only 3 line(s) of code. More lines are okay.'''
    def __matmul__(self, y):
        ##############################
        ## INSERT YOUR CODE HERE (12.0 points)
        z = Scalar(np.dot(self.data, y.data).item())
        z.grad_fn = self.matmul_grad_fn
        z.grad_fn_params = [y]

        # 6 passed, 1 failed
        ##############################
        return z
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_3.py -m Vector___matmul__
        (Mac /Linux): python3 -m pytest -v test_3.py -m Vector___matmul__
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: matmul_grad_fn  ------
    ''' Goal: This is the gradient function of the @ operator. This function works together with the above '__matmul__' function. This function is used during the backward pass, when the output variable z need to call back the gradient function of the input variable x. Suppose the current variable (self) is x and the output variable is z (z = x@y). This gradient function is used in the backward pass to back propagate the gradient from the output variable (z) to the input variable (x). In this function, you need to do the following: (1) compute the local gradients dz_dx and dz_dy correctly; (2) compute the global gradients dL_dx and dL_dy using chain rule; (3) the inputs (x and y) may be computed by another operator (for example, x=a.square() and y = x@b), we need to call the backward functions in both x (self) and y variable so that the gradient can backpropagate. Note in this case, the object (a) is not given directly, but a's gradient function should have been already stored in x.grad_fn.    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * dL_dz: the gradient of the loss (L) on the output variable (z), assuming z = x@y
    * y: a Vector instance, the other input in the '@' operator, so that we want to compute x@y 
    ---- Outputs: --------
    * dz_dx: the local gradient of the output (z) on the input (x or self), a numpy vector
    * dz_dy: the local gradient of the output (z) on the input (y), a numpy vector
    * dL_dx: the global gradient of the loss (L) on the input (x or self), a numpy vector
    * dL_dy: the global gradient of the loss (L) on the input (y), a numpy vector
    ---- Hints: --------
    * Make sure to backpropagate the gradient in both x (self) and y variable, by calling their backward() function. When calling their function, make sure to pass the global gradients dL_dx and dL_dy to them separately as a parameter. 
    * This problem can be solved using only 6 line(s) of code. More lines are okay.'''
    def matmul_grad_fn(self, dL_dz, y):
        ##############################
        ## INSERT YOUR CODE HERE (12.0 points)
        dz_dx = y.data
        dz_dy = self.data
        dL_dx = dL_dz * dz_dx
        dL_dy = dL_dz * dz_dy
        self.grad += dL_dx
        y.grad += dL_dy
        if self.grad_fn:
            self.grad_fn(self.grad, *self.grad_fn_params)
        if y.grad_fn:
            y.grad_fn(y.grad, *y.grad_fn_params)

        # 5 passed, 2 failed
        ##############################
        return dz_dx, dz_dy, dL_dx, dL_dy
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_3.py -m Vector_matmul_grad_fn
        (Mac /Linux): python3 -m pytest -v test_3.py -m Vector_matmul_grad_fn
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
'''------------- Class: LinearModel (6.0 points) -------
    In this class, the goal is to build a linear regression model (least square regression) with p-dimensional input features. In this model, we have 1 dimensional linear model z = w@x + b, here w and b are the parameters of the linear model. We will use the automatic gradient system in the previous problem to optimize the parameters of this model 
'''
''' ---- Class Properties ----
    * w: the weight paramester of the linear model, a vector.
    * b: the bias paramester of the linear model, a Scalar.
    '''
class LinearModel(LinearModel1D):
    #------------- Method: __init__  ------
    ''' Goal: create and initialize the paramesters of the 1d linear model    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * p: the number of dimensions in the feature vector, an integer scalar
    '''
    def __init__(self, p):
        self.w = Vector(np.random.rand(p)) # a Vector variable with random value
        self.b = Scalar(np.random.rand()) # a Scalar variable with random value
        
        
    #----------------------------------------------------------
    
    #------------- Method: compute_z  ------
    ''' Goal: Please compute the output of the linear model z =  w@x + b    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: the feature vector of one data sample, a p-dimensional Vector
    ---- Outputs: --------
    * z: the output of the linear model, a Scalar
    ---- Hints: --------
    * You can access to the parameters (w and b) through 'self', such as 'self.w'. 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_z(self, x):
        ##############################
        ## INSERT YOUR CODE HERE (6.0 points)
        
        ##############################
        return z
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_3.py -m LinearModel_compute_z
        (Mac /Linux): python3 -m pytest -v test_3.py -m LinearModel_compute_z
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




