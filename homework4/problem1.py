
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
    Goal of Problem 1: A Simple Example of Automatic Gradient Computation System (Scalar) (50 points)
    When building deep neural networks, we need to use an automatic gradient computation system, such as PyTorch and Tensorflow, to help handling the different steps of gradient descent algorithm: (1) build computational graphs, (2) compute values in the forward pass, (3) compute gradients of all parameters in the backward pass, (4) update the parameters with one step of gradient descent.  In this problem, we will build a simple example of the automatic gradient computation system. Pytorch and Tensorflow systems starts with Tensors and operators on Tensors, which can be hard to understand. Here we start with a simple case, scalars, where the nodes of the compuational graphs are not tensors, but scalars; connections in the graphs are operators designed for scalars instead of tensors. So this is the easiest, most intuitive case of the pytorch or tensorflow systems. How can we build a system to support automatic gradient computation in this graphs of scalars and scalar operators. We need to (1) create a class 'Scalar' to support scalar variables (value and gradient) and compute gradients on these scalar variables recursively; (2) build an optimizer class to support the stochastic gradient descent algorithm on scalar variables. For simplicity, we assume all scalar varialbes require gradient computation. In Pytorch and Tensorflow, it is possible to set 'requires_grad=False' to indicate that this variable doesn't require gradient computation.
    
'''
# ---------------------------------------------------------

'''------------- Class: Scalar (45.0 points) -------
    To understand how an automatic gradient system works, we can start with a scalar variable and build an automatic gradient computation system for scalar objects. We will need scalar classes to store data and gradients, operators to build computational graphs. In this class, the goal is to build an automatic gradient computing system to support scalar variables. We will implement the basic opporations for scalar variables, such as add, exp and log.  For each scalar object, we have class properties to store the value (data) and gradient (grad). We also create a few operators (methods) to build a computational graph on the variables 
'''
''' ---- Class Properties ----
    * data: a float scalar to store the value of this variable instance.
    * grad: a float scalar, the gradient of the loss function on this variable (dL_dx).
    * grad_fn: a reference to the gradient function that needs to be called in backward process.
    * grad_fn_params: a list of parameters to pass to the gradient function.
    '''
class Scalar:
    #------------- Method: __init__  ------
    ''' Goal: create and initialize an scalar object with given data    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * data: the inital value of the scalar variable, a float scalar
    '''
    def __init__(self, data):
        assert float(data) == data # test if the input data is a number
        self.data = data # store the value in the class property
        self.grad = 0. # initialize the gradient as 0
        self.grad_fn = None # initialize the gradient function as unknown
        self.grad_fn_params = [] # initialize the list of parameters as an empty list 
        
        
    #----------------------------------------------------------
    
    #------------- Method: backward  ------
    ''' Goal: Given a computational graph, we only want to call the backward function in the final loss varialbe. Then the system will automatically find all the variables in the graph to recursively call their backward functions. Here in this function, assuming the current variable is the final loss of the model, call this backward function to run the back propagation starting from the current variable to compute the gradients of all the variables in the computational graph    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * grad: the gradient of the final loss on the current variable, grad = dL_dx (if x is the current variable, L is the loss variable)
    '''
    def backward(self, grad=1.0):
        self.grad+=grad # accumulate the gradient in the current variable
        if self.grad_fn: # if the current variable is computed from another operator, the grad_fn will point to the upstream variable's gradient function in the computational graph
            self.grad_fn(grad,*self.grad_fn_params) # call the gradient function of the upstream variable in the computational graph and pass the list of parameters
        
        
    #----------------------------------------------------------
    
    #------------- Method: square  ------
    ''' Goal: Let's build our first operator, square opperator on the current variable (self). For each operator (such as square, add, exp), we need to have a pair of functions: (1) one function for the forward pass to build the computational graph; (2) another function for the backward pass for computing the gradients. In this example, for the square operator, we will need to build two functions/methods, the 'square()' method is for the forward pass, the 'square_grad_fn()' method is for the backward pass. So let's start! Build the square operator on the current variable (self or x). Compute the square of the current variable (self or x), create a new output variable (y = x^2) and build up the computational graph by adding the square operator and the output variable (y). Make sure to connect the variables (x and y) for the forward pass (y.data) and backward pass (y.grad_fn)    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    ---- Outputs: --------
    * y: the output Scalar instance, the value of y is the square of the current variable (self or x)
    ---- Example: --------
    Suppose the current variable (self or x) is a constant scalar variable (x=2), and we want to compute the square value of x through code 'y = x.square()'. 
    In the 'square' method of variable x, we should create a new Scalar instance (y) and set the value of y to be 4. 
    Because y is computed by the square operator on x, we need to build a connection in the computational graph, indicating that y is created by a function 'square operator', so that during the backward pass (back-propagation), when computing the gradients of all variables (y and x), we just need to start with the last variable y and call the gradient call-back function on y using code 'y.grad_fn(...)' (see the 'backward()' method for more details). 
    But in order to make it work, we need set the y.grad_fn property with a gradient function for which the variable y was created. (for example A()) , by 'y.grad_fn = A'. 
    So that when we need to call the function A, we can simply call y.grad_fn() instead. This is the idea of call-back function. 
    ---- Hints: --------
    * Step 1: create a new scalar object and assign the squared value of the current object (self.data). 
    * Step 2: connect the compuational graph by assigning the grad_fn attribute of the output variable y pointing toward the square_grad_fn function of the current variable self. Because y is created by squared operator on current variable self. 
    * The output (y) should be a new Scalar object that is different from the current object (self or x). 
    * You could create a new ojbect in class A using "y = A()"; if the __init__(i) funciton in class A requires a parameter, you could create a new object using "y = A(5)" . 
    * You may want to read the 'backward()' method and see how the gradient functions will be used during the backward pass. 
    * The function 'self.square_grad_fn' seems to be a good call-back function for the output variable y. 
    * There are multiple test cases on this function. If a test case failed, you could look into the test file for more explanations. 
    * This problem can be solved using only 2 line(s) of code. More lines are okay.'''
    def square(self):
        ##############################
        ## INSERT YOUR CODE HERE (4.5 points)
        y = Scalar(self.data ** 2)
        y.grad_fn = self.square_grad_fn
        y.grad_fn_params = [self]
    
        ##############################
        return y
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m Scalar_square
        (Mac /Linux): python3 -m pytest -v test_1.py -m Scalar_square
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: square_grad_fn  ------
    ''' Goal: This is the gradient function of the square operator. This function works together with the above 'square' function. This function is used during the backward pass, when the output variable y need to call back the gradient function of the input variable x. Suppose the current variable (self) is x and the output variable is y (y = x.square()). This gradient function is used in the backward pass to back propagate the gradient from the output variable (y) to the input variable (x). In this function, you need to do the following: (1) compute the local gradient dy_dx correctly; (2) compute the global gradient dL_dx using chain rule; (3) if the input (x) is computed by another operator (for example, x=z.square(), then y = x.square()),  we need to call backward function in the current variable (self) to back-propagate the gradient    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * dL_dy: the gradient of the loss (L) on the output variable (y), assuming y = x.square()
    ---- Outputs: --------
    * dy_dx: the local gradient of the output (y) on the input (x or self)
    * dL_dx: the global gradient of the loss (L) on the input (x or self)
    ---- Hints: --------
    * Step 1: compute the local gradient dy_dx. 
    * Step 2: compute the global gradient dL_x using chain rule. 
    * Step 3: the current variable may also be created by another operator upstream in the computational graph, so call the backward function of this variable (self) to continue to backpropagation process. . 
    * In this function, we need to compute the global gradient dL_dx for the current variable x, but instead of overwriting the value of self.grad, we want the gradient to accumulate, so add the dL_dx to the self.grad, so that the value can accumulate. . 
    * You may want to read the 'backward()' method and see if you could use this function in your solution. 
    * When calling the backward function in the current node (x or self), make sure to pass along the dL_dx in the parameter, so that the parent node/variable can use it to build its gradient using chain rule. 
    * This problem can be solved using only 3 line(s) of code. More lines are okay.'''
    def square_grad_fn(self, dL_dy=1.0):
        ##############################
        ## INSERT YOUR CODE HERE (4.5 points)
        
        ##############################
        return dy_dx, dL_dx
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m Scalar_square_grad_fn
        (Mac /Linux): python3 -m pytest -v test_1.py -m Scalar_square_grad_fn
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: exp  ------
    ''' Goal: Great job! You have already built your first operator (square), now let's build our second operator (exp) on scalar objects, so that our system can support two different operators: square and exp. Build the exp operator on the current variable (self or x). Create a new output variable (y) and build up the computational graph by adding the exp operator and the output variable (y). Make sure to connect the variables (x and y) for the forward pass (y.data) and backward pass (y.grad_fn)    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    ---- Outputs: --------
    * y: the output Scalar instance, the value of y is the exp of the current variable (self or x)
    ---- Example: --------
    Suppose the current variable (self or x) is a constant scalar variable (x=2), and we want to compute the exp value of x through code 'y = x.exp()'. 
    In the 'exp' method of variable x, we should create a new Scalar instance (y) and set the value of y to be exp(2). You could use np.exp() function to compute exp. 
    ---- Hints: --------
    * This problem can be solved using only 2 line(s) of code. More lines are okay.'''
    def exp(self):
        ##############################
        ## INSERT YOUR CODE HERE (4.5 points)
        y = Scalar(np.exp(self.data))
        y.grad_fn = self.exp_grad_fn
        y.grad_fn_params = [self]

        ##############################
        return y
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m Scalar_exp
        (Mac /Linux): python3 -m pytest -v test_1.py -m Scalar_exp
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: exp_grad_fn  ------
    ''' Goal: This is the gradient function of the exp operator. This function works together with the above 'exp' function. This function is used during the backward pass, when the output variable y need to call back the gradient function of the input variable x. Suppose the current variable (self) is x and the output variable is y (y = x.exp()). This gradient function is used in the backward pass to back propagate the gradient from the output variable (y) to the input variable (x). In this function, you need to do the following: (1) compute the local gradient dy_dx correctly; (2) compute the global gradient dL_dx using chain rule; (3) if the input (x) is computed by another operator (for example, x=z.square(), then y = x.exp()), in this case, we need to call the gradient function of z so that the gradient of z can also be computed. Note in this case, the object (z) is not given directly, but z's gradient function should have been already stored in x.grad_fn. Otherwise you need to check your code in the 'exp()' method above, and make sure to assign the gradient function to x.grad_fn    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * dL_dy: the gradient of the loss (L) on the output variable (y), assuming y = x.exp()
    ---- Outputs: --------
    * dy_dx: the local gradient of the output (y) on the input (x or self)
    * dL_dx: the global gradient of the loss (L) on the input (x or self)
    ---- Hints: --------
    * This problem can be solved using only 3 line(s) of code. More lines are okay.'''
    def exp_grad_fn(self, dL_dy):
        ##############################
        ## INSERT YOUR CODE HERE (4.5 points)
       
        ##############################
        return dy_dx, dL_dx
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m Scalar_exp_grad_fn
        (Mac /Linux): python3 -m pytest -v test_1.py -m Scalar_exp_grad_fn
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: log  ------
    ''' Goal: Now let's build our third operator (log), we will need this operator when computing loss functions in logistic regression models. Build the log operator on the current variable (self or x). Create a new output variable (y) and build up the computational graph by adding the log operator and the output variable (y). Make sure to connect the variables (x and y) for the forward pass (y.data) and backward pass (y.grad_fn)    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    ---- Outputs: --------
    * y: the output Scalar instance, the value of y is the log of the current variable (self or x)
    ---- Example: --------
    Suppose the current variable (self or x) is a constant scalar variable (x=2), and we want to compute the log value of x through code 'y = x.log()'. 
    In the 'log' method of variable x, we should create a new Scalar instance (y) and set the value of y to be log(2). You could use np.log() function to compute log. 
    ---- Hints: --------
    * This problem can be solved using only 2 line(s) of code. More lines are okay.'''
    def log(self):
        assert self.data>0 # the input value to the log functin should be positive
        ##############################
        ## INSERT YOUR CODE HERE (2.25 points)
        y = Scalar(np.log(self.data))
        y.grad_fn = self.log_grad_fn 
        y.grad_fn_params = [self]

        ##############################
        return y
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m Scalar_log
        (Mac /Linux): python3 -m pytest -v test_1.py -m Scalar_log
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: log_grad_fn  ------
    ''' Goal: This is the gradient function of the log operator. This function works together with the above 'log' function. This function is used during the backward pass, when the output variable y need to call back the gradient function of the input variable x. Suppose the current variable (self) is x and the output variable is y (y = x.log()). This gradient function is used in the backward pass to back propagate the gradient from the output variable (y) to the input variable (x). In this function, you need to do the following: (1) compute the local gradient dy_dx correctly; (2) compute the global gradient dL_dx using chain rule; (3) if the input (x) is computed by another operator (for example, x=z.square(), then y = x.log()), in this case, we need to call the gradient function of z so that the gradient of z can also be computed. Note in this case, the object (z) is not given directly, but z's gradient function should have been already stored in x.grad_fn. Otherwise you need to check your code in the 'log()' method above, and make sure to assign the gradient function to x.grad_fn    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * dL_dy: the gradient of the loss (L) on the output variable (y), assuming y = x.log()
    ---- Outputs: --------
    * dy_dx: the local gradient of the output (y) on the input (x or self)
    * dL_dx: the global gradient of the loss (L) on the input (x or self)
    ---- Hints: --------
    * This problem can be solved using only 3 line(s) of code. More lines are okay.'''
    def log_grad_fn(self, dL_dy):
        ##############################
        ## INSERT YOUR CODE HERE (2.25 points)
        
        ##############################
        return dy_dx, dL_dx
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m Scalar_log_grad_fn
        (Mac /Linux): python3 -m pytest -v test_1.py -m Scalar_log_grad_fn
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: __add__  ------
    ''' Goal: Great job! Now let's implement the add operator. The goal is to support the computation of 'z=x+y' in the Scalar class. Build the add operator on the current variable (self or x) by overloading the '__add__' operator in Python. Create a new output variable (z) and build up the computational graph by adding the add operator and the output variable (y). Make sure to connect the variables (x, y and z) for the forward pass (z.data) and backward pass (z.grad_fn)    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * y: a Scalar instance, the other input in the 'add' operator, so that we want to compute x+y 
    ---- Outputs: --------
    * z: the output Scalar instance, the value of z is the sum of the current variable (self or x) and the input vairable (y)
    ---- Hints: --------
    * When connecting the gradient function from x to z, note that the varialbe y is also neede in the gradient function, because both x (self) and y will be needed in the backward pass. So we need put y into the parameter list of the gradient function in z (i.e., z.grad_fn_params). 
    * This problem can be solved using only 3 line(s) of code. More lines are okay.'''
    def __add__(self, y):
        ##############################
        ## INSERT YOUR CODE HERE (4.5 points)
        z = Scalar(self.data + y.data)
        z.grad_fn = self.add_grad_fn
        z.grad_fn_params = [self, y] 
        # passes 5, fails 1
        ##############################
        return z
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m Scalar___add__
        (Mac /Linux): python3 -m pytest -v test_1.py -m Scalar___add__
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: add_grad_fn  ------
    ''' Goal: This is the gradient function of the add operator. This function works together with the above 'add' function. This function is used during the backward pass, when the output variable z need to call back the gradient function of the input variable x. Suppose the current variable (self) is x and the output variable is z (z = x+y). This gradient function is used in the backward pass to back propagate the gradient from the output variable (z) to the input variable (x). In this function, you need to do the following: (1) compute the local gradients dz_dx and dz_dy correctly; (2) compute the global gradients dL_dx and dL_dy using chain rule; (3) the inputs (x and y) may be computed by another operator (for example, x=a.square() and y = x+b), we need to call the backward functions in both x (self) and y variable so that the gradient can backpropagate. Note in this case, the object (a) is not given directly, but a's gradient function should have been already stored in x.grad_fn. Otherwise you need to check your code in the '__add__()' method above, and make sure to assign the gradient function to x.grad_fn    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * dL_dz: the gradient of the loss (L) on the output variable (z), assuming z = x+y
    * y: a Scalar instance, the other input in the 'add' operator, so that we want to compute x+y 
    ---- Outputs: --------
    * dz_dx: the local gradient of the output (z) on the input (x or self)
    * dz_dy: the local gradient of the output (z) on the input (y)
    * dL_dx: the global gradient of the loss (L) on the input (x or self)
    * dL_dy: the global gradient of the loss (L) on the input (y)
    ---- Hints: --------
    * Make sure to backpropagate the gradient in both x (self) and y variable, by calling their backward() function. When calling their function, make sure to pass the dL_dx and dL_dy to them separately as the parameters. 
    * This problem can be solved using only 6 line(s) of code. More lines are okay.'''
    def add_grad_fn(self, dL_dz, y):
        ##############################
        ## INSERT YOUR CODE HERE (4.5 points)
        dz_dx = 1  # Local gradient of z with respect to x (self)
        dz_dy = 1  # Local gradient of z with respect to y
        dL_dx = dL_dz * dz_dx  # Global gradient for x
        dL_dy = dL_dz * dz_dy  # Global gradient for y
        self.backward(dL_dx)   # Backpropagate gradient for x (self)
        y.backward(dL_dy)      # Backpropagate gradient for y
        # passes 4, fails 4
        ##############################
        return dz_dx, dz_dy, dL_dx, dL_dy
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m Scalar_add_grad_fn
        (Mac /Linux): python3 -m pytest -v test_1.py -m Scalar_add_grad_fn
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: __sub__  ------
    ''' Goal: Great job! Now let's implement the subtract operator. The goal is to support the computation of 'z=x-y' in the Scalar class. Build the subtract operator on the current variable (self or x) by overloading the '__sub__' operator in Python. Create a new output variable (z) and build up the computational graph by adding the subtract operator and the output variable (y). Make sure to connect the variables (x, y and z) for the forward pass (z.data) and backward pass (z.grad_fn)    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * y: a Scalar instance, the other input in the 'subtract' operator, so that we want to compute x-y 
    ---- Outputs: --------
    * z: the output Scalar instance, the value of z is the subtraction between the current variable (self or x) and the input vairable (y)
    ---- Hints: --------
    * This problem can be solved using only 3 line(s) of code. More lines are okay.'''
    def __sub__(self, y):
        ##############################
        ## INSERT YOUR CODE HERE (2.25 points)
        z = Scalar(self.data - y.data)
        z.grad_fn = self.sub_grad_fn
        z.grad_fn_params = [self, y]
        # passes 5, fails 1 
        ##############################
        return z
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m Scalar___sub__
        (Mac /Linux): python3 -m pytest -v test_1.py -m Scalar___sub__
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: sub_grad_fn  ------
    ''' Goal: This is the gradient function of the subtract operator. This function works together with the above '__sub__' function. This function is used during the backward pass, when the output variable z need to call back the gradient function of the input variable x. Suppose the current variable (self) is x and the output variable is z (z = x-y). This gradient function is used in the backward pass to back propagate the gradient from the output variable (z) to the input variable (x). In this function, you need to do the following: (1) compute the local gradient dz_dx and dz_dy correctly; (2) compute the global gradient dL_dx and dL_dy using chain rule; (3) if the inputs (x and y) are computed by another operator (for example, x=a.square(), then z = x*y), in this case, we need to call the gradient function of a so that the gradient of a can also be computed. Note in this case, the object (a) is not given directly, but a's gradient function should have been already stored in x.grad_fn. Otherwise you need to check your code in the '__sub__()' method above, and make sure to assign the gradient function to x.grad_fn    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * dL_dz: the gradient of the loss (L) on the output variable (z), assuming z = x-y
    * y: a Scalar instance, the other input in the 'subtract' operator, so that we want to compute x-y 
    ---- Outputs: --------
    * dz_dx: the local gradient of the output (z) on the input (x or self)
    * dz_dy: the local gradient of the output (z) on the input (y)
    * dL_dx: the global gradient of the loss (L) on the input (x or self)
    * dL_dy: the global gradient of the loss (L) on the input (y)
    ---- Hints: --------
    * This problem can be solved using only 6 line(s) of code. More lines are okay.'''
    def sub_grad_fn(self, dL_dz, y):
        ##############################
        ## INSERT YOUR CODE HERE (2.25 points)
        dz_dx = 1
        dz_dy = -1 
        dL_dx = dL_dz * dz_dx
        dL_dy = dL_dz * dz_dy 
        self.backward(dL_dx)  
        y.backward(dL_dy) 
        # passes 4, fails 4
        ##############################
        return dz_dx, dz_dy, dL_dx, dL_dy
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m Scalar_sub_grad_fn
        (Mac /Linux): python3 -m pytest -v test_1.py -m Scalar_sub_grad_fn
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: __mul__  ------
    ''' Goal: Great job! Now let's implement the multiplication operator. The goal is to support the computation of 'z=x*y' in the Scalar class. Build the mul operator on the current variable (self or x). Create a new output variable (z) and build up the computational graph by adding the add operator and the output variable (y). Make sure to connect the variables (x, y and z) for the forward pass (z.data) and backward pass (z.grad_fn)    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * y: a Scalar instance, the other input in the 'multiply' operator, so that we want to compute x*y 
    ---- Outputs: --------
    * z: the output Scalar instance, the value of z is the product of the current variable (self or x) and the input vairable (y)
    ---- Hints: --------
    * This problem can be solved using only 3 line(s) of code. More lines are okay.'''
    def __mul__(self, y):
        ##############################
        ## INSERT YOUR CODE HERE (4.5 points)
        z = Scalar(self.data * y.data)
        z.grad_fn = self.mul_grad_fn 
        z.grad_fn_params = [self, y]
        # 6 passed, 1 failed
        ##############################
        return z
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m Scalar___mul__
        (Mac /Linux): python3 -m pytest -v test_1.py -m Scalar___mul__
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: mul_grad_fn  ------
    ''' Goal: This is the gradient function of the mul operator. This function works together with the above 'mul' function. This function is used during the backward pass, when the output variable z need to call back the gradient function of the input variable x. Suppose the current variable (self) is x and the output variable is z (z = x*y). This gradient function is used in the backward pass to back propagate the gradient from the output variable (z) to the input variable (x). In this function, you need to do the following: (1) compute the local gradient dz_dx and dz_dy correctly; (2) compute the global gradient dL_dx and dL_dy using chain rule; (3) if the inputs (x and y) are computed by another operator (for example, x=a.square(), then z = x*y), in this case, we need to call the gradient function of a so that the gradient of a can also be computed. Note in this case, the object (a) is not given directly, but a's gradient function should have been already stored in x.grad_fn. Otherwise you need to check your code in the '__mul__()' method above, and make sure to assign the gradient function to x.grad_fn    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * dL_dz: the gradient of the loss (L) on the output variable (z), assuming z = x*y
    * y: a Scalar instance, the other input in the 'mul' operator, so that we want to compute x*y 
    ---- Outputs: --------
    * dz_dx: the local gradient of the output (z) on the input (x or self)
    * dz_dy: the local gradient of the output (z) on the input (y)
    * dL_dx: the global gradient of the loss (L) on the input (x or self)
    * dL_dy: the global gradient of the loss (L) on the input (y)
    ---- Hints: --------
    * This problem can be solved using only 6 line(s) of code. More lines are okay.'''
    def mul_grad_fn(self, dL_dz, y):
        ##############################
        ## INSERT YOUR CODE HERE (4.5 points)
        
        ##############################
        return dz_dx, dz_dy, dL_dx, dL_dy
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m Scalar_mul_grad_fn
        (Mac /Linux): python3 -m pytest -v test_1.py -m Scalar_mul_grad_fn
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
'''------------- Class: SGD_optimizer (5.0 points) -------
    In this class, the goal is to build an optimizer for stochastic gradient descent. We can use this optimizer to reset the gradients in the paramester (using zero_grad() ) and perform one step of gradient descent (using step())  
'''
''' ---- Class Properties ----
    * vars: a list of Scalar variables (or model parameters) that need to be optimized by this optimizer.
    * lr: the learning rate of the stochastic gradient descent, a float scalar.
    '''
class SGD_optimizer:
    #------------- Method: __init__  ------
    ''' Goal: Create and initialize an optimizer object with given data    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * vars: a list of Scalar variables (or model parameters) that need to be optimized by this optimizer
    * lr: the learning rate of the stochastic gradient descent, a float scalar
    '''
    def __init__(self, vars, lr=0.01):
        self.vars = vars
        self.lr = lr
        
        
    #----------------------------------------------------------
    
    #------------- Method: zero_grad  ------
    ''' Goal: reset the gradients of all the variables in the self.vars list to zeros    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    ---- Hints: --------
    * This problem can be solved using only 2 line(s) of code. More lines are okay.'''
    def zero_grad(self):
        ##############################
        ## INSERT YOUR CODE HERE (2.5 points)
        pass 
        ##############################
        
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m SGD_optimizer_zero_grad
        (Mac /Linux): python3 -m pytest -v test_1.py -m SGD_optimizer_zero_grad
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: step  ------
    ''' Goal: Assuming that the gradients of all the variables are already computed in the backward pass, now we want to call this method to perform one step of gradient descent on all variables in the self.vars list    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    ---- Hints: --------
    * This problem can be solved using only 2 line(s) of code. More lines are okay.'''
    def step(self):
        ##############################
        ## INSERT YOUR CODE HERE (2.5 points)
        pass 
        ##############################
        
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m SGD_optimizer_step
        (Mac /Linux): python3 -m pytest -v test_1.py -m SGD_optimizer_step
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    

'''-------- TEST problem1.py file: (50 points) ----------
Please type the following command in your terminal to test the correctness of all the above functions in this file:
        (Windows OS): python -m pytest -v test_1.py
        (Mac /Linux): python3 -m pytest -v test_1.py
------------------------------------------------------'''






