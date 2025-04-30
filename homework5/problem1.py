
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
import torch as th
import torch.nn as nn
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
    Goal of Problem 1: Using PyTorch Auto-gradient System (Tensor class and SGD optimizer) (30 points)
    In this problem, we will use PyTorch auto-gradient system to build the softmax regression model again. We will use the automatic gradient system in PyTorch (Tensor and SGD optimizer) to perform gradient descent algorithm on our model.
    
'''
# ---------------------------------------------------------

'''------------- Class: SoftmaxRegression (30.0 points) -------
    In this class, the goal is to build a softmax regression model with PyTorch tensor and SGD optimizer. We will use the automatic gradient system in PyTorch to optimize the parameters of this model 
'''
''' ---- Class Properties ----
    * p: the number of input features, an integer scalar.
    * c: the number of classes in the classification task, an integer scalar.
    * W: the weights of the linear model, a PyTorch float vector (requires gradients) of length p. Here w[i] is the weight of the model on the i-th feature.
    * b: the bias of the linear model, a PyTorch float scalar (requires gradients).
    '''
class SoftmaxRegression(nn.Module):
    #------------- Method: __init__  ------
    ''' Goal: create and initialize the module    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * p: the number of input features, an integer scalar
    * c: the number of classes in the classification task, an integer scalar
    * lr: also called alpha, the learning rate of the stochastic gradient descent algorithm, a float scalar, controlling the speed of gradient descent. Note in the slides we use the notation 'alpha' to refer to this value
    * n_epoch: an integer scalar, the number of passes to iterate through all training examples during stochastic gradient descent.
    '''
    def __init__(self, p, c, lr=0.001, n_epoch=100):
        super(SoftmaxRegression, self).__init__()
        self.W = nn.Parameter(th.randn(p,c)/p) # initialize the parameter Weights W randomly
        self.b = nn.Parameter(th.zeros(c)) # initialize the parameter bias b as zeros
        self.optimizer = th.optim.SGD([self.W, self.b],lr = lr) # initialize SGD optimizer to handle the gradient descent of the parameters
        self.n_epoch = n_epoch
        self.loss_fn = nn.CrossEntropyLoss() # the loss function for multi-class classification
        
        
    #----------------------------------------------------------
    
    #------------- Method: forward  ------
    ''' Goal: Given a softmax regression model with parameters W and b, please compute the linear logits z on a mini-batch of data samples x1, x2, ... x_batch_size. In the mean time, please also connect the global gradients of the linear logits z (dL_dz) with the global gradients of the weights dL_dW and the biases dL_db in the PyTorch tensors    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: the feature vectors of a mini-batch of data samples, a float torch tensor of shape (batch_size, p)
    ---- Outputs: --------
    * z: the linear logits on a mini-batch of data samples, a float torch tensor of shape (batch_size, c)
    ---- Hints: --------
    * When computing z values, in order to connect the global gradients dL_dz with dL_dW and dL_db, you may want to use the operators in PyTorch, instead of in NumPy or Python. For example, np.dot() is the numpy product of two numpy arrays, which will only compute the values z correctly, but cannot connect the global gradients of the torch tensors W and b. Instead, you may want to find the PyTorch version of dot product for two torch tensors. 
    * For PyTorch tensors, A@B represents the matrix multiplication between two torch matrices A and B. 
    * The parameters W and b are accessable through self.W and self.b. 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def forward(self, x):
        ##############################
        ## INSERT YOUR CODE HERE (6.0 points)
        z = x @ self.W + self.b
        ##############################
        return z
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m SoftmaxRegression_forward
        (Mac /Linux): python3 -m pytest -v test_1.py -m SoftmaxRegression_forward
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_L  ------
    ''' Goal: Suppose we are given a softmax regression model and we have already computed the linear logits z on a mini-batch of training samples. Suppose the labels of the training samples are in y. Please compute the average loss of the softmax regression model on the mini-batch of training samples. In the mean time, please also connect the global gradients of the linear logits z (dL_dz) with the loss L correctly    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * z: the linear logits on a mini-batch of data samples, a float torch tensor of shape (batch_size, c)
    * y: the labels of a mini-batch of data samples, a torch integer vector of length batch_size. The value of each element can be 0,1,2, ..., or (c-1)
    ---- Outputs: --------
    * L: the average multi-class cross entropy loss on a mini-batch of training samples, a torch float scalar
    ---- Hints: --------
    * The loss L is a scalar, computed from the average of the cross entropy loss on all samples in the mini-batch. For example, if the loss on the four training samples are 0.1, 0.2, 0.3, 0.4, then the final loss L is the average of these numbers as (0.1+0.2+0.3+0.4)/4 = 0.25. 
    * You could use the loss function in self.loss_fn to compute the loss. 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_L(self, z, y):
        ##############################
        ## INSERT YOUR CODE HERE (6.0 points)
        L = self.loss_fn(z, y)
        ##############################
        return L
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m SoftmaxRegression_compute_L
        (Mac /Linux): python3 -m pytest -v test_1.py -m SoftmaxRegression_compute_L
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: update_parameters  ------
    ''' Goal: (Gradient Descent) Suppose we are given a softmax regression model with parameters (W and b) and we have a mini-batch of training data samples (x,y).  Suppose we have already computed the global gradients of the average loss L w.r.t. the weights W on the mini-batch of data samples. Assume that we have already created an optimizer for the parameter W and b in self.optimizer. Please update the weights W and b using gradient descent. After the update, the global gradients of W and b should be set to all zeros    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    ---- Hints: --------
    * You could use self.optimizer to update the parameters and set their gradients to zero. 
    * Why there is no input for this function? Maybe you don't need any input. In the __init__() function, we have already properly set up the optimizer for the parameters W and b, so you could just use the optimizer to perform gradient descent (updating the parameters). 
    * Although the parameters W and b are NOT given explicitly in the input of this function, but we can assume the W and b are already properly configured in the optimizer. So the optimizer is configured to handle the parameters W and b. 
    * Although the gradients of the parameters dL_dW and dL_db are NOT given explicitly in the input of this function, but we can assume that in the PyTorch tensors W and b, the gradients are already properly computed and are stored in W.grad (for dL_dW) and b.grad (for dL_db). 
    * Although the learning rate is NOT given explicitly in the input of this function, but we can assume that the optimizer was already configured with the learning rate parameter. 
    * This problem can be solved using only 2 line(s) of code. More lines are okay.'''
    def update_parameters(self):
        ##############################
        ## INSERT YOUR CODE HERE (6.0 points)
        self.optimizer.step()
        self.optimizer.zero_grad()
        ##############################
        
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m SoftmaxRegression_update_parameters
        (Mac /Linux): python3 -m pytest -v test_1.py -m SoftmaxRegression_update_parameters
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: train  ------
    ''' Goal: (Training Softmax Regression) Given a training dataset X (features), Y (labels) in a data loader, train the softmax regression model using mini-batch stochastic gradient descent: iteratively update the weights W and biases b using the gradients on each mini-batch of random data samples.  We repeat n_epoch passes over all the training samples    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * data_loader: a PyTorch loader of a dataset
    ---- Hints: --------
    * Step 1 Forward pass: compute the linear logits and loss. 
    * Step 2 Back propagation: compute the gradients of W and b. 
    * Step 3 Gradient descent: update the parameters W and b using gradient descent. 
    * This problem can be solved using only 4 line(s) of code. More lines are okay.'''
    def train(self, data_loader):
        for _ in range(self.n_epoch): # iterate through the dataset n_epoch times
            for mini_batch in data_loader: # iterate through the dataset, with one mini-batch of random training samples (x,y) at a time
                x=mini_batch[0] # the feature vectors of the data samples in a mini-batch
                y=mini_batch[1] # the labels of the samples in a mini-batch
                ##############################
                ## INSERT YOUR CODE HERE (6.0 points)
                z = self.forward(x)
                loss = self.compute_L(z, y)
                loss.backward()
                self.update_parameters()
                ##############################
        
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m SoftmaxRegression_train
        (Mac /Linux): python3 -m pytest -v test_1.py -m SoftmaxRegression_train
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: predict  ------
    ''' Goal: (Using Softmax Regression)  Given a trained softmax regression model with parameters W and b. Suppose we have a mini-batch of test data samples. Please use the softmax regression model to predict the labels    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: the feature vectors of a mini-batch of test data samples, a float torch tensor of shape (batch_size, p)
    ---- Outputs: --------
    * y_predict: the predicted labels of a mini-batch of test data samples, a torch integer vector of length batch_size. y_predict[i] represents the predicted label on the i-th test sample in the mini-batch
    ---- Hints: --------
    * This is a multi-class classification task, for each sample, the label should be predicted as the index of the largest value of each row of the linear logit z. 
    * You could use the argmax() function in PyTorch to return the indices of the largest values in a tensor. 
    * This problem can be solved using only 2 line(s) of code. More lines are okay.'''
    def predict(self, x):
        ##############################
        ## INSERT YOUR CODE HERE (6.0 points)
        z = self.forward(x)
        y_predict = th.argmax(z, dim=1)
        ##############################
        return y_predict
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m SoftmaxRegression_predict
        (Mac /Linux): python3 -m pytest -v test_1.py -m SoftmaxRegression_predict
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    

'''-------- TEST problem1.py file: (30 points) ----------
Please type the following command in your terminal to test the correctness of all the above functions in this file:
        (Windows OS): python -m pytest -v test_1.py
        (Mac /Linux): python3 -m pytest -v test_1.py
------------------------------------------------------'''






