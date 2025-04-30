




#------------ No New Package --------------
# NOTE: Please don't import any new package. You should be able to solve the problems using only the package(s) imported here.
import torch as th
import torch.nn as nn
#---------------------------------------------------------


# ---------------------------------------------------------
'''
    Goal of Problem 3: Recurrent Neural Network for Binary Time Sequence Classification (with PyTorch) (30 points)
    In this problem, you will implement the recurrent neural network for binary sequence classification problems.  Here we assume that each time sequence is assigned with one binary label.  For example, in audio classification, each time sequence is a short clip of audio recording, and the label of the sequence is either 0 (non-wake word) or 1 (wake word).  The goal of this problem is to learn the details of recurrent neural network by building RNN from scratch.  The structure of the RNN includes one recurrent layer repeating itself for l time steps and a fully-connected layer attached to the last time step of the recurrent layer to predict the label of a time sequence.  (Recurrent layer for time step 1)-> (Recurrent layer for time step 2) -> ...(Recurrent layer for time step t) -> (Fully connected layer) -> predicted label.
    
'''
# ---------------------------------------------------------

'''------------- Class: RecurrentLayer (15.0 points) -------
    Recurrent Layer is the core component responsible for processing sequential data (on each time step) by maintaining and updating a hidden state over time. This hidden state captures information about the sequence seen up to the current time step and is updated recursively as new input is processed. 
'''
''' ---- Class Properties ----
    * U: the weights of the recurrent layers on the input features of the time sequence at the current time step, a float torch Tensor of shape (p, h).
    * V: the weights of the recurrent layers on the old memory ht_1 (the hidden states at the previous time step (t-1), a float torch Tensor of shape (h, h). Here V[j,k] is the weight connecting the j-th neuron in ht_1 (input) to the k-th neuron in ht (output).
    * b: the biases of the recurrent layer, a float torch vector of length h. b_h[k] is the bias on the k-th neuron of the hidden states.
    '''
class RecurrentLayer(nn.Module):
    #------------- Method: __init__  ------
    ''' Goal: create and initialize the module    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * p: the number of input features at each time step of a time sequence, an integer scalar
    * h: the number of neurons in the hidden states (or the activations of the recurrent layer), an integer scalar
    '''
    def __init__(self, p, h):
        super(RecurrentLayer, self).__init__()
        self.p = p
        self.h = h
        self.U = nn.Parameter(th.randn(p,h)/h) # initialize the parameter Weights W randomly
        self.V = nn.Parameter(th.randn(h,h)/h) # initialize the parameter Weights W randomly
        self.b = nn.Parameter(th.zeros(h)) # initialize the parameter bias b as zero
        
        
    #----------------------------------------------------------
    
    #------------- Method: compute_zt  ------
    ''' Goal: (Recurrent Layer: Linear Logits) Given a recurrent neural network layer with parameters weights U, V and biases b_h. We have a mini-batch of data samples xt at the t-th time step. Suppose we have already computed the hidden states h_(t-1) at the previous time step (t-1). Please compute the linear logits zt in the recurrent layer at the t-th time step on a mini-batch of data samples    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * xt: a mini-batch of time sequences at the t-th time step, a float torch tensor of shape (n, p). xt[k] represents the k-th time sequence in the mini-batch at the t-th time step
    * ht_1: the hidden states (i.e., the activations of the recurrent layer) at the end of the (t-1)th time step, a float torch tensor of shape (n, h)
    ---- Outputs: --------
    * zt: the linear logits of the recurrent layer at the t-th time step on a mini-batch of time sequences,  a float torch tensor of shape (n, h)
    ---- Hints: --------
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_zt(self, xt, ht_1):
        ##############################
        ## INSERT YOUR CODE HERE (4.5 points)
        zt = xt @ self.U + ht_1 @ self.V + self.b
        ##############################
        return zt
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_3.py -m RecurrentLayer_compute_zt
        (Mac /Linux): python3 -m pytest -v test_3.py -m RecurrentLayer_compute_zt
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_ht  ------
    ''' Goal: (Recurrent Layer: Tanh Activation) Given the linear logits zt of a recurrent layer at time step t, please use the element-wise hyperbolic tangent function to compute the activations h_(t) (also called hidden states) at time step t. Each element ht[i] is computed as tanh(zt[i])    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * zt: the linear logits of the recurrent layer at the t-th time step on a mini-batch of time sequences,  a float torch tensor of shape (n, h)
    ---- Outputs: --------
    * ht: the hidden states (i.e., the activations of the recurrent layer) at the end of the t-th time step, a float torch tensor of shape (n, h)
    ---- Hints: --------
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_ht(self, zt):
        ##############################
        ## INSERT YOUR CODE HERE (4.5 points)
        ht = th.tanh(zt)
        ##############################
        return ht
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_3.py -m RecurrentLayer_compute_ht
        (Mac /Linux): python3 -m pytest -v test_3.py -m RecurrentLayer_compute_ht
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: forward  ------
    ''' Goal: (Recurrent Layer: Step Forward) Given a recurrent neural network with parameters U, V and b and we have a mini-batch of data samples x_t at time step t. Suppose we have already computed the hidden state h_(t-1) at the previous time step t-1. Please compute the activations (also called hidden state) h_(t) of the recurrent layer for time step t    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * xt: a mini-batch of time sequences at the t-th time step, a float torch tensor of shape (n, p). xt[k] represents the k-th time sequence in the mini-batch at the t-th time step
    * ht_1: the hidden states (i.e., the activations of the recurrent layer) at the end of the (t-1)th time step, a float torch tensor of shape (n, h)
    ---- Outputs: --------
    * ht: the hidden states (i.e., the activations of the recurrent layer) at the end of the t-th time step, a float torch tensor of shape (n, h)
    ---- Hints: --------
    * This problem can be solved using only 2 line(s) of code. More lines are okay.'''
    def forward(self, xt, ht_1):
        ##############################
        ## INSERT YOUR CODE HERE (6.0 points)
        zt = self.compute_zt(xt, ht_1)
        ht = self.compute_ht(zt)
        ##############################
        return ht
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_3.py -m RecurrentLayer_forward
        (Mac /Linux): python3 -m pytest -v test_3.py -m RecurrentLayer_forward
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
'''------------- Class: RNN (15.0 points) -------
    Recurrent Neural Network for Binary Time Sequence Classification.  The structure of the RNN includes one recurrent layer repeating itself for l time steps and a fully-connected layer attached to the last time step of the recurrent layer to predict the label of a time sequence.  (Recurrent layer for time step 1)-> (Recurrent layer for time step 2) -> ...(Recurrent layer for time step t) -> (Fully connected layer) -> predicted label 
'''
''' ---- Class Properties ----
    * p: the number of input features, an integer scalar.
    * h: the number of classes in the classification task, an integer scalar.
    '''
class RNN(nn.Module):
    #------------- Method: __init__  ------
    ''' Goal: create and initialize the module    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * p: the number of input features, an integer scalar
    * h: the number of classes in the classification task, an integer scalar
    * lr: also called alpha, the learning rate of the stochastic gradient descent algorithm, a float scalar, controlling the speed of gradient descent. Note in the slides we use the notation 'alpha' to refer to this value
    * n_epoch: an integer scalar, the number of passes to iterate through all training examples during stochastic gradient descent.
    '''
    def __init__(self, p, h, lr=0.1, n_epoch=100):
        super(RNN, self).__init__()
        # Recurrent Layer
        self.rnn = RecurrentLayer(p=p,h=h)
        # Linear Layer : fully-connected layer to predict the final label (binary classification)
        self.W = nn.Parameter(th.randn(h)/h) # the weights in the fully-connected linear layer
        self.b = nn.Parameter(th.zeros(1)) # the bias b  in the linear layer
        # Loss function for binary classification
        self.loss_fn = nn.BCEWithLogitsLoss() # the loss function for binary classification
        self.optimizer = th.optim.SGD(self.parameters(),lr = lr) # initialize SGD optimizer to handle the gradient descent of the parameters
        self.n_epoch = n_epoch
        
        
    #----------------------------------------------------------
    
    #------------- Method: compute_z  ------
    ''' Goal: (Fully-Connected Layer: Linear Logit) Given the hidden state h_(t) of the recurrent neural network layer at time step t on a mini-batch of time sequences. Suppose the current time step t is the last time step (t=l) of the time sequences, please compute the linear logit z in the second layer (fully-connected layer) on the mini-batch of time sequences    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * ht: the hidden states (i.e., the activations of the recurrent layer) at the end of the t-th time step, a float torch tensor of shape (n, h)
    ---- Outputs: --------
    * z: the linear logits of the fully connected layer (2nd layer) of RNN on a mini-batch of data samples, a float torch vector of length n
    ---- Hints: --------
    * Here we are assuming that the classification task is binary classification. So the linear logit z is a scalar on each time sequence in the mini-batch. 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_z(self, ht):
        ##############################
        ## INSERT YOUR CODE HERE (3.0 points)
        z = ht @ self.W + self.b
        ##############################
        return z
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_3.py -m RNN_compute_z
        (Mac /Linux): python3 -m pytest -v test_3.py -m RNN_compute_z
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: forward  ------
    ''' Goal: (Forward Pass) Given a recurrent neural network with parameters U, V, b_h, W and b, and a mini-batch of time sequences x, where each time sequence has l time steps. Suppose the initial hidden states of the RNN before seeing any data are given as h_(t=0). Please compute the linear logits z of the RNN on the mini-batch of time sequences    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: a mini-batch of time sequences, a float torch tensor of shape (n, l, p). x[k,t] represents the k-th time sequence in the mini-batch at the t-th time step
    * ht: the initial hidden states before processing any data, a float torch tensor of shape (n, h)
    ---- Outputs: --------
    * z: the linear logits of the fully connected layer (2nd layer) of RNN on a mini-batch of data samples, a float torch vector of length n
    ---- Hints: --------
    * Step 1 Recurrent Layer: apply the recurrent layer to each time step of the time sequences in the mini-batch. 
    * Step 2 Fully-connected Layer: compute the linear logit z each time sequence in the mini-batch. 
    * This problem can be solved using only 3 line(s) of code. More lines are okay.'''
    def forward(self, x, ht):
        ##############################
        ## INSERT YOUR CODE HERE (6.0 points)
        for t in range(x.size(1)):
            xt = x.select(1, t)
            ht = self.rnn(xt, ht)
        
        z = self.compute_z(ht)
        ##############################
        return z
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_3.py -m RNN_forward
        (Mac /Linux): python3 -m pytest -v test_3.py -m RNN_forward
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_L  ------
    ''' Goal: Given a recurrent neural network and suppose we have already computed the linear logits z in the second layer (fully-connected layer) in the last time step t on a mini-batch of training samples. Suppose the labels of the training samples are in y. Please compute the average binary cross-entropy loss on the mini-batch of training samples    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * z: the linear logits of the fully connected layer (2nd layer) of RNN on a mini-batch of data samples, a float torch vector of length n
    * y: the binary labels of the time sequences in a mini-batch, a torch integer vector of length n. The value of each element can be 0 or 1
    ---- Outputs: --------
    * L: the average binary cross entropy loss on a mini-batch of training samples, a torch float scalar
    ---- Hints: --------
    * In our problem setting, the classification task is assumed to be binary classification (e.g., predicting 'wake word' or not) instead of multi-class classification (e.g., predicting different types of commands). So the loss function should be binary cross entropy loss instead of multi-class cross entropy loss. 
    * You could use a layer in the __init__() function to compute the loss here. 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_L(self, z, y):
        ##############################
        ## INSERT YOUR CODE HERE (1.5 points)
        L = self.loss_fn(z, y)
        ##############################
        return L
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_3.py -m RNN_compute_L
        (Mac /Linux): python3 -m pytest -v test_3.py -m RNN_compute_L
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: update_parameters  ------
    ''' Goal: (Gradient Descent) Suppose we are given a recurrent neural network with parameters (U, V, bh, W and b) and we have a mini-batch of training data samples (x,y).  Suppose we have already computed the global gradients of the average loss L w.r.t. the parameters on the mini-batch of data samples. Assume that we have already created an optimizer for the parameters. Please update the parameter values using gradient descent. After the update, the global gradients of all the parameters should be set to zero    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    ---- Hints: --------
    * Why no input is given? The optimizer for all parameters of RNN has been already created in __init__() function, you could just use the optimizer to update the paramters, without any input here. 
    * This problem can be solved using only 2 line(s) of code. More lines are okay.'''
    def update_parameters(self):
        ##############################
        ## INSERT YOUR CODE HERE (1.5 points)
        self.optimizer.step()
        self.optimizer.zero_grad()
        ##############################
        
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_3.py -m RNN_update_parameters
        (Mac /Linux): python3 -m pytest -v test_3.py -m RNN_update_parameters
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: train  ------
    ''' Goal: (Training Recurrent Neural Network) Given a training dataset X (time sequences), Y (labels) in a data loader, train the recurrent neural network using mini-batch stochastic gradient descent: iteratively update the parameters using the gradients on each mini-batch of random data samples.  We repeat n_epoch passes over all the training samples    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * data_loader: the PyTorch loader of a dataset
    ---- Hints: --------
    * Step 1 Forward pass: compute the linear logits in the last layer z and the loss L. 
    * Step 2 Back propagation: compute the gradients of all parameters. 
    * Step 3 Gradient descent: update the parameters using gradient descent. 
    * This problem can be solved using only 4 line(s) of code. More lines are okay.'''
    def train(self, data_loader):
        for _ in range(self.n_epoch): # iterate through the dataset n_epoch times
            for mini_batch in data_loader: # iterate through the dataset with one mini-batch of random training samples (x,y) at a time
                x=mini_batch[0] # the feature vectors of the data samples in a mini-batch
                y=mini_batch[1] # the labels of the samples in a mini-batch
                ht= th.zeros(y.size()[0],self.rnn.h) # initialize hidden memory
                ##############################
                ## INSERT YOUR CODE HERE (3.0 points)
                z = self.forward(x, ht)
                L = self.compute_L(z, y)
                L.backward()
                self.update_parameters()
                ##############################
        
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_3.py -m RNN_train
        (Mac /Linux): python3 -m pytest -v test_3.py -m RNN_train
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




