
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
    Goal of Problem 1: Long-Short Term Memory (LSTM) Method for Multi-class Time Sequence Classification (with PyTorch) (30 points)
    In this problem, you will implement another kind of recurrent neural network (LSTM, Long-Short Term Memory) for multi-class sequence classification problems.  Here we assume that each long time sequence is assigned with one multi-class label.  For example, in long audio classification, each time sequence is a long clip of audio recording, and the label of the sequence is one out of the multiple possible categories (0 check time, 1 check email, 2 add calendar event, 3 turn on the light, etc).  The goal of this problem is to learn the details of LSTM by building LSTM from scratch.  The structure of the LSTM includes one recurrent layer repeating itself for l time steps and a fully-connected layer attached to the last time step of the recurrent layer to predict the label of a time sequence.  (Recurrent layer for time step 1)-> (Recurrent layer for time step 2) -> ...(Recurrent layer for time step t) -> (Fully connected layer) -> predicted label.
    
'''
# ---------------------------------------------------------

'''------------- Class: Attention_Gate (9.0 points) -------
    An LSTM layer involves multiple gate functions. Most of the gate functions can be considered as attention module, which generate a softmax probability/attention weight as the output. For example, the forget gate, input gate, output gate are all gate functions based upon attention module. Here we first implement an attention module in LSTM, then we can use it to serve as different gate modules in LSTM layers. 
'''
''' ---- Class Properties ----
    * p: the number of input features at each time step of a time sequence, an integer scalar.
    * h: the number of neurons in the cell/hidden states (or the activations of the recurrent layer), an integer scalar.
    * W: the weights of the gates on the current input xt and past memory ht_1, a float torch matrix of shape (p+h, h).
    * b: the biases of the gates, a float torch vector of length h.
    '''
class Attention_Gate(nn.Module):
    #------------- Method: __init__  ------
    ''' Goal: create and initialize the module    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * p: the number of input features at each time step of a time sequence, an integer scalar
    * h: the number of neurons in the hidden states or cell states (or the activations of the recurrent layer), an integer scalar
    '''
    def __init__(self, p, h):
        super(Attention_Gate, self).__init__()
        self.p = p
        self.h = h
        self.W = nn.Parameter(th.randn(p+h,h)) # initialize the parameter Weights W randomly
        self.b = nn.Parameter(th.zeros(h)) # initialize the parameter bias b as zero
        
        
    #----------------------------------------------------------
    
    #------------- Method: compute_xh  ------
    ''' Goal: (Recurrent Layer: concatenating xt with ht_1) Given a mini-batch of time sequences at the t-th time step (xt). Suppose we have already computed the hidden states h_(t-1) at the previous time step (t-1). Please concatenate the two vectors for each sample in a mini-batch. This design is different from RNN, where we define the weights on the input xt and hidden state ht_1 separately as U and V. Here we want to combine the weights into on matrix W, so the inputs (xt and ht_1) should also be concatenated.    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * xt: a mini-batch of time sequences at the t-th time step, a float torch tensor of shape (n, p). xt[k] represents the k-th time sequence in the mini-batch at the t-th time step
    * ht_1: the hidden states at the end of the (t-1)th time step, a float torch tensor of shape (n, h)
    ---- Outputs: --------
    * xh: the concatenation of xt with ht_1 on a mini-batch of time sequences at the t-th time step, a float torch tensor of shape (n, p+h). xh[i] represents the vector concatenating the i-th time sequence input feature xt[i], with the hidden states on the i-th time sequence after the (t-1)-th time step: ht_1. For example, if we have two samples in a mini-batch at time step t, xt= [ [1,2], [3,4] ]. Suppose at the end of step t-1, the hidden states ht_1 on the two samples are [ [0.1,0.2,0.3], [0.4,0.5,0.6] ]. Then after concatenation, xh = [ [1,2,0.1,0.2,0.3], [3,4,0.4,0.5,0.6] ]
    ---- Hints: --------
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_xh(self, xt, ht_1):
        ##############################
        ## INSERT YOUR CODE HERE (1.8 points)
        xh = th.cat((xt, ht_1), dim=1)
        ##############################
        return xh
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m Attention_Gate_compute_xh
        (Mac /Linux): python3 -m pytest -v test_1.py -m Attention_Gate_compute_xh
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_z  ------
    ''' Goal: (Linear Logits) Given attention-based gates (such as forget gate, or input game) in an LSTM with parameters weights W and biases b. We have a mini-batch of data samples xt at the t-th time step. Suppose we have already computed the hidden states h_(t-1) at the previous time step (t-1). Suppose we have already concatenated xt and h_(t-1) into xh. Please compute the linear logits z for the gates at the t-th time step on the mini-batch of data samples    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * xh: the concatenation of xt with ht_1 on a mini-batch of time sequences at the t-th time step, a float torch tensor of shape (n, p+h). xh[i] represents the vector concatenating the i-th time sequence input feature xt[i], with the hidden states on the i-th time sequence after the (t-1)-th time step: ht_1. For example, if we have two samples in a mini-batch at time step t, xt= [ [1,2], [3,4] ]. Suppose at the end of step t-1, the hidden states ht_1 on the two samples are [ [0.1,0.2,0.3], [0.4,0.5,0.6] ]. Then after concatenation, xh = [ [1,2,0.1,0.2,0.3], [3,4,0.4,0.5,0.6] ]
    ---- Outputs: --------
    * z: the linear logits of the gates at the t-th time step on a mini-batch of data samples, a float torch tensor of shape (n, h)
    ---- Hints: --------
    * The parameters W and b are accessable through self.W and self.b. 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_z(self, xh):
        ##############################
        ## INSERT YOUR CODE HERE (1.8 points)
        z = xh @ self.W + self.b
        ##############################
        return z
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m Attention_Gate_compute_z
        (Mac /Linux): python3 -m pytest -v test_1.py -m Attention_Gate_compute_z
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_a  ------
    ''' Goal: (Activation, or Attention) Given the linear logits z of the gates at time step t on a mini-batch of training samples, please use the element-wise sigmoid function to compute the gates' activations/attentions (a) at time step t. Each element a[i] is computed as sigmoid(z[i])    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * z: the linear logits of the gates at the t-th time step on a mini-batch of data samples, a float torch tensor of shape (n, h)
    ---- Outputs: --------
    * a: the gates' attentions (i.e., the activations of the gates) at the t-th time step, a float torch tensor of shape (n, h)
    ---- Hints: --------
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_a(self, z):
        ##############################
        ## INSERT YOUR CODE HERE (1.8 points)
        a = th.sigmoid(z)
        ##############################
        return a
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m Attention_Gate_compute_a
        (Mac /Linux): python3 -m pytest -v test_1.py -m Attention_Gate_compute_a
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: forward  ------
    ''' Goal: (Forward Pass) Given a type of attention gates in LSTM model (such as forget gate), and we have a mini-batch of data samples x_t at time step t. Suppose we have already computed the hidden state h_(t-1) at the previous time step t-1. Please compute the activations/attentions of the gates a_(t) for time step t    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * xt: a mini-batch of time sequences at the t-th time step, a float torch tensor of shape (n, p). xt[k] represents the k-th time sequence in the mini-batch at the t-th time step
    * ht_1: the hidden states at the end of the (t-1)th time step, a float torch tensor of shape (n, h)
    ---- Outputs: --------
    * a: the attentions or activations of the gates for the t-th time step, a float torch tensor of shape (n, h)
    ---- Hints: --------
    * This problem can be solved using only 3 line(s) of code. More lines are okay.'''
    def forward(self, xt, ht_1):
        ##############################
        ## INSERT YOUR CODE HERE (3.6 points)
        xh = self.compute_xh(xt, ht_1)
        z = self.compute_z(xh)
        a = self.compute_a(z)
        ##############################
        return a
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m Attention_Gate_forward
        (Mac /Linux): python3 -m pytest -v test_1.py -m Attention_Gate_forward
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
'''------------- Class: Candidate_Change (3.0 points) -------
    In LSTM, the candidate changes to the cell memory is very similar to the attention gates (input gates, forget gates), except that the activation function is tanh() instead of sigmoid. Here we just need to overwrite the activation function of the attention gate class to create a module for computing candidate changes to cell states in LSTM 
'''

class Candidate_Change(Attention_Gate):
    #------------- Method: compute_a  ------
    ''' Goal: (Activation, or Attention) Given the linear logits z of the gates at time step t on a mini-batch of training samples, please use the element-wise sigmoid function to compute the gates' activations/attentions (a) at time step t. Each element a[i] is computed as tanh(z[i])    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * z: the linear logits of the gates at the t-th time step on a mini-batch of data samples, a float torch tensor of shape (n, h)
    ---- Outputs: --------
    * a: the gates' attentions (i.e., the activations of the gates) at the t-th time step, a float torch tensor of shape (n, h)
    ---- Hints: --------
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_a(self, z):
        ##############################
        ## INSERT YOUR CODE HERE (3.0 points)
        a = th.tanh(z)
        ##############################
        return a
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m Candidate_Change_compute_a
        (Mac /Linux): python3 -m pytest -v test_1.py -m Candidate_Change_compute_a
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
'''------------- Class: LSTM_Layer (6.0 points) -------
    LSTM Layer is the core component responsible for processing sequential data (on each time step) by maintaining and updating a cell state and hidden state over time. The cell state and hidden state capture information about the sequence seen up to the current time step with long and short memories and are updated recursively as new input is processed. 
'''
''' ---- Class Properties ----
    * p: the number of input features, an integer scalar.
    * h: the number of neurons in the cell/hidden states (or the activations of the recurrent layer), an integer scalar.
    '''
class LSTM_Layer(nn.Module):
    #------------- Method: __init__  ------
    ''' Goal: create and initialize the module    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * p: the number of input features, an integer scalar
    * h: the number of neurons in the cell/hidden states, an integer scalar
    '''
    def __init__(self, p, h):
        super(LSTM_Layer, self).__init__()
        self.h=h
        self.forget_gate = Attention_Gate(p=p,h=h)
        self.forget_gate = Attention_Gate(p=p,h=h)
        self.input_gate = Attention_Gate(p=p,h=h)
        self.output_gate = Attention_Gate(p=p,h=h)
        self.candidate_change= Candidate_Change(p=p,h=h)
        
        
    #----------------------------------------------------------
    
    #------------- Method: compute_Ct  ------
    ''' Goal: (Update Cell State) Suppose we have the forget gates (f_t), input gates (i_t) and candidate cell states (C_c) at time step t.  We also have the old cell states (Ct_1) at the end of time step t-1, please compute the new cell state (Ct) for the time step t    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * f_t: the forget gates (i.e., the activations of the forget gates) at the t-th time step, a float torch tensor of shape (n, h)
    * i_t: the input gates (i.e., the activations of the input gates) at the t-th time step, a float torch tensor of shape (n, h)
    * C_c: the candidate cell states (i.e., the activations in the candidate cell states) at the t-th time step on a mini-batch of data samples, a float torch tensor of shape (n, h)
    * Ct_1: the old cell states of the LSTM cells at the end of (t-1)-th time step on a mini-batch of data samples, a float torch tensor of shape (n, h)
    ---- Outputs: --------
    * Ct: the new cell states of the LSTM cells at the end of t-th time step on a mini-batch of data samples, a float torch tensor of shape (n, h)
    ---- Hints: --------
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_Ct(self, f_t, i_t, C_c, Ct_1):
        ##############################
        ## INSERT YOUR CODE HERE (1.8 points)
        Ct = f_t * Ct_1 + i_t * C_c
        ##############################
        return Ct
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m LSTM_Layer_compute_Ct
        (Mac /Linux): python3 -m pytest -v test_1.py -m LSTM_Layer_compute_Ct
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_ht  ------
    ''' Goal: (Output New Hidden States) Given the new cell states Ct of the LSTM recurrent layer at time step t. Suppose we have also computed the output gates for time step t. Please compute the new hidden states h_(t) at time step t    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * Ct: the new cell states of the LSTM cells at the end of t-th time step on a mini-batch of data samples, a float torch tensor of shape (n, h)
    * o_t: the output gates (i.e., the activations of the output gates) at the t-th time step, a float torch tensor of shape (n, h)
    ---- Outputs: --------
    * ht: the output hidden states at the end of the t-th time step, a float torch tensor of shape (n, h)
    ---- Hints: --------
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_ht(self, Ct, o_t):
        ##############################
        ## INSERT YOUR CODE HERE (1.8 points)
        ht = o_t * th.tanh(Ct)
        ##############################
        return ht
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m LSTM_Layer_compute_ht
        (Mac /Linux): python3 -m pytest -v test_1.py -m LSTM_Layer_compute_ht
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: forward  ------
    ''' Goal: (Forward) Given an LSTM recurrent layer and we have a mini-batch of data samples x_t at time step t. Suppose we have already computed the old cell states C_(t-1) and the hidden states h_(t-1) at the previous time step t-1. Please compute the new cell states Ct and hidden states h_(t) of the recurrent layer on the mini-batch of samples for time step t    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * xt: a mini-batch of time sequences at the t-th time step, a float torch tensor of shape (n, p). xt[k] represents the k-th time sequence in the mini-batch at the t-th time step
    * Ct_1: the old cell states of the LSTM cells at the end of (t-1)-th time step on a mini-batch of data samples, a float torch tensor of shape (n, h)
    * ht_1: the hidden states (i.e., the activations of the recurrent layer) at the end of the (t-1)th time step, a float torch tensor of shape (n, h)
    ---- Outputs: --------
    * Ct: the new cell states of the LSTM cells at the end of t-th time step on a mini-batch of data samples, a float torch tensor of shape (n, h)
    * ht: the new hidden states at the end of the t-th time step, a float torch tensor of shape (n, h)
    ---- Hints: --------
    * It's easier to follow a certain order to compute all the values: f_t, i_t .... 
    * This problem can be solved using only 6 line(s) of code. More lines are okay.'''
    def forward(self, xt, Ct_1, ht_1):
        ##############################
        ## INSERT YOUR CODE HERE (2.4 points)
        f_t = self.forget_gate(xt, ht_1)
        i_t = self.input_gate(xt, ht_1)
        o_t = self.output_gate(xt, ht_1)
        C_c = self.candidate_change(xt, ht_1)
        Ct = self.compute_Ct(f_t, i_t, C_c, Ct_1)
        ht = self.compute_ht(Ct, o_t)
        ##############################
        return Ct, ht
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m LSTM_Layer_forward
        (Mac /Linux): python3 -m pytest -v test_1.py -m LSTM_Layer_forward
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
'''------------- Class: Linear_Layer (6.0 points) -------
    The second layer of LSTM, which is a linear layer for multiple classes as the output. 
'''

class Linear_Layer(nn.Module):
    #------------- Method: __init__  ------
    ''' Goal: create and initialize the module    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * h: the number of neurons in the hidden states or cell states (or the activations of the recurrent layer), an integer scalar
    * c: the number of classes in the classification task, an integer scalar
    '''
    def __init__(self, h, c):
        super(Linear_Layer, self).__init__()
        self.h = h
        self.W = nn.Parameter(th.randn(h,c)) # initialize the parameter Weights W randomly
        self.b = nn.Parameter(th.zeros(c)) # initialize the parameter bias b as zero
        
        
    #----------------------------------------------------------
    
    #------------- Method: forward  ------
    ''' Goal: (Fully-Connected Layer: Linear Logit) Given the hidden state h_(t) of the recurrent layer at time step t on a mini-batch of time sequences. Suppose the current time step t is the last time step (t=l) of the time sequences, please compute the linear logit z in the second layer (fully-connected layer) on the mini-batch of time sequences    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * ht: the output hidden states at the end of the t-th time step, a float torch tensor of shape (n, h)
    ---- Outputs: --------
    * z: the linear logits of the fully connected layer (2nd layer) on a mini-batch of data samples, a float torch matrix of shape (n,c)
    ---- Hints: --------
    * Here we are assuming that the classification task is multi-class classification. So the linear logit z is a vector on each time sequence in the mini-batch. 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def forward(self, ht):
        ##############################
        ## INSERT YOUR CODE HERE (6.0 points)
        z = ht @ self.W + self.b
        ##############################
        return z
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m Linear_Layer_forward
        (Mac /Linux): python3 -m pytest -v test_1.py -m Linear_Layer_forward
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
'''------------- Class: LSTM (6.0 points) -------
    In this class, we will implement LSTM (Long-Short Term Memory) for multi-class sequence classification problems.  Here we assume that each long time sequence is assigned with one multi-class label.  For example, in long audio classification, each time sequence is a long clip of audio recording, and the label of the sequence is one out of the multiple possible categories (0 check time, 1 check email, 2 add calendar event, 3 turn on the light, etc).  The goal of this problem is to learn the details of LSTM by building LSTM from scratch.  The structure of the LSTM includes one recurrent layer repeating itself for l time steps and a fully-connected layer attached to the last time step of the recurrent layer to predict the label of a time sequence.  (Recurrent layer for time step 1)-> (Recurrent layer for time step 2) -> ...(Recurrent layer for time step t) -> (Fully connected layer) -> predicted label 
'''
''' ---- Class Properties ----
    * p: the number of input features, an integer scalar.
    * h: the number of classes in the classification task, an integer scalar.
    '''
class LSTM(nn.Module):
    #------------- Method: __init__  ------
    ''' Goal: create and initialize the module    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * p: the number of input features, an integer scalar
    * h: the number of classes in the classification task, an integer scalar
    * c: the number of classes in the classification task, an integer scalar
    * lr: also called alpha, the learning rate of the stochastic gradient descent algorithm, a float scalar, controlling the speed of gradient descent. Note in the slides we use the notation 'alpha' to refer to this value
    * n_epoch: an integer scalar, the number of passes to iterate through all training examples during stochastic gradient descent.
    '''
    def __init__(self, p, h, c, lr=0.1, n_epoch=100):
        super(LSTM, self).__init__()
        # Recurrent Layer
        self.lstm = LSTM_Layer(p=p,h=h)
        # Linear Layer : fully-connected layer to predict the final label (multi-class classification)
        self.linear = Linear_Layer(h=h,c=c)
        # Loss function for binary classification
        self.loss_fn = nn.CrossEntropyLoss() # the loss function for multi-class classification
        self.optimizer = th.optim.SGD(self.parameters(),lr = lr) # initialize SGD optimizer to handle the gradient descent of the parameters
        self.n_epoch = n_epoch
        
        
    #----------------------------------------------------------
    
    #------------- Method: forward  ------
    ''' Goal: (Forward Pass) Given an LSTM model, and a mini-batch of time sequences x, where each time sequence has l time steps. Suppose the initial hidden states before seeing any data are given as h_(t=0). Similarly the initial cell states before seeing any data are given as C_(t=0)  Please compute the linear logits z of the LSTM on the mini-batch of time sequences    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: a mini-batch of time sequences, a float torch tensor of shape (n, l, p). x[k,t] represents the k-th time sequence in the mini-batch at the t-th time step
    * Ct: the initial cell states before processing any data, a float torch tensor of shape (n, h)
    * ht: the initial hidden states before processing any data, a float torch tensor of shape (n, h)
    ---- Outputs: --------
    * z: the linear logits of the fully connected layer (2nd layer) on a mini-batch of data samples, a float torch matrix of shape (n,c)
    ---- Hints: --------
    * Step 1 Recurrent Layer: apply the LSTM recurrent layer to each time step of the time sequences in the mini-batch. 
    * Step 2 Fully-connected Layer: compute the linear logit z each time sequence in the mini-batch. 
    * This problem can be solved using only 3 line(s) of code. More lines are okay.'''
    def forward(self, x, Ct, ht):
        ##############################
        ## INSERT YOUR CODE HERE (2.4 points)
        for t in range(x.shape[1]):
            xt = x.select(1, t)
            Ct, ht = self.lstm(xt, Ct, ht)
        z = self.linear(ht)
        ##############################
        return z
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m LSTM_forward
        (Mac /Linux): python3 -m pytest -v test_1.py -m LSTM_forward
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_L  ------
    ''' Goal: Given an LSTM model and suppose we have already computed the linear logits z in the second layer (fully-connected layer) in the last time step t on a mini-batch of training samples. Suppose the labels of the training samples are in y. Please compute the average multi-class cross-entropy loss on the mini-batch of training samples    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * z: the linear logits of the fully connected layer (2nd layer) on a mini-batch of data samples, a float torch matrix of shape (n,c)
    * y: the labels of the time sequences in a mini-batch, a torch integer vector of length n. The value of each element can be 0 or 1,..., c-1
    ---- Outputs: --------
    * L: the average cross entropy loss on a mini-batch of training samples, a torch float scalar
    ---- Hints: --------
    * In this problem setting, the classification task is assumed to be multi-class classification (e.g., predicting different types of commands). So the loss function should be multi-class cross entropy loss. 
    * You could use a layer in the __init__() function to compute the loss here. 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_L(self, z, y):
        ##############################
        ## INSERT YOUR CODE HERE (1.2 points)
        L = self.loss_fn(z, y)
        ##############################
        return L
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m LSTM_compute_L
        (Mac /Linux): python3 -m pytest -v test_1.py -m LSTM_compute_L
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: update_parameters  ------
    ''' Goal: (Gradient Descent) Suppose we are given an LSTM model with parameters (W_f, b_f, W_i, b_i, W_o, b_o, W_c, b_c, W and b) and we have a mini-batch of training data samples (x,y).  Suppose we have already computed the global gradients of the average loss L w.r.t. the parameters on the mini-batch of data samples. Assume that we have already created an optimizer for the parameters. Please update the parameter values using gradient descent. After the update, the global gradients of all the parameters should be set to zero    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    ---- Hints: --------
    * Why no input is given? The optimizer for all parameters of RNN has been already created in __init__() function, you could just use the optimizer to update the paramters, without any input here. 
    * This problem can be solved using only 2 line(s) of code. More lines are okay.'''
    def update_parameters(self):
        ##############################
        ## INSERT YOUR CODE HERE (1.2 points)
        self.optimizer.step()
        self.optimizer.zero_grad()
        ##############################
        
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m LSTM_update_parameters
        (Mac /Linux): python3 -m pytest -v test_1.py -m LSTM_update_parameters
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: train  ------
    ''' Goal: (Training LSTM) Given a training dataset X (time sequences), Y (labels) in a data loader, train an LSTM model using mini-batch stochastic gradient descent: iteratively update the parameters using the gradients on each mini-batch of random data samples.  We repeat n_epoch passes over all the training samples    '''
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
                Ct= th.zeros(y.size()[0],self.lstm.h) # initialize cell memory
                ht= th.zeros(y.size()[0],self.lstm.h) # initialize hidden memory
                ##############################
                ## INSERT YOUR CODE HERE (1.2 points)
                z = self.forward(x, Ct, ht)
                L = self.compute_L(z, y)
                L.backward()
                self.update_parameters()
                ##############################
        
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m LSTM_train
        (Mac /Linux): python3 -m pytest -v test_1.py -m LSTM_train
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    

'''-------- TEST problem1.py file: (30 points) ----------
Please type the following command in your terminal to test the correctness of all the above functions in this file:
        (Windows OS): python -m pytest -v test_1.py
        (Mac /Linux): python3 -m pytest -v test_1.py
------------------------------------------------------'''






