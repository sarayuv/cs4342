




#------------ No New Package --------------
# NOTE: Please don't import any new package. You should be able to solve the problems using only the package(s) imported here.
from problem1 import Multiclass_Classification, SGD
from abc import abstractmethod
import numpy as np
#---------------------------------------------------------


# ---------------------------------------------------------
'''
    Goal of Problem 3: Modular Designs for Deep Neural Networks (30 points)
    In the prevous problem, we implemented a two-layer fully connected neural network with the most intuitive design. If we want to extend to 3 or more layers, the design will no longer work well. In order to implement neural network with more layers (deep neural networks), we have to adopt modular designs. Writing a fully connected neural network with a modular design involves breaking down the network architecture into smaller, reusable components or modules. Modular approaches enhance code reusability, readability, and efficiency in coding. In this problem, we will implement a few common components that can be used to build deep neural networks: 1) Linear Layer: Represents a single linear layer in the network; 2) Activation Function: Applies a non-linear transformation to the output of a layer; 3) Loss Function: Calculates the difference between the predicted and actual outputs. Different from the problem 2, in this problem, we will name the input of each module/layer as x and the output as y, so x and y in different modules have different meanings. This is different from the problem 2 where each variable has a global name, such as z for logits, a for activations.  Another difference is that in the hidden layers, we will use ReLU as the activation function instead of sigmoid function, because for deep neural networks, sigmoid function can cause vanishing gradient problems..
    
'''
# ---------------------------------------------------------

'''------------- Class: NN_Module (0.0 points) -------
    This is the parent class for computational modules in neural networks. A NN module could be a linear layer, an activation, a loss function, or any operator on data 
'''

class NN_Module:
    #------------- Method: __init__  ------
    ''' Goal: Initialize the model object    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * input_size: the number of inputs of this module, an integer scaler
    '''
    def __init__(self, input_size=None):
        self.input_size = input_size
        
        
    #----------------------------------------------------------
    
    #------------- Method: __call__  ------
    ''' Goal: You could call a module/layer as a function. for example, m = NN_Module(),  m(x) will call the forward function in the class    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * *args: the list arguments
    * **kargs: the keywords arguments
    '''
    def __call__(self, *args, **kargs):
        return self.forward(*args,**kargs)
        
        
    #----------------------------------------------------------
    
    #------------- Method: forward  ------
    ''' Goal: compute the output values using input values of the module during forward pass, and store the values that will be needed in the backward pass    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * *args: the input values as a list
    * **kwargs: the input values as keywords
    '''
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
        
        
    #----------------------------------------------------------
    
    #------------- Method: backward  ------
    ''' Goal: compute the gradients of the module during backward pass    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * dL_dy: the partial gradients of the loss function L w.r.t. the output of this module y, a float numpy vector of length output_size.  The i-th element dL_dy[i] represents the partial gradient of the loss function L w.r.t. the i-th output of the module y[i],  d_L / d_y[i]
    '''
    @abstractmethod
    def backward(self, dL_dy):
        pass
        
        
    #----------------------------------------------------------
    
'''------------- Class: Loss_Cross_Entropy (6.0 points) -------
    Let's start with a loss function, which has no parameter and output the final loss of the neural networks. In this class, we will implement a layer for computing the multi-class cross entropy loss in neural networks 
'''
''' ---- Class Properties ----
    * input_size: the number of inputs of the module, an integer scalar. Usually this is the number of classes in the multi-class classification problems..
    '''
class Loss_Cross_Entropy(NN_Module):
    #------------- Method: forward  ------
    ''' Goal: (Value Forward Function) compute the multi-class cross entropy loss L on a data sample. Here x is the input of this module, which is the softmax activiation of the last layer of the neural network on the training data sample. 1) compute the loss as the output, 2) in the backward pass, we will need x and y to compute the gradients, so we need to store the input x into self.x and label y in self.y in this function, so that we can use them later in the backward function.    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: the input softmax activations of one data sample, a numpy vector of length input_size
    * y: the label of one data sample, an integer scalar of value (0,1, or ...c-1), here c is the total number of classes
    ---- Outputs: --------
    * L: the multi-class cross entropy loss on the data sample x,  a float numpy scalar
    ---- Hints: --------
    * this function is closely related to the compute_L function in problem 2. 
    * This problem can be solved using only 3 line(s) of code. More lines are okay.'''
    def forward(self, x, y):
        ##############################
        ## INSERT YOUR CODE HERE (3.0 points)
        self.x = x
        self.y = y
        L = -np.log(x[y]) if x[y] > 0 else 10000000000
        ##############################
        return L
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_3.py -m Loss_Cross_Entropy_forward
        (Mac /Linux): python3 -m pytest -v test_3.py -m Loss_Cross_Entropy_forward
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: backward  ------
    ''' Goal: Back Propagation: This module has no parameter, so we don't need to compute gradients of the parameters here. But in order for the previous layer to compute gradients, we need to compute the gradient of the loss L w.r.t. the input of this layer (i.e., the output of the previous layer), dL_dx, so that the previous layer can use this gradient to continue the backpropagation process.    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * dL_dy: the partial gradients of the final loss L w.r.t. to the output of this function, a float numpy vector of length output_size. For the final loss function, this gradient is usually set to 1.
    ---- Outputs: --------
    * dL_dx: the partial gradients of the final loss L w.r.t. the input of the module, a float numpy vector of length input_size.  The i-th element dL_dx[i] represents the partial gradient of the loss function L w.r.t. the i-th input of linear layer x[i],  d_L / d_x[i]
    ---- Hints: --------
    * Here we assume the training sample is stored in self.x and self.y (in the forward function), you could use them in this function to compute the gradient. 
    * Why do we need dL_dy? It is usually set as 1., but in very rare cases, we may want to pass a different value when we try to stack with other modules, in that case dL_dy should be the gradient of the final loss L w.r.t. the output of this module (y).. 
    * this function is closely related the comptue_dL_da2 in problem2. 
    * This problem can be solved using only 2 line(s) of code. More lines are okay.'''
    def backward(self, dL_dy=1.0):
        ##############################
        ## INSERT YOUR CODE HERE (3.0 points)
        dL_dx = np.zeros_like(self.x)
        dL_dx[self.y] = -dL_dy / np.clip(self.x[self.y], 1e-10, 1.0)
        ##############################
        return dL_dx
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_3.py -m Loss_Cross_Entropy_backward
        (Mac /Linux): python3 -m pytest -v test_3.py -m Loss_Cross_Entropy_backward
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
'''------------- Class: Softmax (6.0 points) -------
    Let's implement the first activation function softmax, which has no parameter. In this class, we will implement a module for computing the softmax function by taking the linear logits of the last linear layer as the input (x), and produce activations as the output (y) 
'''

class Softmax(NN_Module):
    #------------- Method: forward  ------
    ''' Goal: (Value Forward Function) Suppose we have already computed the linear logits in the last layer (x) on a data sample, compute the softmax activation on the data sample. Here x is the input of this module, which is the linear logits of the last layer of the neural network on the training data sample. 1) compute the softmax activations as the output, 2) in the backward pass, we will need activations (y) to compute the gradients, so we need to store the label y in self.y in this function, so that we can use them later in the backward function.    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: the input vector of one data sample for this linear layer, a numpy vector of length input_size
    ---- Outputs: --------
    * y: the multi-class softmax activations on the data sample x,  a float numpy scalar
    ---- Hints: --------
    * this function is closely related to the compute_a2 function in problem 2. 
    * This problem can be solved using only 3 line(s) of code. More lines are okay.'''
    def forward(self, x):
        ##############################
        ## INSERT YOUR CODE HERE (3.0 points)
        e_x = np.exp(x - np.max(x))
        y = e_x / np.sum(e_x)
        self.y = y
        ##############################
        return y
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_3.py -m Softmax_forward
        (Mac /Linux): python3 -m pytest -v test_3.py -m Softmax_forward
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: backward  ------
    ''' Goal: Back Propagation: This module has no parameter, so we don't need to compute gradients of the parameters here. But in order for the previous layer to compute gradients, we need to compute the gradient of the loss L w.r.t. the input of this layer (i.e., the output of the previous layer), dL_dx, so that the previous layer can use this gradient to continue the backpropagation process.    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * dL_dy: the partial gradients of the final loss L w.r.t. to the output of this function, a float numpy vector of length output_size. For the final loss function, this gradient is usually set to 1.
    ---- Outputs: --------
    * dL_dx: the partial gradients of the final loss L w.r.t. the input of the module, a float numpy vector of length input_size.  The i-th element dL_dx[i] represents the partial gradient of the loss function L w.r.t. the i-th input of linear layer x[i],  d_L / d_x[i]
    ---- Hints: --------
    * Here we assume the output of the training sample is stored in self.y (in the forward function), you could use it in this function to compute the gradient. 
    * this function is closely related the compute_da2_dz2 and comptue_dL_dz2 in problem2. 
    * This problem can be solved using only 4 line(s) of code. More lines are okay.'''
    def backward(self, dL_dy):
        ##############################
        ## INSERT YOUR CODE HERE (3.0 points)
        dy_times_y = dL_dy * self.y
        sum_dy_times_y = np.sum(dy_times_y)
        diff = dL_dy - sum_dy_times_y
        dL_dx = self.y * diff
        ##############################
        return dL_dx
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_3.py -m Softmax_backward
        (Mac /Linux): python3 -m pytest -v test_3.py -m Softmax_backward
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
'''------------- Class: ReLU (6.0 points) -------
    Let's implement the second activation function ReLU, which has no parameter. In this class, we will implement a module for computing the ReLU function on each element of a input vector. We can use ReLU layer in hidden layers as the activation function, so that when we have many layers of deep neural networks, gradients will not vanish as easily as sigmoid function. Here the input (x) is a vector of linear logits, and produce activations as the output (y) 
'''

class ReLU(NN_Module):
    #------------- Method: forward  ------
    ''' Goal: (Value Forward Function) Suppose we have already computed the linear logits in the last layer (x) on a data sample, compute the ReLU activation on the data sample. Here x is the input of this module, which is the linear logits of a hidden layer of the neural network on the training data sample. 1) compute the ReLU activations as the output, 2) in the backward pass, we will need the input (x) to compute the gradients, so we need to store the input x in self.x in this function, so that we can use them later in the backward function.    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: the input vector of one data sample for this linear layer, a numpy vector of length input_size
    ---- Outputs: --------
    * y: the ReLU activation on the data sample x,  a float numpy scalar
    ---- Hints: --------
    * This problem can be solved using only 2 line(s) of code. More lines are okay.'''
    def forward(self, x):
        ##############################
        ## INSERT YOUR CODE HERE (3.0 points)
        self.x = x
        y = np.maximum(0, x)
        ##############################
        return y
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_3.py -m ReLU_forward
        (Mac /Linux): python3 -m pytest -v test_3.py -m ReLU_forward
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: backward  ------
    ''' Goal: Back Propagation: This module has no parameter, so we don't need to compute gradients of the parameters here. But in order for the previous layer to compute gradients, we need to compute the gradient of the loss L w.r.t. the input of this layer (i.e., the output of the previous layer), dL_dx, so that the previous layer can use this gradient to continue the backpropagation process.    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * dL_dy: the partial gradients of the final loss L w.r.t. to the output of this function, a float numpy vector of length output_size. For the final loss function, this gradient is usually set to 1.
    ---- Outputs: --------
    * dL_dx: the partial gradients of the final loss L w.r.t. the input of the module, a float numpy vector of length input_size.  The i-th element dL_dx[i] represents the partial gradient of the loss function L w.r.t. the i-th input of linear layer x[i],  d_L / d_x[i]
    ---- Hints: --------
    * Here we assume the input of the training sample is stored in self.x (in the forward function), you could use it in this function to compute the gradient. 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def backward(self, dL_dy):
        ##############################
        ## INSERT YOUR CODE HERE (3.0 points)
        dL_dx = dL_dy.copy()
        dL_dx[self.x <= 0] = 0
        ##############################
        return dL_dx
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_3.py -m ReLU_backward
        (Mac /Linux): python3 -m pytest -v test_3.py -m ReLU_backward
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
'''------------- Class: Linear (12.0 points) -------
    A linear layer in fully connected neural network (not including activation function), the output of this layer is the linear logits, y = W x + b. Different from the previous modules we implemented, this module has parameters (W and b), so during the backpropagation process, we will also need to compute the gradients of the loss dL_dW and dL_db 
'''
''' ---- Class Properties ----
    * input_size: the number of inputs of the layer, an integer scalar.
    * output_size: the number of outputs of the layer, an integer scalar.
    * W: the weight matrix of the linear layer, a float numpy matrix of shape (output_size by input_size).
    * b: the bias values of the linear layer, a float numpy vector of length output_size.
    * dL_dW: the parital gradients of the final loss L w.r.t. the weight matrix of the linear layer, a float numpy matrix of shape (output_size by input_size).
    * dL_db: the parital gradients of the final loss L w.r.t. the bias values of the linear layer, a float numpy vector of length output_size.
    '''
class Linear(NN_Module):
    #------------- Method: __init__  ------
    ''' Goal: Initialize the model object    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * input_size: the number of input features/measurements in each data sample, an integer scaler
    * output_size: the number of outputs, an integer scalar
    '''
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.W = np.random.randn(output_size, input_size) # initialize W randomly using standard normal distribution
        self.b = np.zeros(output_size) # initialize b as all zeros
        self.dL_dW = np.empty((output_size, input_size)) # the gradients of the weights
        self.dL_db = np.empty(output_size) # the gradients of the biases 
        
        
    #----------------------------------------------------------
    
    #------------- Method: forward  ------
    ''' Goal: (Value Forward Function) compute the linear logit z(x) on a data sample x. In the backward pass, we will need x in the gradient computation, so store the input x into self.x, so that we can use it later in the backward function    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: the input vector of one data sample for this linear layer, a numpy vector of length input_size
    ---- Outputs: --------
    * z: the output logits on the data sample x in this linear layer, i.e. z(x), a float numpy vector of length output_size
    ---- Hints: --------
    * this function is closely related to the compute_z1 or compute_z2 in the problem 2. 
    * This problem can be solved using only 2 line(s) of code. More lines are okay.'''
    def forward(self, x):
        ##############################
        ## INSERT YOUR CODE HERE (2.4 points)
        self.x = x
        z = np.dot(self.W, x) + self.b
        ##############################
        return z
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_3.py -m Linear_forward
        (Mac /Linux): python3 -m pytest -v test_3.py -m Linear_forward
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_dL_dW  ------
    ''' Goal: (Global Gradient of Parameter W)  Suppose we have already computed the global gradients of the loss L w.r.t. the linear logits y (output of this layer) on a training sample. Suppose the input of the sample at this layer is x, which has already been stored in the layer (self.x) Please compute the partial gradient of the loss L w.r.t. the weights W in the layer using chain rule and store it in self.dL_dW    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * dL_dy: the partial gradients of the loss function L w.r.t. the linear logits y, a float numpy vector of length output_size.  The i-th element dL_dy[i] represents the partial gradient of the loss function L w.r.t. the i-th linear logit y[i],  d_L / d_y[i]
    ---- Hints: --------
    * This function is closed related to (compute_dz1_dW1 or compute_dz2_dW2) and (compute_dL_W2 or compute_dL_dW1) in problem 2. 
    * You could use the outer product of two vectors (np.outer(a,b) for two vectors a and b) in this problem.. 
    * the inputs are already stored in self.x after forward pass. 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_dL_dW(self, dL_dy):
        ##############################
        ## INSERT YOUR CODE HERE (2.4 points)
        self.dL_dW = np.outer(dL_dy, self.x) 
        ##############################
        
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_3.py -m Linear_compute_dL_dW
        (Mac /Linux): python3 -m pytest -v test_3.py -m Linear_compute_dL_dW
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_dL_dx  ------
    ''' Goal: (Global Gradient for Inputs)  Suppose we have already computed the global gradients of the loss L w.r.t. the linear logits y (output of this layer) on a training sample. Suppose the input of the sample at this layer is x. The parameters of this linear layer are stored in self.W and self.b. Please compute the partial gradient of the loss L w.r.t. the input of this layer (x) using chain rule    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * dL_dy: the partial gradients of the loss function L w.r.t. the linear logits y, a float numpy vector of length output_size.  The i-th element dL_dy[i] represents the partial gradient of the loss function L w.r.t. the i-th linear logit y[i],  d_L / d_y[i]
    ---- Outputs: --------
    * dL_dx: the partial gradients of the loss function L w.r.t. the input of this linear layer x, a float numpy vector of length input_size.  The i-th element dL_dx[i] represents the partial gradient of the loss function L w.r.t. the i-th input of the layer x[i],  d_L / d_x[i]
    ---- Hints: --------
    * this function is closely related to the compute_dL_da1 and compute_dz2_da1 functions in problem 2 . 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_dL_dx(self, dL_dy):
        ##############################
        ## INSERT YOUR CODE HERE (2.4 points)
        dL_dx = np.dot(self.W.T, dL_dy)
        ##############################
        return dL_dx
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_3.py -m Linear_compute_dL_dx
        (Mac /Linux): python3 -m pytest -v test_3.py -m Linear_compute_dL_dx
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: backward  ------
    ''' Goal: Back Propagation of this linea layer: Suppose we are given a fully-connected neural network with parameters and we have a training data sample, where the input to this linear layer is x (x could be the output of the previous layer or the raw feature of the data sample).  Suppose we have already computed the loss in the forward pass and stored the data in self.x. Please compute the global gradients of the loss w.r.t. the parameters W and b on the data sample using back propagation, store them in the object attributes, self.dL_dW and self.dL_db. Then in order for the previous layer to compute gradients, we also need to compute the gradient of the loss L w.r.t. the input of this layer (i.e., the output of the previous layer), dL_dx. The previous layer will need this gradient to contiue the backpropagation process.    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * dL_dy: the partial gradients of the final loss L w.r.t. the linear logits y in this linear layer, a float numpy vector of length output_size.  The i-th element dL_dy[i] represents the partial gradient of the loss function L w.r.t. the i-th linear logit y[i],  d_L / d_y[i]
    ---- Outputs: --------
    * dL_dx: the partial gradients of the final loss L w.r.t. the input of the linear layer, a float numpy vector of length input_size.  The i-th element dL_dx[i] represents the partial gradient of the loss function L w.r.t. the i-th input of linear layer x[i],  d_L / d_x[i]
    ---- Hints: --------
    * This problem can be solved using only 3 line(s) of code. More lines are okay.'''
    def backward(self, dL_dy):
        ##############################
        ## INSERT YOUR CODE HERE (2.4 points)
        self.dL_dW = np.outer(dL_dy, self.x)
        self.dL_db = dL_dy
        dL_dx = np.dot(self.W.T, dL_dy)
        ##############################
        return dL_dx
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_3.py -m Linear_backward
        (Mac /Linux): python3 -m pytest -v test_3.py -m Linear_backward
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: update_parameters  ------
    ''' Goal: Once the gradients of the parameters are computed using forward-backward passes, now we can update the parameters (W and b) using stochastic gradient descent. Update the parameters (W and b) using the given learning rate and perform one step of gradient descent    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * lr: learning rate
    ---- Hints: --------
    * the gradients of the parameters W and b are already computed and stored in self.dL_dW, self.dL_db. Now you could use them to update the paramters in self.W and self.b using one step of gradient descent. 
    * This problem can be solved using only 2 line(s) of code. More lines are okay.'''
    def update_parameters(self, lr):
        ##############################
        ## INSERT YOUR CODE HERE (2.4 points)
        self.W -= lr * self.dL_dW
        self.b -= lr * self.dL_db 
        ##############################
        
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_3.py -m Linear_update_parameters
        (Mac /Linux): python3 -m pytest -v test_3.py -m Linear_update_parameters
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
'''------------- Class: Fully_Connected_NN_3layers (0.0 points) -------
    Now this is a showcase of how we can use the modules defined above to build Fully-connected neural networks with multiple linear layers. For simplicity, let's focus on 3 linear layers. The activations in the first two hidden layers are ReLU, while the last layer uses softmax as the activation function. 
'''
''' ---- Class Properties ----
    * p: the number of input features, an integer scalar.
    * c: the number of classes in the classification task, an integer scalar.
    * n_epoch: an integer scalar, the number of passes to iterate through all training examples during stochastic gradient descent..
    * lr: the learning rate of the stochastic gradient descent algorithm, a float scalar, controlling the speed of gradient descent.
    * h1: the number of hidden units in the 1st linear layer (or the number of outputs in the first layer).
    * h2: the number of hidden units in the 2nd linear layer (or the number of outputs in the second layer).
    '''
class Fully_Connected_NN_3layers(NN_Module,Multiclass_Classification,SGD):
    #------------- Method: __init__  ------
    ''' Goal: Initialize the model object    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * p: the number of features/measurements in each data sample, an integer scaler
    * c: the number of classes in the classification task, an integer scalar
    * h1: the number of hidden units in the 1st linear layer (or the number of outputs in the first layer)
    * h2: the number of hidden units in the 2nd linear layer (or the number of outputs in the second layer)
    * n_epoch: an integer scalar, the number of passes to iterate through all training examples during stochastic gradient descent.
    * lr: also called alpha, the learning rate of the stochastic gradient descent algorithm, a float scalar, controlling the speed of gradient descent. Note in the slides we use the notation 'alpha' to refer to this value
    '''
    def __init__(self, p, c, h1, h2, n_epoch=100, lr=0.001):
        Multiclass_Classification.__init__(self,p=p,c=c)
        SGD.__init__(self,n_epoch=n_epoch,lr=lr)
        self.h1 = h1
        self.h2 = h2
        # define layers of modules
        self.linear1 = Linear(p,h1) # the first linear layer
        self.relu1 = ReLU() # ReLU activation used in the first layer
        self.linear2 = Linear(h1,h2) # the second linear layer
        self.relu2 = ReLU() # ReLU activation used in the second layer
        self.linear3 = Linear(h2,c) # the last linear layer
        self.softmax= Softmax() # Softmax activation used in the last layers
        self.loss = Loss_Cross_Entropy() # Loss
        
        
    #----------------------------------------------------------
    
    #------------- Method: forward  ------
    ''' Goal: (Forward Pass) Suppose we are given a fully-connected neural network. Given a data sample (x), please compute the activations a1(x) and a2(x) on the sample    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: the feature vector of one data sample, a numpy vector of length p
    ---- Outputs: --------
    * a: the softmax activations, a float numpy vector of length c
    '''
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        a = self.softmax(x)
        return a
        
    #----------------------------------------------------------
    
    #------------- Method: backward  ------
    ''' Goal: Back Propagation can be implemented using the modules defined above    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * dL_dy: the gradient of the L loss w.r.t. the output of the neural network (softmax activations)
    '''
    def backward(self, dL_dy):
        d = self.softmax.backward(dL_dy)
        d = self.linear3.backward(d)
        d = self.relu2.backward(d)
        d = self.linear2.backward(d)
        d = self.relu1.backward(d)
        self.linear1.backward(d)
        
        
    #----------------------------------------------------------
    
    #------------- Method: train  ------
    ''' Goal: (Train a Fully-Connected Neural Network) Given a training dataset, train the Fully Connected Neural Network by iteratively updating the weights W and biases b using the gradients computed over each data instance    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * X: the feature matrix of training instances, a float numpy matrix of shape (n by p)
    * Y: the labels of training instance, a numpy integer numpy array of length n. The values can be 0, 1, ..., or c-1
    '''
    def train(self, X, Y):
        n = X.shape[0] # n: the number of training samples
        for _ in range(self.n_epoch): # iterate through the dataset n_epoch times
            indices = np.random.permutation(n) # shuffle the indices of all samples
            for i in indices: # iterate through each random training sample (x,y)
                x=X[i] # the feature vector of the i-th random sample
                y=Y[i] # the label of the i-th random sample
                a = self(x) # forward
                self.loss(a,y) # compute loss
                dL_da = self.loss.backward() # backward pass
                self.backward(dL_da) # the gradients of all linear layers have been computed
                self.linear3.update_parameters(self.lr) # update parameters in the linear layer
                self.linear2.update_parameters(self.lr) # update parameters in the linear layer
                self.linear1.update_parameters(self.lr) # update parameters in the linear layer
        
        
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




