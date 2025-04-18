




#------------ No New Package --------------
# NOTE: Please don't import any new package. You should be able to solve the problems using only the package(s) imported here.
import torch as th
import torch.nn as nn
#---------------------------------------------------------


# ---------------------------------------------------------
'''
    Goal of Problem 2: Convolutional Neural Network for Binary Image Classification (using PyTorch) (40 points)
    In this problem, you will implement a convolutional neural network (CNN) with two convolution layers with max pooling and ReLU activations, and one fully-connected layer at the end to predict the label. The classification task is that given an input RGB color image, predict the binary label of the image (e.g., whether the image is the owner of the smartphone or not).  The goal of this problem is to learn the details of convolutional neural network by building CNN from scratch. The structure of the CNN is (Conv layer 1) -> ReLU -> maxpooling -> (Conv layer 2) -> ReLU -> maxpooling -> (Fully-connected layer).
    
'''
# ---------------------------------------------------------

'''------------- Class: Conv2d_v1 (2.0 points) -------
    (Example A: Convolutional Layer with 1 filter and 1 input/color channel on 1 image) Let's first get familiar with the 2D Convolution. Let's start with one filter and one image with one input channel. Given a convolutional filter (with weights W and bias  b) and an image x with one color channel, height h and width w, please compute the 2D convolution with the filter on the image. Here we assume that stride=1 and padding = 0 
'''
''' ---- Class Properties ----
    * s: the size of the convolutional filter, an integer scalar.
    * W: the weights of 1 convolutional filter with 1 color/input channel,  a float torch matrix of shape (s, s) .
    * b: the bias of 1 convolutional filter with 1 color/input channel,  a float scalar.
    '''
class Conv2d_v1(nn.Module):
    #------------- Method: __init__  ------
    ''' Goal: create and initialize the module    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * s: the size of the convolutional filter, an integer scalar
    '''
    def __init__(self, s):
        super(Conv2d_v1, self).__init__()
        self.s = s
        self.W = nn.Parameter(th.randn(s,s)/s) # initialize the parameter Weights W randomly
        self.b = nn.Parameter(th.zeros(1)) # initialize the parameter bias b as zero
        
        
    #----------------------------------------------------------
    
    #------------- Method: forward  ------
    ''' Goal: Given a convolutional filter (with weights W and bias  b) and an image x with one color channel, height h and width w, please compute the 2D convolution with the filter on the image. Here we assume that stride=1 and padding = 0    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: one input image with 1 input/color channel with hight h and width w of pixels, a float torch tensor of shape (h, w)
    ---- Outputs: --------
    * z: the linear logits of the convolutional layer with 1 filter and 1 channel on one image,  a float torch matrix of shape (h-s+1, w-s+1)
    ---- Hints: --------
    * You could use A.size() to get the shape of a torch tensor A. 
    * You could use th.empty() to create an empty torch tensor. 
    * In order to connect the global gradients of z_a (dL_dz) with the global gradients of W_a (dL_dW) and b_a (dL_db), please use operators/functions in PyTorch to build the computational graph. 
    * You could use A*B to compute the element-wise product of two torch tensors A and B. 
    * You could use A.sum() to compute the sum of all elements in a torch tensor A. 
    * You could use A+B to compute the element-wise sum of two torch tensors A and B. 
    * You could use A[i:j,k:l] to get a sub-matrix of a torch matrix A. 
    * This problem can be solved using only 5 line(s) of code. More lines are okay.'''
    def forward(self, x):
        ##############################
        ## INSERT YOUR CODE HERE (2.0 points)
        
        ##############################
        return z
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Conv2d_v1_forward
        (Mac /Linux): python3 -m pytest -v test_2.py -m Conv2d_v1_forward
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
'''------------- Class: Conv2d_v2 (2.0 points) -------
    (Example B: Convolutional Layer with 1 filter, c input/color channels on 1 image) Let's continue with one filter and one image with multiple (c) input channels. Given a convolutional filter (with weights W and bias b) and an image x with c color channels, please compute the 2D convolution with the filter on the image. Here we assume that stride=1 and padding = 0 
'''
''' ---- Class Properties ----
    * s: the size of the convolutional filter, an integer scalar.
    * c: the number of color channels in each input image, an integer scalar.
    * W: the weights of 1 convolutional filter with c color/input channels,  a float torch tensor of shape (c, s, s). Here W_b[i] represents the weights of the filter on the i-th input/color channel.
    * b: the bias of 1 convolutional filter with c color/input channels,  a float torch scalar.
    '''
class Conv2d_v2(nn.Module):
    #------------- Method: __init__  ------
    ''' Goal: create and initialize the module    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * s: the size of the convolutional filter, an integer scalar
    * c: the number of color channels in each input image, an integer scalar
    '''
    def __init__(self, s, c):
        super(Conv2d_v2, self).__init__()
        self.s = s
        self.c = c
        self.W = nn.Parameter(th.randn(c,s,s)/c/s/s) # initialize the parameter Weights W randomly
        self.b = nn.Parameter(th.zeros(1)) # initialize the parameter bias b as zero
        
        
    #----------------------------------------------------------
    
    #------------- Method: forward  ------
    ''' Goal: Given a convolutional filter (with weights W and bias b) and an image x with c color channels, please compute the 2D convolution with the filter on the image. Here we assume that stride=1 and padding = 0    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: one input image with c color/input channels with hight h and width w of pixels, a float torch tensor of shape (c, h, w). Here x[i] represents i-th color/input channel of the color image
    ---- Outputs: --------
    * z: the linear logits of the convolutional layer with 1 filter and c channels on one image,  a float torch matrix of shape (h-s+1, w-s+1)
    ---- Hints: --------
    * You could use A[:, i:j,k:l] to get all the indices in the first dimension of a 3D torch tensor A, while only using sub-sets of the indices in the 2nd and 3rd dimension of the tensor A. 
    * This problem can be solved using only 6 line(s) of code. More lines are okay.'''
    def forward(self, x):
        ##############################
        ## INSERT YOUR CODE HERE (2.0 points)
        
        ##############################
        return z
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Conv2d_v2_forward
        (Mac /Linux): python3 -m pytest -v test_2.py -m Conv2d_v2_forward
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
'''------------- Class: Conv2d_v3 (2.0 points) -------
    (Example C: Convolutional Layer with c_out filters, c_in input/color channels on 1 image) Let's continue with multiple (c_out) filters and one image with multiple (c_in) input channels. Given c_out convolutional filters (with weights W and biases b) and an image x with c_in color channels, please compute the 2D convolution with the filter on the image. Here we assume that stride=1 and padding = 0 
'''
''' ---- Class Properties ----
    * s: the size of the convolutional filter, an integer scalar.
    * c_in: the number of color channels in each input image, an integer scalar.
    * c_out: the number of convolutional filters, an integer scalar.
    * W: the weights of c_out convolutional filters, where each filter has c_in color/input channels,  a float torch tensor of shape (c_out, c_in, s, s). Here W[i] represents the weights of i-th convolutional filter.
    * b: the biases of c_out convolutional filters,  a float torch vector of length c_out. Here b[i] represents the bias of the i-th convolutional filter.
    '''
class Conv2d_v3(nn.Module):
    #------------- Method: __init__  ------
    ''' Goal: create and initialize the module    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * s: the size of the convolutional filter, an integer scalar
    * c_in: the number of color channels in each input image, an integer scalar
    * c_out: the number of convolutional filters, an integer scalar
    '''
    def __init__(self, s, c_in, c_out):
        super(Conv2d_v3, self).__init__()
        self.s = s
        self.c_in = c_in
        self.c_out = c_out
        self.W = nn.Parameter(th.randn(c_out,c_in,s,s)/c_in/s/s) # initialize the parameter Weights W randomly
        self.b = nn.Parameter(th.zeros(c_out)) # initialize the parameter bias b as zeros
        
        
    #----------------------------------------------------------
    
    #------------- Method: forward  ------
    ''' Goal: Given c_out convolutional filters (with weights W and biases b) and an image x with c_in color channels, please compute the 2D convolution with the filter on the image. Here we assume that stride=1 and padding = 0    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: one input image with c color/input channels with hight h and width w of pixels, a float torch tensor of shape (c, h, w). Here x[i] represents i-th color/input channel of the color image
    ---- Outputs: --------
    * z: the linear logits of the c_out convolutional filters on one image, a float torch tensor of shape (c_out, h-s+1, w-s+1). Here z[i] represents the linear logits the i-th convolutional filter
    ---- Hints: --------
    * This problem can be solved using only 7 line(s) of code. More lines are okay.'''
    def forward(self, x):
        ##############################
        ## INSERT YOUR CODE HERE (2.0 points)
        
        ##############################
        return z
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Conv2d_v3_forward
        (Mac /Linux): python3 -m pytest -v test_2.py -m Conv2d_v3_forward
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
'''------------- Class: Conv2d (2.0 points) -------
    (Convolutional Layer) (Convolutional layer with c_out filters and c_in input/color channels on batch_size images in a mini-batch) Let's continue with a convolutional layer of the CNN. Here we have multiple (c_out) filters and multiple (batch_size) images in a mini-batch, where each image has multiple (c_in) color channels. Given c_out convolutional filters (with weights W and biases b) and a mini-batch of images (x), please compute the 2D convolution on the mini-batch of images. Here we assume that stride=1 and padding = 0 
'''
''' ---- Class Properties ----
    * s: the size of the convolutional filter, an integer scalar.
    * c_in: the number of color channels in each input image, an integer scalar.
    * c_out: the number of convolutional filters, an integer scalar.
    * W: the weights of c_out convolutional filters, where each filter has c_in color/input channels,  a float torch tensor of shape (c_out, c_in, s, s). Here W[i] represents the weights of i-th convolutional filter.
    * b: the biases of c_out convolutional filters,  a float torch vector of length c_out. Here b[i] represents the bias of the i-th convolutional filter.
    '''
class Conv2d(Conv2d_v3):
    #------------- Method: forward  ------
    ''' Goal: Given c_out convolutional filters (with weights W and biases b) and a mini-batch of images (x), please compute the 2D convolution on the mini-batch of images. Here we assume that stride=1 and padding = 0    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: a mini-batch of input images, a float torch tensor of shape (batch_size, c_in, h, w)
    ---- Outputs: --------
    * z: the linear logits of the convolutional layer of CNN on a mini-batch of data samples, a float torch tensor of shape (batch_size, c_out, h- s + 1, w - s + 1)
    ---- Hints: --------
    * To speed up your code, please do not use for-loops in this function. In this function, you may want to use the th.nn.functional.conv2d() function, where pytorch has optimized the running performance, so that the code is faster. We will use this module in the CNN model, so the running speed of this function is very important.. 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def forward(self, x):
        ##############################
        ## INSERT YOUR CODE HERE (2.0 points)
        
        ##############################
        return z
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Conv2d_forward
        (Mac /Linux): python3 -m pytest -v test_2.py -m Conv2d_forward
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
'''------------- Class: ReLU (2.0 points) -------
    ReLU activation. Given the linear logits (z) of a convolutional layer, please compute the ReLU activations 
'''

class ReLU(nn.Module):
    #------------- Method: forward  ------
    ''' Goal: Given the linear logits (z) of a convolutional layer, please compute the ReLU activations    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * z: the linear logits of a convolutional layer of CNN on a mini-batch of data samples, a float torch tensor of shape (batch_size, c_out,h,w)
    ---- Outputs: --------
    * a: the ReLU activations of a convolutional layer of CNN on a mini-batch of data samples, a float torch tensor of shape (batch_size, c_out, h, w)
    ---- Hints: --------
    * To speed up your code, please do not use for-loops in this function. In this function, you may want to use the th.nn.functional.relu() function, where pytorch has optimized the running performance, so that the code is faster. We will use this module in the CNN model, so the running speed of this function is very important.. 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def forward(self, z):
        ##############################
        ## INSERT YOUR CODE HERE (2.0 points)
        
        ##############################
        return a
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m ReLU_forward
        (Mac /Linux): python3 -m pytest -v test_2.py -m ReLU_forward
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
'''------------- Class: MaxPooling (2.0 points) -------
     Maxpooling: Given the activations of a convolutional layer, compute the max pooling results. Here we assume that the size of the pooling window is 2 x 2 
'''

class MaxPooling(nn.Module):
    #------------- Method: forward  ------
    ''' Goal: Given the activations of a convolutional layer on a mini-batch of samples, compute the max pooling results. Here we assume that the size of the pooling window is 2 x 2    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * a: the ReLU activations of a convolutional layer of CNN on a mini-batch of data samples, a float torch tensor of shape (n, c_in, h, w)
    ---- Outputs: --------
    * p: the pooled activations (using max pooling) of the convolutional layer of CNN on a mini-batch of data samples, a float torch tensor of shape (n, c_in, h/2, w/2)
    ---- Hints: --------
    * To speed up your code, please do not use for-loops in this function. In this function, you may want to use the th.nn.functional.max_pool2d() function, where pytorch has optimized the running performance, so that the code is faster. We will use this module in the CNN model, so the running speed of this function is very important.. 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def forward(self, a):
        ##############################
        ## INSERT YOUR CODE HERE (2.0 points)
        
        ##############################
        return p
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m MaxPooling_forward
        (Mac /Linux): python3 -m pytest -v test_2.py -m MaxPooling_forward
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
'''------------- Class: CNN (28.0 points) -------
    A convolutional neural network (CNN) with two convolution layers with max pooling and ReLU activations, and one fully-connected layer at the end to predict the label. The classification task is that given an input RGB color image, predict the binary label of the image (e.g., whether the image is the owner of the smartphone or not).  The goal of this problem is to learn the details of convolutional neural network by building CNN from scratch. The structure of the CNN is (Conv layer 1) -> ReLU -> maxpooling -> (Conv layer 2) -> ReLU -> maxpooling -> (Fully-connected layer). Here we assume the pooling window is 2x2 
'''
''' ---- Class Properties ----
    * p: the number of input features, an integer scalar.
    * c: the number of classes in the classification task, an integer scalar.
    * W: the weights of the linear model, a PyTorch float vector (requires gradients) of length p. Here w[i] is the weight of the model on the i-th feature.
    * b: the bias of the linear model, a PyTorch float scalar (requires gradients).
    '''
class CNN(nn.Module):
    #------------- Method: __init__  ------
    ''' Goal: create and initialize the module    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * h: the height of each input image, an integer scalar
    * w: the width of each input image, an integer scalar
    * c: the number of color channels in each input image, an integer scalar
    * c1: the number of filters in the first convolutional layer, an integer scalar
    * c2: the number of filters in the second convolutional layer, an integer scalar
    * s1: the size of filters (height = width = s1) in the first convolutional layer of CNN, an integer scalar
    * s2: the size of filters (height = width = s2) in the second convolutional layer of CNN, an integer scalar
    * lr: also called alpha, the learning rate of the stochastic gradient descent algorithm, a float scalar, controlling the speed of gradient descent. Note in the slides we use the notation 'alpha' to refer to this value
    * n_epoch: an integer scalar, the number of passes to iterate through all training examples during stochastic gradient descent.
    '''
    def __init__(self, h, w, c, c1=20, c2=10, s1=3, s2=3, lr=0.1, n_epoch=100):
        super(CNN, self).__init__()
        # Layer 1: Convolutional layer + maxpooling
        self.conv1 = Conv2d(s=s1,c_in=c,c_out=c1) # first convolutional layer
        # Layer 2: Convolutional layer + maxpooling
        self.conv2 = Conv2d(s=s2,c_in=c1,c_out=c2) # second convolutional layer
        self.relu = ReLU() # ReLU layer, can be used for both 1st and 2nd layer for computing activations
        self.pool = MaxPooling() # pooling layers, can be used for both 1st and 2nd layer for computing pooling results
        # Layer 3: fully-connected layer to predict the final label (binary classification)
        # Here we need to compute how many dimensions do we have in the input of the linear layer
        h1 = (h-s1+1)//2 # the height of the filtered image (after pooling) in the first layer
        w1 = (w-s1+1)//2 # the width of the filtered image in the first layer
        h2 = (h1-s2+1)//2 # the height of the filtered image (after pooling) in the second layer
        w2 = (w1-s2+1)//2 # the width of the filtered image in the second layer
        n_flat_features = h2*w2*c2
        self.W3 = nn.Parameter(th.randn(n_flat_features)/n_flat_features) # the weights in the 3rd layer
        self.b3 = nn.Parameter(th.zeros(1)) # the bias b  in the 3rd layer
        # Loss function for binary classification
        self.loss_fn = nn.BCEWithLogitsLoss() # the loss function for binary classification
        self.optimizer = th.optim.SGD(self.parameters(),lr = lr) # initialize SGD optimizer to handle the gradient descent of the parameters
        self.n_epoch = n_epoch
        
        
    #----------------------------------------------------------
    
    #------------- Method: compute_z1  ------
    ''' Goal: (Convolutional Layer 1: Linear Logits) (Convolutional layer with c1 filters and c input/color channels on n images) Let's continue with first convolutional layer of the CNN. Here we have multiple (c1) filters and multiple (n) images in a mini-batch, where each image has multiple (c) color channels. Given c1 convolutional filters (with weights W1 and biases b1) and a mini-batch of images (x), please compute the 2D convolution on the mini-batch of images. Here we assume that stride=1 and padding = 0    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: a mini-batch of input images, a float torch tensor of shape (n, c, h, w)
    ---- Outputs: --------
    * z1: the linear logits of the first convolutional layer of CNN on a mini-batch of data samples, a float torch tensor of shape (n, c1, h- s1 + 1, w - s1 + 1)
    ---- Hints: --------
    * You could use a layer in the __init__() function to compute z1. 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_z1(self, x):
        ##############################
        ## INSERT YOUR CODE HERE (2.8 points)
        
        ##############################
        return z1
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m CNN_compute_z1
        (Mac /Linux): python3 -m pytest -v test_2.py -m CNN_compute_z1
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_a1  ------
    ''' Goal: (Convolutional Layer 1:  ReLU activation) Given the linear logits (z1) of the first convolutional layer, please compute the ReLU activations    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * z1: the linear logits of the first convolutional layer of CNN on a mini-batch of data samples, a float torch tensor of shape (n, c1, h- s1 + 1, w - s1 + 1)
    ---- Outputs: --------
    * a1: the ReLU activations of the first convolutional layer of CNN on a mini-batch of data samples, a float torch tensor of shape (n, c1, h - s1 + 1, w - s1 + 1)
    ---- Hints: --------
    * You could use a layer in the __init__() function to compute a1. 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_a1(self, z1):
        ##############################
        ## INSERT YOUR CODE HERE (2.8 points)
        
        ##############################
        return a1
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m CNN_compute_a1
        (Mac /Linux): python3 -m pytest -v test_2.py -m CNN_compute_a1
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_p1  ------
    ''' Goal: (Convolutional Layer 1:  Maxpooling) Given the activations (a1) of first convolutional layer, please compute the max pooling results. Here we assume that the size of the pooling window is 2 x 2    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * a1: the ReLU activations of the first convolutional layer of CNN on a mini-batch of data samples, a float torch tensor of shape (n, c1, h - s1 + 1, w - s1 + 1)
    ---- Outputs: --------
    * p1: the pooled activations (using max pooling) of the first convolutional layer of CNN on a mini-batch of data samples, a float torch tensor of shape (n, c1, h1, w1)
    ---- Hints: --------
    * You could use a layer in the __init__() function to compute p1. 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_p1(self, a1):
        ##############################
        ## INSERT YOUR CODE HERE (2.8 points)
        
        ##############################
        return p1
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m CNN_compute_p1
        (Mac /Linux): python3 -m pytest -v test_2.py -m CNN_compute_p1
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_z2  ------
    ''' Goal: (Convolutional Layer 2: Linear Logits) In the second convolutional layer, suppose we have c2 filters, c1 input channels and a mini-batch of n images. Given c2 convolutional filters (with weights W2 and biases b2), please compute the 2D convolution on feature maps p1 of the mini-batch of images. Here we assume that stride=1 and padding = 0    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * p1: the pooled activations (using max pooling) of the first convolutional layer of CNN on a mini-batch of data samples, a float torch tensor of shape (n, c1, h1, w1)
    ---- Outputs: --------
    * z2: the linear logits of the second convolutional layer of CNN on a mini-batch of data samples, a float torch tensor of shape (n, c2, h1 - s2 + 1, w1 - s2 + 1)
    ---- Hints: --------
    * You could use a layer in the __init__() function to compute z2. 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_z2(self, p1):
        ##############################
        ## INSERT YOUR CODE HERE (1.4 points)
        
        ##############################
        return z2
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m CNN_compute_z2
        (Mac /Linux): python3 -m pytest -v test_2.py -m CNN_compute_z2
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_a2  ------
    ''' Goal: (Convolutional Layer 2:  ReLU activation) Given the linear logits (z2) of the second convolutional layer, please compute the ReLU activations    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * z2: the linear logits of the second convolutional layer of CNN on a mini-batch of data samples, a float torch tensor of shape (n, c2, h1 - s2 + 1, w1 - s2 + 1)
    ---- Outputs: --------
    * a2: the ReLU activations of the second convolutional layer of CNN on a mini-batch of data samples, a float torch tensor of shape (n, c2, h1 - s2 + 1, w1 - s2 + 1 )
    ---- Hints: --------
    * You could use a layer in the __init__() function to compute a2. 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_a2(self, z2):
        ##############################
        ## INSERT YOUR CODE HERE (1.4 points)
        
        ##############################
        return a2
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m CNN_compute_a2
        (Mac /Linux): python3 -m pytest -v test_2.py -m CNN_compute_a2
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_p2  ------
    ''' Goal: (Convolutional Layer 2:  Maxpooling) Given the activations (a2) of second convolutional layer, please compute the max pooling results. Here we assume that the size of the pooling window is 2 x 2    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * a2: the ReLU activations of the second convolutional layer of CNN on a mini-batch of data samples, a float torch tensor of shape (n, c2, h1 - s2 + 1, w1 - s2 + 1 )
    ---- Outputs: --------
    * p2: the pooled activations (using max pooling) of the second convolutional layer of CNN on a mini-batch of data samples, a float torch tensor of shape (n, c2, h2, w2)
    ---- Hints: --------
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_p2(self, a2):
        ##############################
        ## INSERT YOUR CODE HERE (1.4 points)
        
        ##############################
        return p2
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m CNN_compute_p2
        (Mac /Linux): python3 -m pytest -v test_2.py -m CNN_compute_p2
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: flatten  ------
    ''' Goal: (flatten) Given the pooling results (p2) of the second convolutional layer of shape n x c2 x h2 x w2, please flatten the pooling results into a vector, so that it can be used as the input to the fully-connected layer. The flattened features will be a 2D matrix of shape (n x n_flat_features), where n_flat_features is computed as c2 x h2 x w2    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * p2: the pooled activations (using max pooling) of the second convolutional layer of CNN on a mini-batch of data samples, a float torch tensor of shape (n, c2, h2, w2)
    ---- Outputs: --------
    * f: the input features to the fully connected layer after flattening the outputs of the second convolutional layer on a mini-batch of images, a float torch tensor of shape (n, n_flat_features )
    ---- Hints: --------
    * You could use the view() function to reshape a pytorch tensor. 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def flatten(self, p2):
        ##############################
        ## INSERT YOUR CODE HERE (2.8 points)
        
        ##############################
        return f
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m CNN_flatten
        (Mac /Linux): python3 -m pytest -v test_2.py -m CNN_flatten
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_z3  ------
    ''' Goal: (Fully Connected Layer) Given flattened features on a mini-batch of images, please compute the linear logits z3 of the fully-connected layer on the mini-batch of images. Note that the outpu is for binary classification task    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * f: the input features to the fully connected layer after flattening the outputs of the second convolutional layer on a mini-batch of images, a float torch tensor of shape (n, n_flat_features )
    ---- Outputs: --------
    * z3: the linear logit of the fully-connected layer of CNN on a mini-batch of data samples, a float torch vector of length (n)
    ---- Hints: --------
    * The weights and bias of the 3rd layer (fully connected) are stored in self.W3 and self.b3. 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_z3(self, f):
        ##############################
        ## INSERT YOUR CODE HERE (1.4 points)
        
        ##############################
        return z3
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m CNN_compute_z3
        (Mac /Linux): python3 -m pytest -v test_2.py -m CNN_compute_z3
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: forward  ------
    ''' Goal: Now we can put together all the previous functions in this class. Given a convolutional neural network and we have a mini-batch of images x. Please compute the linear logits in fully-connected layer on the mini-batch of images    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: a mini-batch of input images, a float torch tensor of shape (n, c, h, w)
    ---- Outputs: --------
    * z3: the linear logit of the fully-connected layer of CNN on a mini-batch of data samples, a float torch vector of length (n)
    ---- Hints: --------
    * It's easier to follow a certain order to compute all the values: z1, a1, p1, z2, .... 
    * This problem can be solved using only 8 line(s) of code. More lines are okay.'''
    def forward(self, x):
        ##############################
        ## INSERT YOUR CODE HERE (2.8 points)
        
        ##############################
        return z3
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m CNN_forward
        (Mac /Linux): python3 -m pytest -v test_2.py -m CNN_forward
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_L  ------
    ''' Goal: Given a convolutional neural network and suppose we have already computed the linear logits z3 in the last layer (fully-connected layer) on a mini-batch of training images. Suppose the labels of the training images are in y. Please compute the average binary cross-entropy loss on the mini-batch of training images    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * z3: the linear logit of the fully-connected layer of CNN on a mini-batch of data samples, a float torch vector of length (n)
    * y: the binary labels of the images in a mini-batch, a torch integer vector of length n. The value of each element can be 0 or 1
    ---- Outputs: --------
    * L: the average of the binary cross entropy losses on a mini-batch of training images, a torch float scalar
    ---- Hints: --------
    * Because the classification task is binary classification (e.g., predicting 'owner of the smartphone' or not) instead of multi-class classification (e.g., predicting which user in the image). So the loss function should be binary cross entropy loss instead of multi-class cross entropy loss. 
    * You could use a layer in the __init__() function to compute the loss here. 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_L(self, z3, y):
        ##############################
        ## INSERT YOUR CODE HERE (2.8 points)
        
        ##############################
        return L
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m CNN_compute_L
        (Mac /Linux): python3 -m pytest -v test_2.py -m CNN_compute_L
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: update_parameters  ------
    ''' Goal: (Gradient Descent) Suppose we are given a convolutional neural network with parameters (W1, b1, W2 and b2) and we have a mini-batch of training data samples (x,y).  Suppose we have already computed the global gradients of the average loss L w.r.t. the weights W1, W2 and biases b1 and b2 on the mini-batch of data samples. Assume that we have already created an optimizer for the parameter W1, b1, W2 and b2. Please update the weights W1, W2 and biases b1 and b2 using gradient descent. After the update, the global gradients of W1, b1, W2 and b2 should be set to all zeros or None    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    ---- Hints: --------
    * Why no input is given? The optimizer for all parameters of CNN has been already created in __init__() function, you could just use the optimizer to update the paramters, without any input here. 
    * This problem can be solved using only 2 line(s) of code. More lines are okay.'''
    def update_parameters(self):
        ##############################
        ## INSERT YOUR CODE HERE (2.8 points)
        pass 
        ##############################
        
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m CNN_update_parameters
        (Mac /Linux): python3 -m pytest -v test_2.py -m CNN_update_parameters
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: train  ------
    ''' Goal: (Training Convolutional Neural Network) Given a training dataset X (images), Y (labels) in a data loader, please train a convolutional neural network using mini-batch stochastic gradient descent: iteratively update the weights W1, W2, W3 and biases b1, b2, b3 using the gradients on each mini-batch of random data samples.  We repeat n_epoch passes over all the training samples    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * data_loader: the PyTorch loader of a dataset
    ---- Hints: --------
    * Step 1 Forward pass: compute the linear logits in the last layer z3 and the loss L. 
    * Step 2 Back propagation: compute the gradients of W1, b1, W2, b2, W3 and b3. 
    * Step 3 Gradient descent: update the parameters W1, b1, W2, b2, W3 and b3 using gradient descent. 
    * This problem can be solved using only 4 line(s) of code. More lines are okay.'''
    def train(self, data_loader):
        for _ in range(self.n_epoch): # iterate through the dataset n_epoch times
            for mini_batch in data_loader: # iterate through the dataset with one mini-batch of random training samples (x,y) at a time
                x=mini_batch[0] # the feature vectors of the data samples in a mini-batch
                y=mini_batch[1] # the labels of the samples in a mini-batch
                ##############################
                ## INSERT YOUR CODE HERE (2.8 points)
                pass 
                ##############################
        
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m CNN_train
        (Mac /Linux): python3 -m pytest -v test_2.py -m CNN_train
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    

'''-------- TEST problem2.py file: (40 points) ----------
Please type the following command in your terminal to test the correctness of all the above functions in this file:
        (Windows OS): python -m pytest -v test_2.py
        (Mac /Linux): python3 -m pytest -v test_2.py
------------------------------------------------------'''






