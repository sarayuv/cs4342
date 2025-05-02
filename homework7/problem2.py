




#------------ No New Package --------------
# NOTE: Please don't import any new package. You should be able to solve the problems using only the package(s) imported here.
import numpy as np
import torch as th
#---------------------------------------------------------


# ---------------------------------------------------------
'''
    Goal of Problem 2: Q Learning with Neural Networks (50 points)
    In this problem, you will implement a neural network (with one fully-connected layer only) to estimate Q values in a game.
.
    
'''
# ---------------------------------------------------------

'''------------- Class: Qnet (50.0 points) -------
    We will build a class for Q learning agents using neural networks. We will use a linear layer with parameters W and b to predict the Q values of a game state 
'''
''' ---- Class Properties ----
    * lr: learning rate, a float scalar, between 0 and 1.
    * n: the number of states in the game, an integer scalar.
    * gamma: the discount factor, a float scalar between 0 and 1.
    * Q: the current Q function/table, a float matrix of shape n by c, Q[s,a] represents the Q value for (state s and action a).
    * e: (epsilon) the probability of the player to follow the random policy in epsilon-greedy method. e is a float scalar between 0 and 1. The player has 1-e probability in each time step to follow the greedy policy on the Q values.
    * c: the number of possible actions in the game, an integer scalar.
    * C: the current running average of sampled values, a float scalar.
    '''
class Qnet:
    #------------- Method: __init__  ------
    ''' Goal: create and initialize the agent    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * c: the number of possible actions in the game, an integer scalar
    * p: the number of features in a game state, an integer scalar
    * e: (epsilon) the probability of the player to follow the random policy in epsilon-greedy method. e is a float scalar between 0 and 1. The player has 1-e probability in each time step to follow the greedy policy on the Q values
    * lr: learning rate, a float scalar, between 0 and 1
    * gamma: the discount factor, a float scalar between 0 and 1
    * n: the number of game-step samples in a mini-batch, an integer scalar
    '''
    def __init__(self, c=4, p=16, e=0.1, lr=0.2, gamma=0.95, n=10):
        self.c=c
        self.p=p
        self.e=e
        self.lr=lr
        self.gamma=gamma
        self.W=th.zeros(p,c,requires_grad=True) #initialize weights as all zeros
        self.b=th.zeros(c,requires_grad=True)
        self.optimizer =  th.optim.SGD([self.W,self.b], lr=lr)
        self.loss_fn=  th.nn.MSELoss()
        self.n=n
        self.S=[] # memory to store a mini-batch of game-step samples
        self.A=[] # memory to store a mini-batch of game-step samples
        self.S_new=[] # memory to store a mini-batch of game-step samples
        self.R=[] # memory to store a mini-batch of game-step samples
        self.T=[] # memory to store a mini-batch of game-step samples
        self.i=0 # counter for mini-batch of samples
        
        
    #----------------------------------------------------------
    
    #------------- Method: compute_Q  ------
    ''' Goal: (Training: estimate Q values using Q network) Given a Q network with parameters (self.W, self.b) and we have a mini-batch of sampled game states S. Please compute the predicted Q values on the mini-batch of samples    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * S: the current states for a mini-batch of sampled game steps, a torch tensor of shape (n,p), where S[i] is the current game state in the i-th sample in the mini-batch
    ---- Outputs: --------
    * Q: the predicted Q values by the Q network on all actions for a mini-batch of game state samples, a pytorch matrix of shape (n, c). Q[i,j] represents the Q value on the j-th action for the i-th sample in the mini-batch
    ---- Hints: --------
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_Q(self, S):
        ##############################
        ## INSERT YOUR CODE HERE (5.0 points)
        Q = S @ self.W + self.b
        ##############################
        return Q
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Qnet_compute_Q
        (Mac /Linux): python3 -m pytest -v test_2.py -m Qnet_compute_Q
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_Qt  ------
    ''' Goal: (Training: compute Target Q values using  Bellman Optimality Equation) Suppose we have a mini-batch of training samples: including the new/next games states S_new and immediate rewards R in the sampled game steps in the mini-batch. Please compute the target Q values (Qt) for the mini-batch of samples using Bellman Optimality Equation. Note the gradients cannot flow through Qt, i.e., the gradients of Qt tensor should not connect with the parameters W and b    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * S_new: the new/next game states for a mini-batch of sampled games steps after state transition, a torch tensor of shape (n,p). S_new[i] is the next/new game state in the i-th sample of the mini-batch
    * R: a mini-batch of the immediate rewards returned after the transition, a float vector of length (n). R[i] is the received immediate reward of the i-th sampled game step in the mini-batch
    * T: whether or not the new/next game state is a terminal state in a mini-batch of sampled games steps, a boolean torch tensor of length n. T[i]= True if S_new[i] is a terminal state in the game (where the game ends)
    ---- Outputs: --------
    * Qt: the target Q values (estimated by Bellman Optimality Equation with the target Q network) for a mini-batch of samples, a pytorch vector of length (n). Qt[i] represents the target Q value for the i-th sample in the mini-batch
    ---- Hints: --------
    * (Step 1) compute Q values on the new/next game states. 
    * (Step 2.1) If S_new[i] is a terminal state (i.e., T[i] = True), use the immediate reward R[i] as the target reward. 
    * (Step 2.2) Otherwise, use Bellman Optimality Equation to estimate the target Q value. 
    * You could re-use compute_Q() function. 
    * To detach the gradients of a torch tensor x, you could use x.detach(), so that gradient will not flow through x. 
    * To negate the boolean values in a tensor x, you could use ~x. 
    * To convert a boolean-valued tensor x into an integer tensor, you could use x.int(). 
    * To compute the max value of a tensor, you could use th.max() function. 
    * This problem can be solved using only 2 line(s) of code. More lines are okay.'''
    def compute_Qt(self, S_new, R, T):
        ##############################
        ## INSERT YOUR CODE HERE (10.0 points)
        Q_next = self.compute_Q(S_new).detach()
        Qt = R + (~T).float() * self.gamma * th.max(Q_next, dim=1)[0]
        ##############################
        return Qt
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Qnet_compute_Qt
        (Mac /Linux): python3 -m pytest -v test_2.py -m Qnet_compute_Qt
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_L  ------
    ''' Goal: (Training: Loss function) Given estimated Q values by the Q network, the action chosen and the target Q values on a mini-batch of sampled game steps, please compute the mean-squared-error loss on the mini-batch of samples    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * Q: the predicted Q values by the Q network on all actions for a mini-batch of game state samples, a pytorch matrix of shape (n, c). Q[i,j] represents the Q value on the j-th action for the i-th sample in the mini-batch
    * A: a mini-batch of the actions chosen by the player, an integer vector of length (n)
    * Qt: the target Q values (estimated by Bellman Optimality Equation with the target Q network) for a mini-batch of samples, a pytorch vector of length (n). Qt[i] represents the target Q value for the i-th sample in the mini-batch
    ---- Outputs: --------
    * L: the average of the least square losses on a mini-batch of training images, a torch float scalar
    ---- Hints: --------
    * You could use arange(n) function in Pytorch to create an index list of [0,1,2,...,n-1]. 
    * You could use y = X[list1,list2] to select elements of matrix X into a vector. For example if list1=[1,3,5], list2=[2,4,6], then y will be a list of [  X[1,2], X[3,4], X[5,6] ]. 
    * You could use self.loss_fn (loss function) to compute the mean squared error. 
    * This problem can be solved using only 2 line(s) of code. More lines are okay.'''
    def compute_L(self, Q, A, Qt):
        ##############################
        ## INSERT YOUR CODE HERE (5.0 points)
        indices = th.arange(Q.shape[0])
        Q_selected = Q[indices, A]
        L = self.loss_fn(Q_selected, Qt)
        ##############################
        return L
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Qnet_compute_L
        (Mac /Linux): python3 -m pytest -v test_2.py -m Qnet_compute_L
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: update_parameters  ------
    ''' Goal: (Training: Gradient Descent) Suppose we are given a Q neural network with parameters (W, b) and we have a mini-batch of training samples (S,A,S_new,R).  Suppose we have already computed the global gradients of the loss L w.r.t. the weights W and biases b on the mini-batch of samples. Assume that we have already created an optimizer for the parameter W and b. Please update the weights W and biases b using gradient descent. After the update, the global gradients of W and b should be set to all zeros    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    ---- Hints: --------
    * You could use self.optimizer to perform gradient descent. 
    * This problem can be solved using only 2 line(s) of code. More lines are okay.'''
    def update_parameters(self):
        ##############################
        ## INSERT YOUR CODE HERE (5.0 points)
        self.optimizer.step()
        self.optimizer.zero_grad()
        ##############################
        
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Qnet_update_parameters
        (Mac /Linux): python3 -m pytest -v test_2.py -m Qnet_update_parameters
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: update_Q  ------
    ''' Goal: (Training: Train Q Network on a mini-batch of samples) Given a mini-batch of training samples: S (current game states), A (actions chosen), S_new (new/next game states) and R (immediate rewards), suppose the target Q values are already computed (Qt), please train the Q network parameters using gradient descent: update the weights W and biases b using the gradients on the mini-batch of data samples    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * S: the current states for a mini-batch of sampled game steps, a torch tensor of shape (n,p), where S[i] is the current game state in the i-th sample in the mini-batch
    * A: a mini-batch of the actions chosen by the player, an integer vector of length (n)
    * Qt: the target Q values (estimated by Bellman Optimality Equation with the target Q network) for a mini-batch of samples, a pytorch vector of length (n). Qt[i] represents the target Q value for the i-th sample in the mini-batch
    ---- Hints: --------
    * Step 1 Forward pass: compute estimated Q values, target Q values and the loss L. 
    * Step 2 Back propagation: compute the gradients of W and b. 
    * Step 3 Gradient descent: update the parameters W and b using gradient descent. 
    * This problem can be solved using only 4 line(s) of code. More lines are okay.'''
    def update_Q(self, S, A, Qt):
        ##############################
        ## INSERT YOUR CODE HERE (5.0 points)
        Q = self.compute_Q(S)
        L = self.compute_L(Q, A, Qt)
        L.backward()
        self.update_parameters()
        ##############################
        
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Qnet_update_Q
        (Mac /Linux): python3 -m pytest -v test_2.py -m Qnet_update_Q
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: predict_q  ------
    ''' Goal: (Sampling: using Q network for playing the game) Given the Q network with parameters W and b and we have only the current states s in the game. Please compute the estimated Q values on the current game state    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * s: the current state of the game, a torch vector of length p
    ---- Outputs: --------
    * q: the Q values estimated by the Q-network on all actions for the current step of the game, a torch vector of length c. q[i] represents the estimated Q value for the i-th action
    ---- Hints: --------
    * You could re-use the compute_Q() function above by creating a mini-batch of only one sample. 
    * To add a dimension to a torch tensor, you could use unsqueeze() function in torch tensor. 
    * To delete a dimension to a torch tensor, you could use squeeze() function in torch tensor. 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def predict_q(self, s):
        ##############################
        ## INSERT YOUR CODE HERE (5.0 points)
        q = self.compute_Q(s.unsqueeze(0)).squeeze(0)
        ##############################
        return q
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Qnet_predict_q
        (Mac /Linux): python3 -m pytest -v test_2.py -m Qnet_predict_q
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: greedy_policy  ------
    ''' Goal: (Sampling: Policy 1: greedy on Q) Given the Q values estimated by the Q network on the current game state s, choose an action using greedy policy on the Q values. Choose the action with the largest Q value for state s    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * q: the Q values estimated by the Q-network on all actions for the current step of the game, a torch vector of length c. q[i] represents the estimated Q value for the i-th action
    ---- Outputs: --------
    * a: the index of the action being chosen by the player at the current step, an integer scalar between 0 and c-1
    ---- Hints: --------
    * You could us the argmax() function in torch to return the index of the largest value in a vector. 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def greedy_policy(self, q):
        ##############################
        ## INSERT YOUR CODE HERE (5.0 points)
        a = th.argmax(q).item()
        ##############################
        return a
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Qnet_greedy_policy
        (Mac /Linux): python3 -m pytest -v test_2.py -m Qnet_greedy_policy
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: egreedy_policy  ------
    ''' Goal: (Sampling: Policy 2: epsilon-greedy on Q) Given the Q values estimated by the Q network on the current game state s, choose an action using epsilon-greedy policy on the Q values    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * q: the Q values estimated by the Q-network on all actions for the current step of the game, a torch vector of length c. q[i] represents the estimated Q value for the i-th action
    ---- Outputs: --------
    * a: the index of the action being chosen by the player at the current step, an integer scalar between 0 and c-1
    ---- Hints: --------
    * You could use the random.rand() function in numpy to sample a number randomly using uniform distribution between 0 and 1. 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def egreedy_policy(self, q):
        ##############################
        ## INSERT YOUR CODE HERE (5.0 points)
        a = np.random.choice(self.c) if np.random.rand() < self.e else th.argmax(q).item()
        ##############################
        return a
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Qnet_egreedy_policy
        (Mac /Linux): python3 -m pytest -v test_2.py -m Qnet_egreedy_policy
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: sample_action  ------
    ''' Goal: (Sampling: Sample an action) Given the current game state s, sample an action using epsilon-greedy method on the Q values estimated by the Q network. We have epsilon probability to follow the random policy (randomly pick an action with uniform distribution) and  (1-epsilon) probability to follow the greedy policy on Q values (pick the action according to the largest Q value for the current game state s)    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * s: the current state of the game, a torch vector of length p
    ---- Outputs: --------
    * a: the index of the action being chosen by the player at the current step, an integer scalar between 0 and c-1
    ---- Hints: --------
    * (Step 1) use the Q network to predict the Q values on the current game state. 
    * (Step 2) use epsilon-greedy policy on the Q values to sample an action. 
    * This problem can be solved using only 2 line(s) of code. More lines are okay.'''
    def sample_action(self, s):
        ##############################
        ## INSERT YOUR CODE HERE (5.0 points)
        q = self.predict_q(s)
        a = self.egreedy_policy(q)
        ##############################
        return a
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Qnet_sample_action
        (Mac /Linux): python3 -m pytest -v test_2.py -m Qnet_sample_action
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: choose_action  ------
    ''' Goal: (API function) for agents to play the frozen lake game    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * s: the current state of the game, a numpy vector of length p
    ---- Outputs: --------
    * a: the index of the action being chosen by the player at the current step, an integer scalar between 0 and c-1
    '''
    def choose_action(self, s):
        a = self.sample_action(th.Tensor(s))
        return a
        
    #----------------------------------------------------------
    
    #------------- Method: update_memory  ------
    ''' Goal: (API function) for agents to update memory    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * s: the current state of the game, a numpy vector of length p
    * a: the index of the action being chosen by the player at the current step, an integer scalar between 0 and c-1
    * s_new: the next state of the game after the transition, a numpy vector of length p
    * r: the reward returned after the transition, a float scalar
    * done: whether or not the game is done (True if the game has ended)
    '''
    def update_memory(self, s, a, s_new, r, done):
        # store the data into the mini-batch
                self.S.append(s)
                self.A.append(a)
                self.S_new.append(s_new)
                self.R.append(r)
                self.T.append(done)
                # update mini-batch counter
                self.i= (self.i + 1) % self.n 
                if self.i==0:
                    Qt = self.compute_Qt(th.tensor(self.S_new),
                                    th.tensor(self.R),
                                    th.tensor(self.T))
                    # update Q network
                    self.update_Q(th.tensor(self.S),
                             th.LongTensor(self.A),
                             Qt)
                    # reset mini-batch memory
                    self.S = [] 
                    self.A = [] 
                    self.S_new = [] 
                    self.R = [] 
                    self.T = [] 
        
        
    #----------------------------------------------------------
    

'''-------- TEST problem2.py file: (50 points) ----------
Please type the following command in your terminal to test the correctness of all the above functions in this file:
        (Windows OS): python -m pytest -v test_2.py
        (Mac /Linux): python3 -m pytest -v test_2.py
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




