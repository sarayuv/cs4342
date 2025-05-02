
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
    Read_and_Agree = False
    #*******************************************
    return Read_and_Agree
#--------------------------

# ---------------------------------------------------------
'''
    Goal of Problem 1: Reinforcement Learning Problem and Q-Learning Method (50 points)
    In this problem, you will implement an AI player for the frozen lake game.  The main goal of this problem is to get familiar with reinforcement learning problem, and how to use Q learning method to find optimal policy in a game 
.
    
'''
# ---------------------------------------------------------

'''------------- Class: QLearning (50.0 points) -------
    We will build a class for Q learning agents. In this class, we will implement (1) how to use e_greedy method to pick an action in each time step; (2) update the values with a new data sample collected from the game. 
'''
''' ---- Class Properties ----
    * Q: the current Q function/table, a float matrix of shape n by c, Q[s,a] represents the Q value for (state s and action a).
    * lr: learning rate, a float scalar, between 0 and 1.
    * e: (epsilon) the probability of the player to follow the random policy in epsilon-greedy method. e is a float scalar between 0 and 1. The player has 1-e probability in each time step to follow the greedy policy on the Q values.
    * c: the number of possible actions in the game, an integer scalar.
    * n: the number of states in the game, an integer scalar.
    * gamma: the discount factor, a float scalar between 0 and 1.
    * C: the current running average of sampled values, a float scalar.
    '''
class QLearning:
    #------------- Method: __init__  ------
    ''' Goal: create and initialize the agent    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * c: the number of possible actions in the game, an integer scalar
    * n: the number of states in the game, an integer scalar
    * e: (epsilon) the probability of the player to follow the random policy in epsilon-greedy method. e is a float scalar between 0 and 1. The player has 1-e probability in each time step to follow the greedy policy on the Q values
    * lr: learning rate, a float scalar, between 0 and 1
    * gamma: the discount factor, a float scalar between 0 and 1
    '''
    def __init__(self, c, n, e=0.1, lr=0.1, gamma=0.95):
        self.c=c
        self.n=n
        self.e=e
        self.lr=lr
        self.gamma=gamma
        self.Q=np.zeros([n,c]) #initialize Q table as all zeros
        
        
    #----------------------------------------------------------
    
    #------------- Method: random_policy  ------
    ''' Goal: In a game step, choose an action using random policy. Randomly pick an action with uniform distribution: equal probabilities for all actions    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    ---- Outputs: --------
    * a: the index of the action being chosen by the player at the current step, an integer scalar between 0 and c-1
    ---- Hints: --------
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def random_policy(self):
        ##############################
        ## INSERT YOUR CODE HERE (10.0 points)
        
        ##############################
        return a
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m QLearning_random_policy
        (Mac /Linux): python3 -m pytest -v test_1.py -m QLearning_random_policy
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: greedy_policy  ------
    ''' Goal: Given the current game state s and a Q table (self.Q),  choose an action at the current step using greedy policy on the Q function. Choose the action with the largest Q value for state s    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * s: the current state of the game, an integer scalar between 0 and n-1
    ---- Outputs: --------
    * a: the index of the action being chosen by the player at the current step, an integer scalar between 0 and c-1
    ---- Hints: --------
    * You could us the argmax() function in numpy to return the index of the largest value in a vector. 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def greedy_policy(self, s):
        ##############################
        ## INSERT YOUR CODE HERE (10.0 points)
        
        ##############################
        return a
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m QLearning_greedy_policy
        (Mac /Linux): python3 -m pytest -v test_1.py -m QLearning_greedy_policy
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: choose_action  ------
    ''' Goal: Given the current Q values of a game,  choose an action at the current step using epsilon-greedy method on the Q function. We have epsilon (self.e) probability to follow the random policy (randomly pick an action with uniform distribution) and  (1-self.e) probability to follow the greedy policy on Q function (pick the action according to the largest Q value for the current game state s)    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * s: the current state of the game, an integer scalar between 0 and n-1
    ---- Outputs: --------
    * a: the index of the action being chosen by the player at the current step, an integer scalar between 0 and c-1
    ---- Hints: --------
    * You could use the random.rand() function in numpy to sample a number randomly using uniform distribution between 0 and 1. 
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def choose_action(self, s):
        ##############################
        ## INSERT YOUR CODE HERE (10.0 points)
        
        ##############################
        return a
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m QLearning_choose_action
        (Mac /Linux): python3 -m pytest -v test_1.py -m QLearning_choose_action
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: running_average  ------
    ''' Goal: Given a current running average (C) of sample values and a new sampled value (v_sample), please compute the updated running average (C_new) of the samples using running average method    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * C: the current running average of sampled values, a float scalar
    * v_sample: a new sampled value, a float scalar
    ---- Outputs: --------
    * C_new: updated running average of sample values, a float scalar
    ---- Hints: --------
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def running_average(self, C, v_sample):
        ##############################
        ## INSERT YOUR CODE HERE (5.0 points)
        
        ##############################
        return C_new
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m QLearning_running_average
        (Mac /Linux): python3 -m pytest -v test_1.py -m QLearning_running_average
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: update_memory  ------
    ''' Goal: Given the player's current Q table (value function) and a sample of one step of a game episode (i.e., the current state s, the action a chosen by the player, the next game state s_new, and reward received r), update the Q table (memory) using Bellman Optimality Equation (with discount factor gamma) using running average method    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * s: the current state of the game, an integer scalar between 0 and n-1
    * a: the index of the action being chosen by the player at the current step, an integer scalar between 0 and c-1
    * s_new: the next state of the game after the transition, an integer scalar between 0 and n-1
    * r: the reward returned after the transition, a float scalar
    * done: whether or not the game is done (True if the game has ended)
    ---- Hints: --------
    * (Step 1) compute the target Q value using Bellman Optimality Equation. 
    * (Step 2) update the element of Q table using running average method. 
    * This problem can be solved using only 2 line(s) of code. More lines are okay.'''
    def update_memory(self, s, a, s_new, r, done=False):
        ##############################
        ## INSERT YOUR CODE HERE (15.0 points)
        pass 
        ##############################
        
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m QLearning_update_memory
        (Mac /Linux): python3 -m pytest -v test_1.py -m QLearning_update_memory
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    

'''-------- TEST problem1.py file: (50 points) ----------
Please type the following command in your terminal to test the correctness of all the above functions in this file:
        (Windows OS): python -m pytest -v test_1.py
        (Mac /Linux): python3 -m pytest -v test_1.py
------------------------------------------------------'''






