
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
    Goal of Problem 1: Nearest neighbor method for regression problems (20 points)
     In this problem, we will build our first regression model using nearest neighbor method. The nearest neighbor method for regression is a non-parametric algorithm used to predict the value of a target variable based on the value of its neighbor in the feature space. In this method, each data point in the dataset is represented as a point in a multi-dimensional space, where each dimension corresponds to a feature of the data. When making a prediction for a new data point, the algorithm identifies its nearest neighbor in the feature space. The neighbor is determined based on a distance metric, here we use Euclidean distance, between data points. The algorithm then use the value of the target variable of its nearest neighbor as the predicted value for the new data point. For simplicity, here we focus on 1 nearest neighbor instead of k nearest neighbors, as it is the most intuitive method to understand. The key idea is that similar data points in the feature space are likely to have similar target variable values..
    
'''
# ---------------------------------------------------------

'''------------- Class: Nearest_Neighbor (20.0 points) -------
    This is a class for nearest neighbor method for regression. 
'''
''' ---- Class Properties ----
    * X: the feature matrix of the training samples, a numpy matrix of shape n by p, here X[i,j] is the value of the j-th feature on the i-th training sample.
    * y: the labels of the training samples, a numpy float vector of length n, y[i] represents the label of the i-th sample in the dataset.
    '''
class Nearest_Neighbor:
    #------------- Method: train  ------
    ''' Goal: The training process of the nearest neighbor method for regression is quite simple, as this algorithm doesn't involve explicit training in the traditional sense. Instead, it memorizes the entire training dataset, storing the feature vectors and their associated target values    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * X: the feature matrix of the training samples, a numpy matrix of shape n by p, here X[i,j] is the value of the j-th feature on the i-th training sample
    * y: the labels of the training samples, a numpy float vector of length n, y[i] represents the label of the i-th sample in the dataset
    '''
    def train(self, X, y):
        self.X = X
        self.y = y
        
        
    #----------------------------------------------------------
    
    #------------- Method: find_neighbor  ------
    ''' Goal: The prediction process in the nearest neighbor method for regression involves finding the nearest neighbor of a new data point in the feature space and using its target variable value to make a prediction. In this function, given a test data point, find its nearest neighbor within the training samples (self.X) in the feature space and use its target variable value to make a prediction. Here we are using Euclidean distance as the distance measure between two data points in the feature space.    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: the feature vector of a test data point, a numpy vector of length p. p is the number of features
    ---- Outputs: --------
    * y: the predicted label of the test data point, a float scalar
    ---- Hints: --------
    * Here we assume that the model has already been trained, and the features of the training samples are stored in self.X and the labels of the training samples are stored in self.y. 
    * You could use a function in numpy (np.linalg.norm) to compute the Euclidean distance between data points. 
    * Then sort the distances and find the index of the nearest neighbor. 
    * Use the index to find the label of the training data and predict with the label. 
    * This problem can be solved using only 3 line(s) of code. More lines are okay.'''
    def find_neighbor(self, x):
        ##############################
        ## INSERT YOUR CODE HERE (10.0 points)
        distances = np.linalg.norm(self.X - x, axis=1)
        nn_index = np.argmin(distances)
        y = self.y[nn_index]
        ##############################
        return y
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m Nearest_Neighbor_find_neighbor
        (Mac /Linux): python3 -m pytest -v test_1.py -m Nearest_Neighbor_find_neighbor
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: predict  ------
    ''' Goal: Given a set of test data points, predict their labels. For each of the test data point, find its nearest neighbor in the training set and predict the label    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * Xt: the feature matrix of all testing instances, a numpy matrix of shape n by p. n is the number of test samples, p is the number of features. Xt[i] represents the i-th test sample in the dataset for label prediction
    ---- Outputs: --------
    * yt: the predicted labels of the testing instances, a numpy float vector of length n, y[i] represents the predicted label of the i-th instance in the dataset
    ---- Hints: --------
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def predict(self, Xt):
        ##############################
        ## INSERT YOUR CODE HERE (10.0 points)
        yt = np.array([self.find_neighbor(x) for x in Xt]) 
        ##############################
        return yt
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_1.py -m Nearest_Neighbor_predict
        (Mac /Linux): python3 -m pytest -v test_1.py -m Nearest_Neighbor_predict
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    

'''-------- TEST problem1.py file: (20 points) ----------
Please type the following command in your terminal to test the correctness of all the above functions in this file:
        (Windows OS): python -m pytest -v test_1.py
        (Mac /Linux): python3 -m pytest -v test_1.py
------------------------------------------------------'''






