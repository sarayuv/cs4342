




#------------ No New Package --------------
# NOTE: Please don't import any new package. You should be able to solve the problems using only the package(s) imported here.
import torch as th
from torch import nn
import torch.nn.functional as F
import math
#---------------------------------------------------------


# ---------------------------------------------------------
'''
    Goal of Problem 3: Transformer Model (40 points)
    In this problem, you will implement a few key components of the transformer model. Instead of using multi-head attention, in this problem, we focus on building single-head attention layer.
    
'''
# ---------------------------------------------------------

'''------------- Class: QK_DotProduct (8.0 points) -------
    (QK dot product) One of the most important design of the transformer model is the attention module for understanding the words in sentences. The attention model is based upon the QKV (Query, Key, Value) design: we first compute the pair-wise dot product of the Queries and Keys for all word pairs, and then sale the dot products, so that we can find the similarities/relations/connections between different word pairs, then in the next question, we will be able to use the pairwise similarities computed here to propagate the values between words. The similarities between words are computed by first compute the dot products between the query vectors and key vectors (as the raw scale) and then scaling these dot products to compute the similarity scores. So in this function, let's build a neural network layer that can compute the dot products (before scaling) between the queries and keys of all the words : Give the query vectors and key vectors of a set of words, compute the pairwise dot products between queries and keys of the word pairs using dot product. 
'''

class QK_DotProduct(nn.Module):
    #------------- Method: forward  ------
    ''' Goal: Give the query vectors and key vectors of a set of words, compute the pairwise similarity scores between word pairs using scaled dot product    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * q: the query vectors of a mini-batch of sentences, a tensor of shape (batch_size, num_words, d_k). Here d_k is the number of dimensions of an attention head.
    * k: the key vectors of a mini-batch of sentences, a tensor of shape (batch_size, num_words, d_k). Here d_k is the number of dimensions of an attention head.
    ---- Outputs: --------
    * p: the dot product between queries and keys, a tensor of shape (batch_size, num_words, num_words). p[i,j,k] is the dot product from the j-th word (query) to the k-th word (key) in the i-th sentence of the mini-batch
    ---- Hints: --------
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def forward(self, q, k):
        ##############################
        ## INSERT YOUR CODE HERE (8.0 points)
        p = q @ k.transpose(1, 2)
        ##############################
        return p
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_3.py -m QK_DotProduct_forward
        (Mac /Linux): python3 -m pytest -v test_3.py -m QK_DotProduct_forward
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
'''------------- Class: QK_ScaleDotProduct (4.0 points) -------
    (QK scale dot product) Build a neural network layer that can compute scaled dot product scores: Give the dot products between the query vectors and key vectors of a set of words, compute the pairwise similarity scores between word pairs using scaled dot product, i.e., the dot product divided by the square root of the number of dimensions (d_k) in the key vector . 
'''

class QK_ScaleDotProduct(nn.Module):
    #------------- Method: forward  ------
    ''' Goal: Give the dot products of the query vectors and key vectors between words, compute the scaled dot products    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * p: the dot product between queries and keys, a tensor of shape (batch_size, num_words, num_words). p[i,j,k] is the dot product from the j-th word (query) to the k-th word (key) in the i-th sentence of the mini-batch
    * d_k: the number of features/dimensions in each key vector
    ---- Outputs: --------
    * s: the scale dot product between queries and keys, a tensor of shape (batch_size, num_words, num_words). s[i,j,k] is the scale dot product from the j-th word (query) to the k-th word (key) in the i-th sentence of the mini-batch
    ---- Hints: --------
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def forward(self, p, d_k):
        ##############################
        ## INSERT YOUR CODE HERE (4.0 points)
        s = p /math.sqrt(d_k)
        ##############################
        return s
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_3.py -m QK_ScaleDotProduct_forward
        (Mac /Linux): python3 -m pytest -v test_3.py -m QK_ScaleDotProduct_forward
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
'''------------- Class: QK_SoftMax (4.0 points) -------
    (attention scores based upon QK scale dot product) Build a neural network layer that can compute attention scores bewtween words: Give the scaled dot products between words, compute the pairwise attention scores between word pairs using softmax function on each word 
'''

class QK_SoftMax(nn.Module):
    #------------- Method: forward  ------
    ''' Goal: Give the scale dot products, compute the pairwise attention scores between word pairs using softmax on each word.    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * s: the scale dot product between queries and keys, a tensor of shape (batch_size, num_words, num_words). s[i,j,k] is the scale dot product from the j-th word (query) to the k-th word (key) in the i-th sentence of the mini-batch
    ---- Outputs: --------
    * a: the attention scores between words, a tensor of shape (batch_size, num_words, num_words). a[i,j,k] is the attention score from the j-th word (query) to the k-th word (key) in the i-th sentence of the mini-batch
    ---- Hints: --------
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def forward(self, s):
        ##############################
        ## INSERT YOUR CODE HERE (4.0 points)
        a = F.softmax(s, dim=-1)
        ##############################
        return a
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_3.py -m QK_SoftMax_forward
        (Mac /Linux): python3 -m pytest -v test_3.py -m QK_SoftMax_forward
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
'''------------- Class: QKV_Attention (8.0 points) -------
    (QKV_attention) Given the attention scores between words computed, now let's build a neural network layer for value/messages to be passed between the words: Give the attention scores (computed with softmax) between words and the values of each word, compute attention outputs (z) for each of the words. The attention outputs of a word is the sum of all other words's values in the same sequence weighted by the attention scores. 
'''

class QKV_Attention(nn.Module):
    #------------- Method: forward  ------
    ''' Goal: Give the attention scores (a) between words and the values (v) of each word, compute the attention outputs (z) for each of the words. The attention outputs of a word is the sum of all other words's values in the same sequence weighted by the attention scores (a)    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * a: the attention scores between words, a tensor of shape (batch_size, num_words, num_words). a[i,j,k] is the attention score from the j-th word (query) to the k-th word (key) in the i-th sentence of the mini-batch
    * v: the value vectors of the words in a mini-batch of word sequences, a tensor of shape (batch_size, num_words, d_k). v[i,j,:] is the value vector of the j-th word in the i-th sequence in the mini-batch. Here d_k is the number of dimensions of the key vectors.
    ---- Outputs: --------
    * z: the output features of the words, a tensor of shape (batch_size, num_words, d_k)
    ---- Hints: --------
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def forward(self, a, v):
        ##############################
        ## INSERT YOUR CODE HERE (8.0 points)
        z = a @ v
        ##############################
        return z
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_3.py -m QKV_Attention_forward
        (Mac /Linux): python3 -m pytest -v test_3.py -m QKV_Attention_forward
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
'''------------- Class: AttentionHead (16.0 points) -------
    Build a neural network layer for one attention head: Give the word embeddings, first compute the q,k,v tensors, then use the neural network layers implemented above to compute the result features of an attention head 
'''

class AttentionHead(nn.Module):
    #------------- Method: __init__  ------
    ''' Goal: Build a neural network layer for one attention head    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * d_model: the number of dimensions in word embeddings
    * d_k: the number of dimensions of an attention head
    '''
    def __init__(self, d_model, d_k):
        super(AttentionHead,self).__init__()
        self.Wq = th.randn(d_model,d_k)
        self.Wk = th.randn(d_model,d_k)
        self.Wv = th.randn(d_model,d_k)
        self.layer1 = QK_DotProduct()
        self.layer2 = QK_ScaleDotProduct()
        self.layer3 = QK_SoftMax()
        self.layer4 = QKV_Attention()
        
        
    #----------------------------------------------------------
    
    #------------- Method: compute_qkv  ------
    ''' Goal: Give the word embeddings, compute the q,k,v tensors using the weight matrices Wq, Wk, Wv (given as the attributes of self)    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: the word embeddings of a mini-batch of sentences, a tensor of shape (batch_size, num_words, d_model). Here d_model is the number of dimensions of the word embeddings.
    ---- Outputs: --------
    * q: the query vectors of a mini-batch of sentences, a tensor of shape (batch_size, num_words, d_k). Here d_k is the number of dimensions of an attention head.
    * k: the key vectors of a mini-batch of sentences, a tensor of shape (batch_size, num_words, d_k). Here d_k is the number of dimensions of an attention head.
    * v: the value vectors of a mini-batch of sentences, a tensor of shape (batch_size, num_words, d_k). Here d_k is the number of dimensions of an attention head.
    ---- Hints: --------
    * This problem can be solved using only 3 line(s) of code. More lines are okay.'''
    def compute_qkv(self, x):
        ##############################
        ## INSERT YOUR CODE HERE (8.0 points)
        q = x @ self.Wq
        k = x @ self.Wk
        v = x @ self.Wv
        ##############################
        return q, k, v
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_3.py -m AttentionHead_compute_qkv
        (Mac /Linux): python3 -m pytest -v test_3.py -m AttentionHead_compute_qkv
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: forward  ------
    ''' Goal: Give the word embeddings, first compute the q,k,v tensors, then use the neural network layers implemented above to compute the result featurs of the attention head. Note: you may want to use the models/layers given in the attribute of self in this function. There layers are self.layer1, self.layer2, ...    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: the word embeddings of a mini-batch of sentences, a tensor of shape (batch_size, num_words, d_model). Here d_model is the number of dimensions of the word embeddings.
    ---- Outputs: --------
    * z: the output features of the words, a tensor of shape (batch_size, num_words, d_k)
    ---- Hints: --------
    * This problem can be solved using only 5 line(s) of code. More lines are okay.'''
    def forward(self, x):
        ##############################
        ## INSERT YOUR CODE HERE (8.0 points)
        q, k, v = self.compute_qkv(x)
        p = self.layer1(q, k)
        d_k = k.size(-1)
        s = self.layer2(p, d_k)
        a = self.layer3(s)
        z = self.layer4(a, v)
        ##############################
        return z
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_3.py -m AttentionHead_forward
        (Mac /Linux): python3 -m pytest -v test_3.py -m AttentionHead_forward
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    

'''-------- TEST problem3.py file: (40 points) ----------
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




