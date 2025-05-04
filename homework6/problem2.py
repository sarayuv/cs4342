




#------------ No New Package --------------
# NOTE: Please don't import any new package. You should be able to solve the problems using only the package(s) imported here.
import torch as th
import torch.nn as nn
#---------------------------------------------------------


# ---------------------------------------------------------
'''
    Goal of Problem 2: Sequence to Sequence Model (with RNN) for language translation (30 points)
    In this problem, you will build a sequence-to-sequence (seq2seq) model using recurrent neural networks (RNNs) in PyTorch, which involves creating an encoder-decoder architecture. Here we will use multi-layered RNN module to process sequential data in the encoder and decoder. We'll demonstrate how to build a seq2seq model using a simple RNN as the encoder and the decoder. To simplify the code, We'll use the nn.RNN module and nn.Linear module in PyTorch as the basic building blocks of the encoder and decoder..
    
'''
# ---------------------------------------------------------

'''------------- Class: Encoder (15.0 points) -------
    Encoder is responsible for processing the input sequence and producing its hidden representation. 
'''
''' ---- Class Properties ----
    * v: the number of words in the vocabulary, an integer scalar.
    * h: the number of neurons in the cell/hidden states (or the activations of the recurrent layer), an integer scalar.
    * num_layers: the number of RNN layers in the model, an integer scalar.
    '''
class Encoder(nn.Module):
    #------------- Method: __init__  ------
    ''' Goal: create and initialize the module    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * v: the number of words in the vocabulary, an integer scalar
    * h: the number of neurons in the hidden states or cell states (or the activations of the recurrent layer), an integer scalar
    * num_layers: the number of RNN layers in the model, an integer scalar
    '''
    def __init__(self, v, h, num_layers=3):
        super(Encoder, self).__init__()
        self.v = v # vocabulary size
        self.h = h # number of hidden features in RNN
        self.num_layers = num_layers # number layers in RNN
        self.embedding = nn.Embedding(v, h) # word embedding layer to convert word token IDs to vectors of embeddings (h dimensional)
        self.rnn = nn.RNN(h,h, num_layers = num_layers) # RNN model in the encoder to process sequential data.
        
        
    #----------------------------------------------------------
    
    #------------- Method: compute_embedding  ------
    ''' Goal: (Forward Pass) Given a sequence of word tokens (IDs), and process the input sequence using word embedding layer    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: a mini-batch of sequences, a Long torch tensor of shape (t,n). x[i,k] represents the word token ID of i-th word in the k-th sequence of the mini-batch
    ---- Outputs: --------
    * embeddings: embedding of the words in the dataset, a float torch tensor of shape (t, n, h), here h is the number of hidden features in the RNN model
    ---- Hints: --------
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_embedding(self, x):
        ##############################
        ## INSERT YOUR CODE HERE (4.5 points)
        embeddings = self.embedding(x)
        ##############################
        return embeddings
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Encoder_compute_embedding
        (Mac /Linux): python3 -m pytest -v test_2.py -m Encoder_compute_embedding
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: process_sequence  ------
    ''' Goal: (Forward Pass) Given a sequence of word embeddings, process the input sequence using multi-layered RNN model    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: embedding of the words in the dataset, a float torch tensor of shape (t, n, h), here t is the number of time steps in each sequence, n is the batch_size, h is the number of hidden features in the RNN model
    ---- Outputs: --------
    * outputs: the output of RNN model in each time step when processing the input sequences, a float torch tensor of shape (t, n, h). Here the output of RNN model is the hidden state of the last layer of RNN at each time step
    * hidden: the hidden states of all layers in the RNN model after processing the sequences (at the last time step), a float torch tensor of shape (num_layers, n, h)
    ---- Hints: --------
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def process_sequence(self, x):
        h0= th.randn(self.num_layers, x.size()[1], self.h) # create initial hidden states of the RNN model (with multiple layers)
        ##############################
        ## INSERT YOUR CODE HERE (4.5 points)
        h0 = th.randn(self.num_layers, x.size()[1], self.h)
        outputs, hidden = self.rnn(x,h0)
        ##############################
        return outputs, hidden
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Encoder_process_sequence
        (Mac /Linux): python3 -m pytest -v test_2.py -m Encoder_process_sequence
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: forward  ------
    ''' Goal: (Forward Pass) Given a mini-batch of word token sequences, process the data using encoder (RNN)    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: a mini-batch of sequences, a Long torch tensor of shape (t,n). x[i,k] represents the word token ID of i-th word in the k-th sequence of the mini-batch
    ---- Outputs: --------
    * outputs: the output of RNN model in each time step when processing the input sequences, a float torch tensor of shape (t, n, h). Here the output of RNN model is the hidden state of the last layer of RNN at each time step
    * hidden: the hidden states of all layers in the RNN model after processing the sequences (at the last time step), a float torch tensor of shape (num_layers, n, h)
    ---- Hints: --------
    * This problem can be solved using only 2 line(s) of code. More lines are okay.'''
    def forward(self, x):
        ##############################
        ## INSERT YOUR CODE HERE (6.0 points)
        embeddings = self.compute_embedding(x)
        outputs, hidden = self.process_sequence(embeddings)
        ##############################
        return outputs, hidden
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Encoder_forward
        (Mac /Linux): python3 -m pytest -v test_2.py -m Encoder_forward
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
'''------------- Class: Decoder (15.0 points) -------
    The decoder in a sequence-to-sequence (seq2seq) model is responsible for generating the output sequence based on the information encoded from the input sequence by the encoder. Its primary role is to decode the information learned by the encoder and produce the desired output sequence. 
'''
''' ---- Class Properties ----
    * v: the number of words in the vocabulary, an integer scalar.
    * h: the number of neurons in the cell/hidden states (or the activations of the recurrent layer), an integer scalar.
    * num_layers: the number of RNN layers in the model, an integer scalar.
    '''
class Decoder(nn.Module):
    #------------- Method: __init__  ------
    ''' Goal: create and initialize the module    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * v: the number of words in the vocabulary, an integer scalar
    * h: the number of neurons in the hidden states or cell states (or the activations of the recurrent layer), an integer scalar
    * num_layers: the number of RNN layers in the model, an integer scalar
    '''
    def __init__(self, v, h, num_layers=3):
        super(Decoder, self).__init__()
        self.v = v # vocabulary size
        self.h = h # number of hidden features in RNN
        self.num_layers = num_layers # number layers in RNN
        self.embedding = nn.Embedding(v, h) # word embedding layer to convert word token IDs to vector of embedding (h dimensional)
        self.rnn = nn.RNN(h,h, num_layers = num_layers) # RNN model in the encoder to process sequential data
        self.out = nn.Linear(h, v) # output layer to generate a token at each time step
        
        
    #----------------------------------------------------------
    
    #------------- Method: compute_embedding  ------
    ''' Goal: (Forward Pass) Given a word token (ID) generated from the previous time step, and process the input token of one time step (in a mini-batch) using word embedding layer    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: a mini-batch of word tokens generated from the previous time step, a Long torch tensor of shape (n,1). x[i] represents the word token ID in the i-th sample in the mini-batch
    ---- Outputs: --------
    * embeddings: embedding of the words in the mini-batch, a float torch tensor of shape (n, 1,h), here h is the number of hidden features in the RNN model
    ---- Hints: --------
    * This problem can be solved using only 1 line(s) of code. More lines are okay.'''
    def compute_embedding(self, x):
        ##############################
        ## INSERT YOUR CODE HERE (4.5 points)
        embeddings = self.embedding(x)
        ##############################
        return embeddings
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Decoder_compute_embedding
        (Mac /Linux): python3 -m pytest -v test_2.py -m Decoder_compute_embedding
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: compute_outputs  ------
    ''' Goal: (Forward Pass) Given the word embeddings generated by the previous time step, and the hidden states, process the data for one time step using RNN layers and then compute the outputs using output layer    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: the embedding of the words in the mini-batch, a float torch tensor of shape (n, 1,h), here n is the number of samples in a mini-batch, h is the number of hidden features in the RNN model
    * hidden: the hidden states of all layers in the RNN model before processing the current time step, a float torch tensor of shape (num_layers, n, h)
    ---- Outputs: --------
    * outputs: the output of decoder model in the current time step, which can be used to generate a token in the current time step, a float torch tensor of shape (n, v). Here n is the number of samples in a mini-batch, v is the vocabulary size
    * hidden: the hidden states of all layers in the RNN model after processing the current time step, a float torch tensor of shape (num_layers, n, h)
    ---- Hints: --------
    * the output layer takes the hidden states of the last RNN layer as the input and produce linear logits with the dimensions of the vocabulary size. 
    * This problem can be solved using only 2 line(s) of code. More lines are okay.'''
    def compute_outputs(self, x, hidden):
        ##############################
        ## INSERT YOUR CODE HERE (4.5 points)
        output, hidden = self.rnn(x, hidden)
        outputs = self.out(output)
        ##############################
        return outputs, hidden
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Decoder_compute_outputs
        (Mac /Linux): python3 -m pytest -v test_2.py -m Decoder_compute_outputs
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: forward  ------
    ''' Goal: (Forward Pass) Given a mini-batch of word tokens generated from the previous time step, process the data using decoder (RNN)    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: a mini-batch of word tokens generated from the previous time step, a Long torch tensor of shape (n,1). x[i] represents the word token ID in the i-th sample in the mini-batch
    * hidden: the hidden states of all layers in the RNN model after processing the sequences (at the last time step), a float torch tensor of shape (num_layers, n, h)
    ---- Outputs: --------
    * outputs: the output of decoder model in the current time step, which can be used to generate a token in the current time step, a float torch tensor of shape (n, v). Here n is the number of samples in a mini-batch, v is the vocabulary size
    * hidden: the hidden states of all layers in the RNN model after processing the current time step, a float torch tensor of shape (num_layers, n, h)
    ---- Hints: --------
    * After embedding, you may want to change the shape of the embedding tensor before giving it to the RNN model. Try this: e = e.squeeze().unsqueeze(0) . 
    * This problem can be solved using only 3 line(s) of code. More lines are okay.'''
    def forward(self, x, hidden):
        ##############################
        ## INSERT YOUR CODE HERE (6.0 points)
        embeddings = self.compute_embedding(x)
        embeddings = embeddings.permute(1,0,2)
        outputs, hidden = self.compute_outputs(embeddings, hidden)
        ##############################
        return outputs, hidden
        
    '''---------- Test This Method -----------------
    Please type the following command in your terminal to test the correctness of your code above:
        (Windows OS): python -m pytest -v test_2.py -m Decoder_forward
        (Mac /Linux): python3 -m pytest -v test_2.py -m Decoder_forward
    ---------------------------------------------------------------'''

    #----------------------------------------------------------
    
    #------------- Method: generate_tokens  ------
    ''' Goal: Given the outputs the decoder on a mini-batch of sequences, generate one output token (word ID) for each sequence in the mini-batch    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * outputs: the output of decoder model in the current time step, which can be used to generate a token in the current time step, a float torch tensor of shape (n, v). Here n is the number of samples in a mini-batch, v is the vocabulary size
    ---- Outputs: --------
    * y: the generated tokens in a mini-batch a Long torch tensor of shape (n, 1)
    '''
    def generate_tokens(self, outputs):
        y = outputs.argmax(-1)
        return y
        
    #----------------------------------------------------------
    
'''------------- Class: Seq2seq (0.0 points) -------
    A Sequence-to-Sequence (seq2seq) model is a type of neural network architecture designed to handle sequence data. It consists of an encoder and a decoder, both of which are typically recurrent neural networks (RNNs). The seq2seq model is widely used for tasks such as machine translation, text summarization, and question answering. 
'''
''' ---- Class Properties ----
    * v: the number of words in the vocabulary, an integer scalar.
    * h: the number of neurons in the cell/hidden states (or the activations of the recurrent layer), an integer scalar.
    * num_layers: the number of RNN layers in the model, an integer scalar.
    '''
class Seq2seq(nn.Module):
    #------------- Method: __init__  ------
    ''' Goal: create and initialize the module    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * v: the number of words in the vocabulary, an integer scalar
    * h: the number of neurons in the hidden states or cell states (or the activations of the recurrent layer), an integer scalar
    * num_layers: the number of RNN layers in the model, an integer scalar
    '''
    def __init__(self, v, h, num_layers=3):
        super(Seq2seq, self).__init__()
        self.encoder = Encoder(v=v, h=h,num_layers=num_layers)
        self.decoder = Decoder(v=v, h=h,num_layers=num_layers)
        
        
    #----------------------------------------------------------
    
    #------------- Method: forward  ------
    ''' Goal: The forward function in a sequence-to-sequence (seq2seq) model defines how input data flows through the model. It typically takes input data (source sequences) and produces output data (target sequences) based on the architecture of the model, which includes an encoder and a decoder.    '''
    '''---- Inputs: --------
    * self: a reference to the current instance of the class which is used to access properties of the instance.
    * x: a mini-batch of sequences, a Long torch tensor of shape (t,n). x[i,k] represents the word token ID of i-th word in the k-th sequence of the mini-batch
    * max_len: max length of generated sequence, an integer scalar
    ---- Outputs: --------
    * outputs: generated a mini-batch of token sequences
    '''
    def forward(self, x, max_len=10):
        n= x.size()[1] # batch_size
        encoder_output, encoder_hidden = self.encoder(x) # process input sequences
        decoder_hidden = encoder_hidden # start the decoder with the final hidden state of the encoder
        decoder_input = th.zeros((n,1),dtype=th.long)  # Start of sequences with starting token ID=0
        outputs = th.empty(max_len,n) # initialize the outputs
        for i in range(max_len):# generate one token at a time
            o,decoder_hidden = self.decoder(decoder_input,decoder_hidden) # decode move a time step forward
            tokens = self.decoder.generate_tokens(o) # generate a token
            decoder_input = tokens.squeeze().unsqueeze(0) # feed the generated token to the next time step
            outputs[i] = tokens
        return outputs
        
    #----------------------------------------------------------
    

'''-------- TEST problem2.py file: (30 points) ----------
Please type the following command in your terminal to test the correctness of all the above functions in this file:
        (Windows OS): python -m pytest -v test_2.py
        (Mac /Linux): python3 -m pytest -v test_2.py
------------------------------------------------------'''






