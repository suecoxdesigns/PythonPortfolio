# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 07:13:46 2022

@author: sueco
"""

#%%  Introduction to Natural Language Processing (NLP) in PyTorch

## Word Embeddings


#Word embeddings, or word vectors, provide a way of mapping words from a vocabulary into a low-dimeionsional space, where words with similar meanings are close together.  Let's play around with a set of pre-trained word vectors, to get used to their properties.  There exist many sets of pretrained word embeddings; here we use ConceptNet Numberbatch, which provides a relatively small download in an easy-to-work-with format.

#Download word vectors

from urllib.request import urlretrieve
import os
if not os.path.isfile('datasets/mini.h5'):
    print("Downloading Conceptnet Numberbatch word embeddings...")
    conceptnet_url = 'http://conceptnet.s3.amazonaws.com/precomputed-data/2016/numberbatch/17.06/mini.h5'
    urlretrieve(conceptnet_url, 'datasets/mini.h5')
    

#%% 
#To read an h5 file, we'll need to use the h5py package.  If you followed the PyTorch installation instructions in 1A, you should have it downloaded already.  Otherwise, you can install it with

#If your environment isn't currently active, activate it in the powershell prompt:
    
# conda activate pytorch
#pip install h5py

#You may need to re-open this notebook for the installation to take effect.

#Below, we use the backage to open the mini.5 file we just downloaded.  We extract from the file a list of utf-8-encoded words, as well as their 300-d vectors.



#Load the file and pull out the word embeddings
import h5py

with h5py.File('datasets/mini.h5','r') as f:
    all_words = [word.decode('utf-8') for word in f['mat']['axis1'][:]]
    all_embeddings = f['mat']['block0_values'][:]
    
print("all_words dimensions: {}".format(len(all_words)))
print("all_embeddings dimensions: {}".format(all_embeddings.shape))

print("Random example word: {}".format(all_words[1337]))

#%%

#Now *all_words* is a list of V strings (what we call our vocabulary), and *amm_embeddings* is a Vx300 matrix.  The strins are of the form /c/language_coe/word - for example /c/en/cat and /c/es/gato.

#We are interested only in the English words.  We use Python list comprehensions to pull out hte indices of the English words, then extract just the English words (stripping the six-character /c/en prefix) and their embeddings

#Restrict our vocabulary to just the English words

english_words = [word[6:] for word in all_words if word.startswith('/c/en/')]

english_word_indices = [i for i, word in enumerate(all_words) if word.startswith('/c/en/')]

english_embeddings = all_embeddings[english_word_indices]

print("Number of English words in all_words: {0}".format(len(english_words)))

print('English_embedding dimesions: {0}'.format(english_embeddings.shape))

print(english_words[1337])

#%% magnitude of word vectors


#The magnitude of a word vector is less important that its direction; the magnitude can be though of as representing frequency of use, independent of the semantics of the word.  Here, we will be interested in sematics, so we *normalize* our vectors, diving each by its length.  The result is that all of our word vectors are length 1, and as such, lie on a unit circle.  The dot product of the two vectors is proportional to the cosine of the angle between them, and proves a measure of similarity (the bigger the cosine, the smaller the angle).

import numpy as np

norms = np.linalg.norm(english_embeddings, axis=1)

normalized_embeddings = english_embeddings.astype('float32')/norms.astype('float32').reshape([-1,1])

#We want to look up words easily , so we create a dictionary that maps us from a word to its index in the word embeddings matris

index = {word: i for i, word in enumerate(english_words)}

#%% Measure the similarity between pairs of words.

# Now we are ready to measure the similarity between pairs of words.  We use numpy to take dot products.

def similarity_score(w1,w2):
    score = np.dot(normalized_embeddings[index[w1],:], normalized_embeddings[index[w2],:])
    return score

# A word is as similar with itself as posslbe:
print('cat\tcat\t',similarity_score('cat','cat'))

#Closely related words still get high scores:
print('cat\tfeline\t', similarity_score('cat','feline'))
print('cat\tdog\t', similarity_score('cat', 'dog'))

#Unrelated words, not so much
print('cat\tmoo\t', similarity_score('cat','moo'))
print('cat\tfreeze\t', similarity_score('cat', 'freeze'))

# Antonyms are still considered related, sometimes more so than synonyms
print('antonym\topposte\t', similarity_score('antonym','opposite'))
print('antonym\tsynonym\t', similarity_score('antonym', 'synonym'))

#%% We can also find, for instance, the most similar words to a given word

def closest_to_vector (v,n):
    all_scores = np.dot(normalized_embeddings,v)
    best_words = list(map(lambda i: english_words[i], reversed(np.argsort(all_scores))))
    return best_words[:n]

def closest_to_vector_av (v,n):
    all_scores = np.dot(normalized_embeddings,v)
    sort_scores = np.argsort(all_scores)
    best_words_idx = list(map(lambda i: i,reversed(np.argsort(all_scores))))
    array_good_embeddings = normalized_embeddings[best_words_idx[:n],:]
    re_nAv = np.mean(array_good_embeddings, axis=0)
    return re_nAv

def most_similar(w,n):
    return closest_to_vector(normalized_embeddings[index[w],:],n)

print(most_similar('cat',10))
print(most_similar('dick',10))

#%% solve analogies


#We can also use *closest_to_vector* t o find words 'nearby' vectors that we create ourselves.  This allows us to solve analogies. For example, in order to solve the analogy "man:brother :: woman:?", we can computera new vector brother-man+woman: the mean of brother, kinus the meanig of man, plus the meaning of womans.  We can then ask which words are the closest, in the embedding space, to that new vector.

def solve_analogy2 (a1, b1, a2):
    diff = normalized_embeddings[index[b1], :] - normalized_embeddings[index[a1], :]
    b2 = closest_to_vector_av(diff,10)
    b3 = b2 + normalized_embeddings[index[a2], :]
    return closest_to_vector(b2, 1)

def solve_analogy(a1, b1, a2):
    b2 = normalized_embeddings[index[b1], :] - normalized_embeddings[index[a1], :] + normalized_embeddings[index[a2], :]
    return closest_to_vector(b2, 1)
    

print(solve_analogy2("man", "brother", "woman"))
print(solve_analogy("man", "husband", "woman"))
print(solve_analogy("spain", "madrid", "france"))

print(solve_analogy("wife","dog","man"))





#%% Using word embeddings in deep models

# Word embeddings are fun to play around with, but their primary use is that they allow us to think of words as existing in a continuous, Euclidean space; we can then use an exising arsenal of techniques for machine learning with continuous numerical data (like logistic regression of neural networls) to process text.  Lets take a look at an especially simple version of this.  We'll perform setiment analysis on a set of movie reviews: in particular, we will attempty to classify a movie reivew as positive or negative based on its text.

#We will use a *simple word embedding model* (SWEM) to do so.  We will represent a review as the mean of the embeddings of the words in the review.  Then we'll train a two-layer MLP (a neural network) to classify the review as positive or negative.  As you might guess, using just the mean of the embeddings discards a lot of the information in a sentence, but for the tasks like sentiment analysis, it can be suprisingly effective.

#If you don't already have it, download the movie-simple-txt file.  Each line of that file contains
#   1. the numeral o (for negative) or hte numeral 1( for positive), followed by
#   2. a tab (whitespace character) and then
#   3. the review itself

#Let's first read in the data file, parsing each line into an input representation and its corresponding label.  Again, since we're using SWEM, we're going to take the meaning of the word embeddings for all the words as our input.

import string
remove_punct = str.maketrans('','' ,string.punctuation)

# This function converts a line of our data file into a tuple (x,y), where x is a 300- dimensional representation of the words in a review, and y is its label.

def convert_line_to_example(line):
    # Pull out the first character: that's our label (0 or 1)
    y = int(line[0])
    
    # Split the line into words using Python's split() function
    words = line[2:].translate(remove_punct).lower().split()
    
    # Look up the embeddings of each word, ignoring words not
    # in our pretrained vocabulary.
    embeddings = [normalized_embeddings[index[w]] for w in words
                  if w in index]
    
    # Take the mean of the embeddings
    x = np.mean(np.vstack(embeddings), axis=0)
    return x, y

#%%
#apply the function to each line of the file
xs = []
ys = []
with open("datasets/movie-simple.txt", "r", encoding='utf-8', errors='ignore') as f:
    for l in f.readlines():
        x, y = convert_line_to_example(l)
        xs.append(x)
        ys.append(y)

# Concatenate all examples into a numpy array
xs = np.vstack(xs)
ys = np.vstack(ys)

# Examples of np.vstack: stacks arrays in sequence vertically

#a = np.array([1, 2, 3])
#b = np.array([4, 5, 6])
#np.vstack((a,b))
#array([[1, 2, 3],
#       [4, 5, 6]])

print('Shape of inputs: {}'.format(xs.shape))
print("Shape of labels: {}".format(ys.shape))

num_examples = xs.shape[0]

#  Notice that with this set-up, our input words have been converted to vectors as part of our preprocessing.  This essentially locks our word embeddings in place thoughout training, as opposed to learning the word embeddings.  Learning word embeddings, either from scratch or fine-tuned from some pre-trained initialization, is often desireable, as it specializes them for the specific task. However, because our data set is relatively small and our computation budget for this demio, we're going to forgo learning the word embeddings for this model. We'll revist this in a bit.


#%%  Split the data into testing and training sets


#Now that we've parced the data, lets save 20% of the data (rounded to a whole number) for testing, using the rest for training.  The file we loaded had all the negative reviews first, followed by al lthe positive reviews, so we need to shuffle it before we split it into the train and test splits.  We'll then convert the data into PyTorch Tensors so we can feed them into our model.

#%% shuffle the data
print("First 20 labels before shuffling: {0}".format(ys[:20, 0]))

shuffle_idx = np.random.permutation(num_examples)
xs = xs[shuffle_idx, :]
ys = ys[shuffle_idx, :]

print("First 20 labels after shuffling: {0}".format(ys[:20, 0]))

#%% sort into training and testing groups

import torch

num_train = 4*num_examples // 5

x_train = torch.tensor(xs[:num_train])
y_train = torch.tensor(ys[:num_train], dtype=torch.float32)

x_test = torch.tensor(xs[num_train:])
y_test = torch.tensor(ys[num_train:], dtype=torch.float32)

#%% make a TensorDataset and DataLoader

#We could format each batch individually as we feed it into the model, but to make it easier on ourselves, lets create a TensorDataset and DataLoader as we've used in the past for MNIST

reviews_train = torch.utils.data.TensorDataset(x_train, y_train)
reviews_test = torch.utils.data.TensorDataset(x_test, y_test)

train_loader = torch.utils.data.DataLoader(reviews_train, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(reviews_test, batch_size=100, shuffle=False)

#%%  Build model in PyTorch

import torch.nn as nn
import torch.nn.functional as F

# First we build the model, organized as a nn.Module.  We could make the number of outputs for our MLP the number of classes for this dataset (i.e. 2).  However, since we only have two output classes here ('positive' vs 'negative'), we can instead produce a single output value, calling everything greater than 0 'Positive' and everything less than 0 'negative'.  If we hass this output through a sigmoid operation, then values are mapped to [0,1] with 0.5 being the classification treshold.

class SWEM(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(300, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    
#%%    Model training

# To train the model, we instantiate the model.    We use the 'with logits' version for numerical stability.

## Training
# Instantiate model
model = SWEM()

# Binary cross-entropy (BCE) Loss and Adam Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Iterate through train set minibatchs 
for epoch in range(250):
    correct = 0
    num_examples = 0
    for inputs, labels in train_loader:
        # Zero out the gradients
        optimizer.zero_grad()
        
        # Forward pass
        y = model(inputs)
        loss = criterion(y, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        predictions = torch.round(torch.sigmoid(y))
        correct += torch.sum((predictions == labels).float())
        num_examples += len(inputs)
    
    # Print training progress
    if epoch % 25 == 0:
        acc = correct/num_examples
        print("Epoch: {0} \t Train Loss: {1} \t Train Acc: {2}".format(epoch, loss, acc))

## Testing
correct = 0
num_test = 0

with torch.no_grad():
    # Iterate through test set minibatchs 
    for inputs, labels in test_loader:
        # Forward pass
        y = model(inputs)
        
        predictions = torch.round(torch.sigmoid(y))
        correct += torch.sum((predictions == labels).float())
        num_test += len(inputs)
    
print('Test accuracy: {}'.format(correct/num_test))

#%%  Using the model

#We can now examine what our model has learned, seeing how it responds to word vectors for different words:
    
#check some words
words_to_test = ['blond','tomato','original','tired','perplexing']

for word in words_to_test:
    x = torch.tensor(normalized_embeddings[index[word]].reshape(1,300))
    print("Sentiment of the word '{0}': {1}". format(word,torch.sigmoid(model(x))))
    


#%% Learning Word Embeddings

#In the previous example, we used pre-trained word embeddings, but didn't learn them.  The word embeddings were part of the preprocessing and remained unchanged throughout training.  If we have enough data though, we might prefer to learn the word embeddings along with out model.  Pre-trained word embeddings are typically trained on large corpora with unsupervised objectives and are often non- specific.  If we have enough data, we may prefer to learn the word embeddings, either from scratch or with fin-tuning, as mkaing them specific to the task may improve performance.

#How do we learn word embeddings?  To do so, we need to make them a part of our model, rather than as a part of loading the data.  In PyTorch, the preferred way to do so is with the nn.Embedding.  Like other nn layers we've seen (i.g. nn.Linear), nn.Embedding must be instantiated first.  There are two required arguments for instantiation L the number of embeddings (i.e. the vocabulary size V) and the dimension of the word embeddings (300, in our previous example).

VOCAB_SIZE = 5000
EMBED_DIM = 300

embedding =  nn.Embedding(VOCAB_SIZE, EMBED_DIM)

#Under the hood, this createsa word embedding matrix that is 5000 x 300.

embedding.weight.size()

#Notice that this matrix is basically a 300 dimensional word embedding for each of the 5000 words, stacked on top of each other.  Looking up a word embedding in this embedding matrix is simply selecting a specific row of this matrix, corresponding to the word.

# When word embeddings are learned, nn.Embedding loo-up is often one of the first operations in a model module.  For example, if we were to learn the word embeddings for our previous SWEM model, the model might instead look like this:
    
#%% SWEM model with embeddings

class SWEMWithEmbeddings(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_dim, num_outputs):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.fc1 = nn.Linear(embedding_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_outputs)
        
    def forward(self,x):
        x = self.embedding(x)
        x = torch.mean(x,dim=0)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    
#  Here we've abstracted the size of the various layers of the model as constructor arguments, so we need to specify those hyperparameters at initialization.

#%% Model Initialization

model = SWEMWithEmbeddings(
    vocab_size = 5000,
    embedding_size = 300,
    hidden_dim = 64,
    num_outputs = 1,
)

print(model)

#SWEMWithEmbeddings(
#  (embedding): Embedding(5000, 300)
#  (fc1): Linear(in_features=300, out_features=64, bias=True)
#  (fc2): Linear(in_features=64, out_features=1, bias=True)
#)

#Note that by making embedding part of our model, the expected input to the forward() function is now the word toeksn for the input sentence, so we would have to modify our data input pipeline as well.  We'll see how this might be done in the next notebook.



#%%  Recurrent Neural Networks (RNNs)

#In the context of deep learning, sequential data is commonly modeled with Recurrent Neural Networks (RNNs). As natural language can be viewed as a sequence of words, RNNs are commonly used for NLP.  As with the fully connected and convolutional networds we've seen before, RNNs use combinations of linear and nonlinear transformations to project the input into higher level representations, and these representations can be stacked with additional layers.

# Senteces as sequences

#The key difference between sequential models and the previous models we've seen is the presence of a 'time' dimension: words in a sentence (or paragraph, document) have an ordering to them that convey meaning.

# For example, in the sentence "Recurrent neural networks are great!".  Here 'Recurrent' is the t=1 workd, which we denote w1: similarly, 'neural' is w2, and so on.  As the preceding sections have hopefully impressed upon you, it is often more advantageous to model words as embedding vectors, x1,x2,....xT, rather than one-hot vectors (whake tokens w1....wt correspond to), so our first step is often to do an embedding table look-up for each input.  Lets assume 300-dimensional word embeddings and, for simplicity, a minibatch size of 1.

mb = 1
x_dim = 300
sentence =  ["recurrent", "neural", "networks", "are", "great"]

xs = []
for word in sentence:
    xs.append(torch.tensor(normalized_embeddings[index[word]]).view(1,x_dim))
    
xs = torch.stack(xs,dim=0)
print('xs shape: {}'.format(xs.shape))

#Notice that we have formatted our inputs as (words x minibatch x embedding dimensions).  This is the preferred input ordering for PyTorch RNNs

#%%   Review: fully connected layer

# Lets say we want to process this example.  In our previous sentiment analysis example, we just took the average embedding across time, treating the input as a 'bag of words.'  For simple problems, this can work suprising well, but as you might imagine, the ordering of words in a sentence is often important and sometimes, we'd like to be able to model this temporal meaning as well.  Enter RNNs

# Before we introduce the RNN, lets first again revist the fully connected layer that we used in our logistic regression and multilayer perception examples, with a few changes in notation

#               h = f(xW + b)    h is hidden layer


# Instead of calling the result of the fully connected layer y, we're going to call it h, for hiddent state.  The variable y is usually reserved for the final layer of the neural network; since logistic regression was a single layer, using y was fine. However, if we assume there is more than one layer, it is more common to refer to the intermediate representation as h.  Note that we also use f() to denote a nonlinear activation function.  In the past, we've seen f() as a ReLU, but thid could also be a sigmoild or tanh() nonlinearlity.  

#The key thing to notice here is that we project the input x with a linear transformation (xW+b) and then apply a non-linearity to the output, giving us h. 

# During training, our goal is to learn W and b.

#%% a basic RNN

# Unlike the previous examples we've seen using fully connected layers, sequential data have multiple inputs, x1....xt, instead of a single x.  We need to adapt our models accordingly for an RNN.  While there are several variations, a common basic formulation for an RNN is the Elman RNN, which follows:
    
    # ht = tanh( x(t) Wx _ bx) + (h(t-1)Wh + bh))
    
#where tanh() is the hyperbolic tangent, a nonlinear activation function. RNNS process words one at a time in sequence (xt), producing a hiddent state, ht, at every time step.  The first half of the above equation should look familar; as will the fully connected layer, we are linearly transforming each input xt, and then applying a nonlinearlity. Notice that we apply the same linear transformation (wx,bx) at every timestep. 

# The difference is that we also apply a separate linea rtransfom (Wh,hb) to the previous hidden state h(t-1) and add it to our projected input.  This feedback is called a recurrent connection.

#These directed cycles in the RNN architecture gives them the ability to model temporal dynamics, making them particularly suited for modeling sequences (e.g. text).  We can visualize an RNN layer as folllows

#First linear transformation
# input  :xt
#Linear transformation: lt1=  x(t)Wx+bx

# Second linear transformation
#input h(t-1)
#linear transformation: lt2 = h(t-1)Wh+bh

#sum the two transformations and run through activation function
# here the activation function is tanh

#h(t) = tanh( lt1 + lt2) 

# Now h(t) and x(t+1) go into the next two linear transformations and are summed - run through tanh, ect.....

#You can think of these recurrent connections as allowing the model to consider previous hidden states of a sequence when calculating the hidden state for the current input

# Note:  We don't actually need to  seporate biases, bx and bh, as you can combine both biases into a single learnable parameter b.  However, writing ti separately help smake it clear that we're performing a linear transformation on both x(t) and h(t-1).  Speaking of combining variables, we cna also express the above operation by concatenating x(t) and  h(t-1) into a single vector zt, and then performaing a single matrix multiply z(t)Wz+b where Wz is essentialy Wx and Wh concatenated.  Indeed this is how many 'offical' RNNs muldules are implement, as the reduction in the number of separate matrix multiply operations makes it computational more efficient.  These are implementation details though

#%%  RNNs in PyTorch

# How would we implement an RNN in PyTorch?  There are quite a few ways, but lets build the Elman RNN from scratch first, using the input sequence 'recumment neural networks are great'.

# As always, import PyTorch first
import numpy as np
import torch

# In an RNN, we project both the input x(t) and the previous hidden state h(t-1) to some hidden dimeions, which we're going to choose to be 128.  To perform these operations, we're going to define some variables we're going to learn.

h_dim = 128

#for projecting the input
Wx = torch.randn(x_dim,h_dim)/np.sqrt(x_dim)
Wx.requires_grad_()
bx = torch.zeros(h_dim, requires_grad=True)

#for projecting the previous state
Wh = torch.randn(h_dim, h_dim)/np.sqrt(h_dim)
Wh.requires_grad_()
bh = torch.zeros(h_dim, requires_grad=True)

print(Wx.shape, bx.shape, Wh.shape, bh.shape)

#%% For convenience, we define a function for one time-step of the RNN.  This function takes the current input x(t) and previous hidden state, h(t-1), performs the linear transformations xWx+bx and hWh+bh and then a typerbolic tangent nonlinearity.

#%% define one time step

def RNN_step(x,h):
    h_next = torch.tanh((torch.matmul(x,Wx)+bx) + (torch.matmul(h,Wh) +bh))
    
    return h_next

#%% RNN initialization

#EAch step of our RNN is oing to require feeding in an input (i.e. the word representation) and the previous hidden state (the summary of preceding sequence).  Note that at the beginning of a sentence, we don't have a previous hidden state, so we initialize it to some value, for example all zeros:
    
#Word embeding for the first word
x1 = xs[0,:,:]

#Initialize hidden state to 0
h0 = torch.zeros([mb,h_dim])

#%% forward pass of one RN step

#To take one time step of the RNN, we call the function we wrote, passing in x1 and h0.  In this case:
    
h1 = RNN_step(x1,h0)
print("Hidden state h1 dimensions: {0}".format(h1.shape))

#%% Next time step output

#We can call the RNN_step function again to get the next time step output from our RNN

# Word embedding for our second word
x2 = xs[1,:,:]

#Forward pass of one RNN step for time step t=2
h2 = RNN_step(x2,h1)

print("Hidden state h2 dimeions: {0}".format(h2.shape))

# WE can continue unrollig the RNN as far as we need to.  For each step, we feed in the current intpu (xt) and previous hidden state (h(t-1)) to get a new output.


#%%  Using torch.nn

# In practice, much like fully connected and convolutional layers, we typically don't implement RNNs from scratch as above, instead relying on higher level APIs.  PyTorch has RNNs implemente in the torch.nn library

import torch.nn

rnn = nn.RNN(x_dim, h_dim)
print("RNN parameter shapes: {}".format([p.shape for p in rnn.parameters()]))

#Note that the RNN created by torch.nn produces parameters of the same dimensions as our from scratch example above. 

#To perform a forward pass with an RNN, we pass the entire input sequence to the forward() function, which returns the hidden states at every time step (hs) and the final hidden state (h_t)

hs, h_T = rnn(xs)

print("Hidden states shape: {}".format(hs.shape))
print("Final hidden state shape: {}".format(h_T.shape))

#What do we do with these hidden states?  It depends on the model and task.  Just like multilayer percetrons and convolutional neural networks, RNNs can be stacked in pultiple layers as well.  In this case, the outputs, h1....ht are the sequential inputs to the next layer.  If the RNN layer is the ifnal layer, ht or the mean/mans of h1...ht can be used as a summary encoding of the data sequence.  What is being predicted can aslo have an impact on what the RNN outputs are ultimately used for.

#%%  Gated RNNs

#While the RNNs we've jsut explored can successfully model simple sequential data, they tend to struggle with longer sequences, with vanishing gradients, an especially big problem.  A number of RNN variants have been proposed over the years to mitigate this issue and have been shown empirally to be more effective. In particular. Long Short-erm Memory (LSTM) and the Gated Recurrent Unit (GRU) have seen wide use recently in deep learning.  We're not going to go into detail here about what structural differences they ahve from vanilla RNNS; a fantastic summary can be found here.  https://colah.github.io/posts/2015-08-Understanding-LSTMs/   Note that 'RNN' as a name is somewhat overloaded.  It can refer to both the basic recummnet model we went over previously, or recurrent models in general (including LSTMS and GRUs)

#LSTMs and GRUs layers can be created in much the same way as basic RNN layers.  Again, rather than implementing it yourself, its recommended to use the torch.nn implementations, althrough we highly encourage that you peek at the source code so you nderstand what's going on under the hood.

lstm = nn.LSTM(x_dim,h_dim)
print('LSTM parameters: {}'.format([p.shape for p in lstm.parameters()]))

gru = nn.GRU(x_dim, h_dim)
print("GRU parameters: {}".format([p.shape for p in gru.parameters()]))

#%%  Torchtext

#Much like PyTorch as Torchvision (https://pytorch.org/docs/stable/torchvision/index.html), PyTorch also has TorchText (https://torchtext.readthedocs.io/en/latest/) for natural language processing.  As with torchvision, Torchtext has a number of popular NLP benchmark datasets, across a wide range of tasks (e.g. sentiment analysis, language modeling, machine translation).  It also has a few pre-trained word embeddings available as well, including the popular Global Vectors for Word Representation (GloVe).  If you need to load your own dataset, Torchtext has a number of useful containers that can make the data pipeline easier.

#  You'll need to install TorchText to use it: in the prompt shell
    
# If you environment isn't currently active, activate it:
# conda activate pytorch

pip install torchtext    

#%% Other materials

#Natural language processing can be several full courses on its on at most universities, both with or without neural networks.  here are some additional reads.  

#Fantastic introduction to LSTMs and GRUs
#  https://colah.github.io/posts/2015-08-Understanding-LSTMs/



#Popular blog post on the effectiveness of RNNs

# http://karpathy.github.io/2015/05/21/rnn-effectiveness/



