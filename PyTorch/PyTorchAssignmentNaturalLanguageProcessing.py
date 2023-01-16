# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 06:51:17 2022

@author: sueco
"""



#%% Text Classification

# In Notebook 4a, we build a sentiment analysis model for movie reviews.  That particular sentiment analysis was a two-class classification problem, where the two classes were wehther the review as positive or negative.  Of course, natural language comes in all sorts of different forms; sometimes we want to perform other types of classification.

# For example, the AG News dataset contains text from 127600 online news articles, from 4 different categories: World, Sports, Business and Science/Technology.  AG News is typically used for topic classification: given an unseen news article, we're interested in predicting the topic.  For this assignment, you'll be training several models on the AG News dataset.  Unlike the quick example we trained inNotebook 4A, however, we're going to *learn* the word embeddings.  Since you may be unfamilar with AG News, we're going to walk through how to load the data, and get you started.

#%%  Loading AG News with Torchtext

# The AG News dataset is one of the many included in Torchtext.  It can be found grouped together with many of the other text classification datasets.  While we can download the source text onlin, torchtext makes it retievable with a quick API call.  If you're running this notebook in your machine, you can uncomment and run this block: 
    
import torchtext

#agnews_train, agnews_test = torchtext.datasets.text_classification.DATASETS["AG_NEWS"](root="./datasets")

# This didn't work.  "At the time this notebook was created, Torchtext containsa small bug in its csv reader.  You may need to change one line in the source code (https://discuss.pytorch.org/t/one-error-about-the-utils-pys-code/53885), as suggested here to successfully load the AG News dataset. 

#Try two

# Unfortunately, torchtext assumes we have ntework connectivity.  If we don't have network access, such as notebooks running in Coursera Labs, we need to reimplement some torchtext functionality.  Skip this next block if you were able to successfully run the previous code: 


from torchtext.legacy.datasets import text_classification
import torchtext

ngrams = 1
train_csv_path = 'datasets/ag_news_csv/test.csv'
test_csv_path = 'datasets/ag_news_csv/train.csv'
vocab = torchtext.vocab.build_vocab_from_iterator(
    text_classification._csv_iterator(train_csv_path, ngrams),specials=["<unk>"])
vocab.set_default_index(vocab['<unk>'])
train_data, train_labels = text_classification._create_data_from_iterator(
        vocab, text_classification._csv_iterator(train_csv_path, ngrams, yield_cls=True), False)
test_data, test_labels = text_classification._create_data_from_iterator(
        vocab, text_classification._csv_iterator(test_csv_path, ngrams, yield_cls=True), False)
if len(train_labels ^ test_labels) > 0:
    raise ValueError("Training and test labels don't match")
agnews_train = text_classification.TextClassificationDataset(vocab, train_data, train_labels)
agnews_test = text_classification.TextClassificationDataset(vocab, test_data, test_labels)


#%% Checking data format

#Lets inspect the first data example to see how the data is formatted:
    
print(agnews_train[0])

# Results
# (2, tensor([ 1357,    11,    85,   143,  1418,    34,   141,  1692,  3688,   356,  20,  4522,  8982,   192,    64,    41,   16,  5252,    16,    34,  141,    18, 23553,  2769,   399,   195,  8927,     1]))

# We can see that each example as a tuple, with the first element being the label (0,1,2,3) and the second element the text data.  Notice that the text is already 'tokenize': the words of the news article have been represented as word IDs with each number corresponding to a unique word.

#In previous notebooks, we've used DataLoader s to handle shuffling and batching.  However, if we directly try to feed these dataset objects into DataLoader, we will face an error when we try to draw our first batch. Can you figure out why? Here's a hint: 
    
print("Length of the first text example: {}".format(len(agnews_train[0][1])))
print("Length of the second text example: {}".format(len(agnews_train[1][1])))

# Because each example is a news snippit, they can vary in length.  This is natural, as humans don't stick to consistent sentence length while writing.  This creates a bit of a problem while batching, as default tensors expect the size of each dimension to be consistent. 

#How do we fix this?  The common solution is to perform padding and/or truncation, picking a maximum sequence length L.  Inputs longer than the maximum length are truncated and shorter sequences have zeros padded to the end uptil they are the length of L.  We'll focus on padding here, for simplicity.

# We can perform this padding manually, by PyTorch has this functionality implemented.  As an example, lets pad the first two sequences to the same length.

from torch.nn.utils.rnn import pad_sequence

padded_exs = pad_sequence([agnews_train[0][1], agnews_train[1][1]])

#Here we're just padding enough to make these two sequences the same length - we'd want to do this with all the data

print("First sequence padded: {}".format(padded_exs[:,0]))
print("First sequence length: {}".format(len(padded_exs[:,0])))

print("First sequence padded: {}".format(padded_exs[:,1]))
print("First sequence length: {}".format(len(padded_exs[:,1])))

#Although origionally of unequal lengths, both sequences are now the same length, with the shorter one padded with zeros.


#%% define a batch loader with padding


#We'd like the DataLoader to perform this padding operation as part of its batching process, as this will allow us to effectively combine varying-length sequences in the same input tensor.  Fortunately, Dataloader s let us override the default batching behavior with the collate_fn argument.

import numpy as np
import torch

def collator(batch):
    labels = torch.tensor([example[0] for example in batch])
    sentences = [example[1] for example in batch]
    data = pad_sequence(sentences)
    
    return[data,labels]


# Now that we have our collator padding our sequences, we can create our DataLoader s  One last thing we need to do is choose a batch size for our DatLoader.  This may be something you have to play around with. Too big and you may exceed your systems memory; too small and training may take longer (especially on CPU).  Batch size also tens to influence training dyanamics and model generalization.  Fiddle around and see what works best.

BATCH_SIZE = 8192

train_loader = torch.utils.data.DataLoader(agnews_train, batch_size = BATCH_SIZE, shuffle=True, collate_fn = collator)

test_loader = torch.utils.data.DataLoader(agnews_test, batch_size = BATCH_SIZE, shuffle=True, collate_fn = collator)


#%%%%%%%%%%%%%%%%%%%%%%%  SWEM   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%  Simple Word Embedding Model

#First, lets try out the Simple Word Embedding Model (SWEM) that we built in Notebook 4A on the AG News dataset.  Unlike before, though, instead of loading pre-trained embeddings, lets learn the embeddings from scratch.  It will be helpful to define a few more hyperparameters to start.

VOCAB_SIZE = len(agnews_train.get_vocab())
EMBED_DIM = 100
HIDDEN_DIM = 64
NUM_OUTPUTS = len(agnews_train.get_labels())
NUM_EPOCHS = 500

#%%  Now define the model

#Once again, we're going to organize our model as nn.Module.  Instead of assuming the input is already an embedding, we're going to make learning the embedding as part of our model.  We do this by using nn.Embedding to perform an embedding look-up at the beginning of our forward pass.  Once we've done the look-up, we'll have a minibatch of embedding sequences of dimension L x BATCH_SIZE x EMBED_DIM.  For SWEM, remember, we take the mean across the length dimension to get an average embedding for the sequence.

#  * Note *  Technically we should only take the mean across the embeddings at the positions corresponding to 'real' words in our input, and not for the zero paddings we artifically added.  This can be done by generating a binary mask while doing the padding to track the 'real  words in the input.  Ultimately, though, this refeinment doesn't have much impact on the results for this particular task, so we omit it here for simplicity.

import torch.nn as nn
import torch.nn.functional as F

class SWEM(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_dim, num_outputs):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_size)
        
        self.fc1 = nn.Linear(embedding_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_outputs)
        
    def forward(self, x):
        embed = self.embedding(x)
        embed_mean = torch.mean(embed, dim=0)
        
        h = self.fc1(embed_mean)
        h = F.relu(h)
        h = self.fc2(h)
        return h
    
#With the model defined we an instantiate, train and evaluate.  Because of the way we organized our model as an nn.Module and our data pipeline with a DataLoader, you should be able to use much of the same code as we have in other examples. 

#%% Running the model


# Instantiate model
model = SWEM(
    vocab_size = VOCAB_SIZE,
    embedding_size = EMBED_DIM,
    hidden_dim = HIDDEN_DIM,
    num_outputs = NUM_OUTPUTS
)



# Binary cross-entropy (BCE) Loss and Adam Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Iterate through train set minibatchs 
for epoch in range(NUM_EPOCHS):
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
        
        predictions = torch.argmax(y, dim=1)  #This needed to be argmax 
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
        
        predictions = torch.argmax(y, dim=1)
        correct += torch.sum((predictions == labels).float())
        num_test += len(inputs)
    
print('Test accuracy: {}'.format(correct/num_test))


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  RNN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



