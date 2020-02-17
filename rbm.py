# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 11:59:20 2019

# Botlzman machines
"""

# import the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# import the movies dataset
movies = pd.read_csv('Boltzmann_Machines/ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# import the users dataset
users = pd.read_csv('Boltzmann_Machines/ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# import the ratings dataset
ratings = pd.read_csv('Boltzmann_Machines/ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# preparing the training set
training_set = pd.read_csv('Boltzmann_Machines/ml-100k/u1.base', delimiter = '\t')

# convert the training set into an array
training_set = np.array(training_set, dtype = 'int')

# preparing the test set
test_set = pd.read_csv('Boltzmann_Machines/ml-100k/u1.test', delimiter = '\t')

# convert the test set into an array
test_set = np.array(test_set, dtype = 'int')

# get the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0]))) ## adding 'int' converts to an integer value
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1]))) ## adding 'int' converts to an integer value

# converting the data into an array with users in rows and movies in columns
def convert(data):
    new_data = [] # initialize an empty list
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:, 0] == id_users] # take all the movie id's which is in the first column per user
        id_ratings = data[:,2][data[:, 0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data

# apply the convert function to test set & trainig set
training_set = convert(training_set)        
test_set = convert(test_set)        

# convert the list into torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# converting the ratings into binary ratings; 1 - Liked; 0 - Not liked
# training set
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1 # all ratings > 2 become 1, for Liked

# test set
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1 # all ratings > 2 become 1, for Liked

# Creating the architecture of the neural network
class RBM():
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv)
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)
    def sample_h(self, x):
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    def train(self, v0, vk, ph0, phk):
        self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)
nv = len(training_set[0])
nh = 100
batch_size = 100
rbm = RBM(nv, nh)


# Training the RBM
nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    ## loop through all users in batches of batch_size
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_h(v0)
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0 < 0] = v0[v0 < 0] ## exlude all ratings that are -1 or not rated
            phk,_ = rbm.sample_h(vk)
            rbm.train(v0, vk, ph0, phk)
            train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0]))
            s += 1. ## increment our counter in decimal value
        print('epoch: ' + str(epoch) + 'loss: ' + str(train_loss/s))
        
        
        
    
        
    





