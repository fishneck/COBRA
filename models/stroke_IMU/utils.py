import numpy as np
import pandas as pd
import pickle
from scipy.special import softmax
import matplotlib.pyplot as plt
import os
import sklearn.metrics as metrics
from scipy import stats
import seaborn as sns

def pickling(file,path):
    pickle.dump(file,open(path,'wb'))
def unpickling(path):
    file_return=pickle.load(open(path,'rb'))
    return file_return
def collate_confidence(probs,num_class=5): #,pred):
    # probs: sequence of probabilities for each micro activity
    # pred: sequence of micro activity prediction
    # output: a float (not array) between 0 and 1 indicating the overall confidence of the sequence prediction
    if probs.shape[0]==num_class:
        return probs.max(axis=0)
    else:
        return probs.max(axis=1)