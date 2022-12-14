# -*- coding: utf-8 -*-
"""PatRec_Lab2_main.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ijzpqWDjm57YrBKND3v-R1bWKmrSAB5U

## National Technical University of Athens
### School of Electrical & Computer Engineering
### Course: **Pattern Recognition**
##### *Flow S, 9th Semester, 2021-2022*

## Lab 2: Voice Recognition with Hidden Markovian Models (HMM) & Retroactive Neural Networks (RNN)

<br>

##### Full Name: Christos Tsoufis
##### A.M.: 031 17 176

## Main Lab

### Installation of Packages
"""

!apt-get install python3.6

!pip install gensim==3.8.1 matplotlib==3.1.0 nltk==3.4.4 numpy==1.16.4 pandas==0.24.2 pomegranate==0.12.0 scikit-image==0.15.0 scikit-learn==0.21.2 scipy==1.3.0 seaborn==0.9.0 torch==1.3.1 torchvision==0.4.2 tqdm==4.32.1 joblib==0.17.0
!pip install numba==0.48.0 --ignore-installed
!pip install librosa==0.7.1

import os
cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)  # Get all the files in that directory
print("Files in %r: %s" % (cwd, files))
print()

from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# %cd drive/My Drive/PatRec_Labs/Lab2

"""### Imports & Libraries"""

import os
from os import listdir
import re
import copy
import numpy as np
from numpy.core.numeric import NaN
import pandas as pd
import seaborn as sns
import math
import random
import IPython.display as ipd

import librosa
import librosa.display

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import zero_one_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Normalizer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import classification_report

from glob import glob

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim

import pickle
from pomegranate import *
from scipy.stats import multivariate_normal
import itertools

from tqdm import tqdm

import sys
import warnings
warnings.filterwarnings('ignore')

from parser import *
from hmm import init_GMM_HMM,train_GMM_HMM,predict_digit_GMM_HMM,predict_GMM_HMM
from plot_confusion_matrix import plot_confusion_matrix
from lstm import *

"""Here, we should also upload seperately the files .py that we will use.

In order to run this notebook in Google Colab, a folder should be created in Google Drive with the name "PatRec_Labs". In this folder, another folder should be created, called "Lab2". Next, in this folder, folder "recordings" should be created. Finally, from the menu on the left, choose "Files" and then "Prosartisi Drive" (its in greek-lish) and re-run the cell.
"""

!git clone https://github.com/Jakobovski/free-spoken-digit-dataset.git

!mv /content/drive/MyDrive/PatRec_Labs/Lab2/free-spoken-digit-dataset/recordings /content/drive/MyDrive/PatRec_Labs/Lab2/recordings

"""### Implementation of various functions that are used bellow

##### *These function also exist seperately in the aux .py files.*

*This cell is a copy of parser.py*
"""

import os
from glob import glob
import itertools
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from hmm import init_GMM_HMM,train_GMM_HMM,predict_digit_GMM_HMM,predict_GMM_HMM
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from plot_confusion_matrix import plot_confusion_matrix

def parse_free_digits(directory):
    # Parse relevant dataset info
    files = glob(os.path.join(directory, "*.wav"))
    fnames = [f.split("/")[1].split(".")[0].split("_") for f in files]
    ids = [f[2] for f in fnames]
    y = [int(f[0]) for f in fnames]
    speakers = [f[1] for f in fnames]
    _, Fs = librosa.core.load(files[0], sr=None)

    def read_wav(f):
        wav, _ = librosa.core.load(f, sr=None)

        return wav

    # Read all wavs
    wavs = [read_wav(f) for f in files]

    # Print dataset info
    print("Total wavs: {}. Fs = {} Hz".format(len(wavs), Fs))

    return wavs, Fs, ids, y, speakers


def extract_features(wavs, n_mfcc=6, Fs=8000):
    # Extract MFCCs for all wavs
    window = 30 * Fs // 1000
    step = window // 2
    frames = [
        librosa.feature.mfcc(
            wav, Fs, n_fft=window, hop_length=window - step, n_mfcc=n_mfcc
        ).T

        for wav in tqdm(wavs, desc="Extracting mfcc features...")
    ]
    scaler = StandardScaler()
    scaler.fit(np.concatenate(frames))
    for i in range(len(frames)):
        frames[i] = scaler.transform(frames[i])
    print("Feature extraction completed with {} mfccs per frame".format(n_mfcc))

    return frames

def split_free_digits(frames, ids, speakers, labels):
    print("Splitting in train test split using the default dataset split")
    # Split to train-test
    X_train, y_train, spk_train = [], [], []
    X_test, y_test, spk_test = [], [], []
    test_indices = ["0", "1", "2", "3", "4"]

    for idx, frame, label, spk in zip(ids, frames, labels, speakers):
        if str(idx) in test_indices:
            X_test.append(frame)
            y_test.append(label)
            spk_test.append(spk)
        else:
            X_train.append(frame)
            y_train.append(label)
            spk_train.append(spk)

    return X_train, X_test, y_train, y_test, spk_train, spk_test


def make_scale_fn(X_train):
    # Standardize on train data
    scaler = StandardScaler()
    scaler.fit(np.concatenate(X_train))
    print("Normalization will be performed using mean: {}".format(scaler.mean_))
    print("Normalization will be performed using std: {}".format(scaler.scale_))
    def scale(X):
        scaled = []

        for frames in X:
            scaled.append(scaler.transform(frames))
        return scaled
    return scale


def parser(directory, n_mfcc=6):
    wavs, Fs, ids, y, speakers = parse_free_digits(directory)
    frames = extract_features(wavs, n_mfcc=n_mfcc, Fs=Fs)
    X_train, X_test, y_train, y_test, spk_train, spk_test = split_free_digits(
        frames, ids, speakers, y)

    return X_train, X_test, y_train, y_test, spk_train, spk_test

"""*This cell is a copy of plot_confusion_matrix.py*"""

import numpy as np
import itertools
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

"""*This cell is a copy of lstm.py*"""

import os
import torch
import copy
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class FrameLevelDataset(Dataset):
    def __init__(self, feats, labels):
        """
            feats: Python list of numpy arrays that contain the sequence features.
                   Each element of this list is a numpy array of shape seq_length x feature_dimension
            labels: Python list that contains the label for each sequence (each label must be an integer)
        """
        self.lengths =  []
        for sample in feats:
            self.lengths.append(sample.shape[0])

        self.feats = self.zero_pad_and_stack(feats)
        self.labels = np.array(labels).astype('int64')

    def zero_pad_and_stack(self, x):
        """
            This function performs zero padding on a list of features and forms them into a numpy 3D array
            returns
                padded: a 3D numpy array of shape num_sequences x max_sequence_length x feature_dimension
        """
        padded = []
        max_sequence_length = max(self.lengths)
        for sample in x:
            padding = np.zeros((max_sequence_length - sample.shape[0],sample.shape[1]))
            padded.append(np.vstack((sample,padding)))
        return np.array(padded)

    def __getitem__(self, item):
        return self.feats[item], self.labels[item], self.lengths[item]

    def __len__(self):
        return len(self.feats)

class BasicLSTM(nn.Module):
    def __init__(self, input_dim, rnn_size, output_dim, num_layers, bidirectional=False, dropout = 0):
        super(BasicLSTM, self).__init__()
        self.bidirectional = bidirectional
        self.feature_size = rnn_size * 2 if self.bidirectional else rnn_size
        self.lstm = nn.LSTM(input_dim, self.feature_size, num_layers, bidirectional = bidirectional, dropout = dropout, batch_first = True)
        self.linear = nn.Linear(self.feature_size, out_features = output_dim)

    def forward(self, x, lengths):
        """
            x : 3D numpy array of dimension N x L x D
                N: batch index
                L: sequence index
                D: feature index

            lengths: N x 1
        """
        batch_outputs, _ = self.lstm(x)
        last_outputs  = self.last_timestep(batch_outputs, lengths, self.bidirectional)
        predictions   = self.linear(last_outputs).squeeze(dim=-1)
        return predictions

    def last_timestep(self, outputs, lengths, bidirectional=False):
        """
            Returns the last output of the LSTM taking into account the zero padding
        """
        if bidirectional:
            forward, backward = self.split_directions(outputs)
            last_forward = self.last_by_index(forward, lengths)
            last_backward = backward[:, 0, :]
            # Concatenate and return - maybe add more functionalities like average
            return torch.cat((last_forward, last_backward), dim=-1)

        else:
            return self.last_by_index(outputs, lengths)

    @staticmethod
    def split_directions(outputs):
        direction_size = int(outputs.size(-1) / 2)
        forward = outputs[:, :, :direction_size]
        backward = outputs[:, :, direction_size:]
        return forward, backward

    @staticmethod
    def last_by_index(outputs, lengths):
        # Index of the last output for each sequence.
        idx = (lengths - 1).view(-1, 1).expand(outputs.size(0),
                                               outputs.size(2)).unsqueeze(1)
        return outputs.gather(1, idx).squeeze()

def eval_dataset(model, dataset):
    data_loader = DataLoader(dataset, batch_size = 128)
    loss = nn.CrossEntropyLoss()
    predictions = []
    true_labels = []

    with torch.no_grad():
        total_loss = 0
        for i, data in enumerate(data_loader):
            X_batch, y_batch, lengths_batch = data
            y_pred = model(X_batch.float(), lengths_batch)
            L = loss(y_pred.float(), y_batch)
            total_loss += L
            predictions = np.concatenate((predictions,np.argmax(y_pred, 1)))
            true_labels = np.concatenate((true_labels, y_batch))

    return total_loss/(i+1), predictions, true_labels

def train(model,
          X_train, y_train,
          X_dev, y_dev,
          batch_size = 512,
          epochs = 20,
          learning_rate = 1e-5,
          momentum = 0.73,
          weight_regularization = 0,
          early_stopping = False):

    train_data      = FrameLevelDataset(X_train, y_train)
    validation_data = FrameLevelDataset(X_dev, y_dev)

    data_loader = DataLoader(train_data, batch_size = batch_size)
    loss        = nn.CrossEntropyLoss() #nn.MSELoss(reduction = 'mean')
    optimizer   = optim.SGD(model.parameters(),
                            lr = learning_rate,
                            momentum = momentum,
                            weight_decay = weight_regularization
                            )
    training_loss   = []
    validation_loss = []
    best_loss  = None
    best_model = None

    print("==========================================")

    val_loss,   _, _ = eval_dataset(model, validation_data)
    train_loss, _, _ = eval_dataset(model, train_data)

    print(f'Training   loss before training: {train_loss:.8f}')
    print(f'Validation loss before training: {val_loss:.8f}')

    validation_loss.append(val_loss)
    training_loss.append(train_loss)

    loss_increasing = 0

    for epoch in tqdm(range(epochs)):
        total_loss = 0
        for index, data in enumerate(data_loader):
            X_batch, y_batch, lengths_batch = data
            y_pred = model(X_batch.float(), lengths_batch)
            L      = loss(y_pred.float(), y_batch)
            total_loss += L.item()

            optimizer.zero_grad()
            L.backward()
            optimizer.step()

        avg_loss = total_loss / (index+1)
        val_loss,   _, _ = eval_dataset(model, validation_data)

        validation_loss.append(val_loss)
        training_loss.append(avg_loss)

        print(f'Epoch {epoch}')
        print(f'Training   Loss: {avg_loss:.8f}')
        print(f'Validation Loss: {val_loss:.8f}')

        if( best_loss is None or val_loss < best_loss ):
            best_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())

        if early_stopping:
            # Checking if model is overfitting the training set
            # or if the validation loss is climbing

            if avg_loss * 2 < val_loss: # Overfitting
                break

            if validation_loss[:-1] > validation_loss[:-2]:
                loss_increasing += 1
            else:
                loss_increasing = 0

            if loss_increasing >= 5:
                break

    print("==========================================")
    return best_model, training_loss, validation_loss

"""*This cell is a copy of hmm.py*"""

import numpy as np
from pomegranate import *

def init_GMM_HMM(data,n_states,n_mixtures):
    X = data[0]
    for i in range(1, len(data)):
        X = np.concatenate([X, data[i]], axis=0)
    X = np.array(X, dtype=np.float64)
    if (n_mixtures == 1):
        gmm = False
    else:
        gmm = True
    dists = []
    for i in range(n_states):
        if gmm:
            a = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, n_mixtures, X)
        else:
            a = MultivariateGaussianDistribution.from_samples(X)

        dists.append(a)
    trans_mat = np.zeros((n_states, n_states))
    for i in range(n_states):
        if (i == n_states - 1):
            trans_mat[i % n_states, i % n_states] = 1.0
        else:
            trans_mat[i % n_states, (i+1) % n_states]= 0.5
            trans_mat[i % n_states, i % n_states]= 0.5
    starts = [0 for i in range(n_states)]
    starts[0] = 1

    ends = [1 for i in range(n_states-1)]
    ends.append(1)

    model = HiddenMarkovModel.from_matrix(trans_mat, dists, starts, ends, state_names=['s{}'.format(i) for i in range(n_states)])
    return model

def train_GMM_HMM(model,data,max_iterations=20):
    # Fit the model
    model.fit(data, max_iterations=max_iterations)
    return model

def predict_digit_GMM_HMM(models,sample):
    probs = []
    for i in range(10):
        logp, _ = models[i].viterbi(sample)
        probs.append(logp)
    return np.argmax(probs)

def predict_GMM_HMM(models,X):
    preds = []
    for sample in X:
        pred = predict_digit_GMM_HMM(models,sample)
        preds.append(pred)
    return preds

"""### Step 09: New Data Preprocessing"""

from parser import *
X_train, X_test, y_train, y_test, spk_train, spk_test = parser('recordings')

# spliting to train 80% - val 20% but stratified
X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train,stratify=y_train,test_size=0.2,random_state=42)

"""### Step 10: Digit Recognition with GMM-HMM

This implementation is part of hmm.py and can be also be seen above.

### Step 11: Model Training
"""

# map samples to their digits
X_digits = [[] for i in range(10)]

for i in range(len(X_train)):
    X_digits[y_train[i]].append(X_train[i])

states = [1,2,3,4]
mixtures = [1,2,3,4,5]
param_combos = list(itertools.product(states, mixtures))
max_accuracy = 0
best_params = []

for combo in param_combos:
    print("Combo: ",combo)
    n_states = combo[0]
    n_mixtures = combo[1]
    models = []
    for i in range(10): # for every digit
        model = init_GMM_HMM(X_digits[i],n_states,n_mixtures)
        model = train_GMM_HMM(model,X_digits[i])
        models.append(model)
    y_pred = predict_GMM_HMM(models,X_dev)
    accuracy = accuracy_score(y_dev,y_pred)
    print(accuracy)
    if accuracy > max_accuracy:
        max_accuracy = accuracy
        best_params = combo

print("The best parameters are: n_states = %d, n_mixtures = %d" %(best_params[0],best_params[1]))

# train again for best parameters
models = []
for i in range(10):
    model = init_GMM_HMM(X_digits[i],best_params[0],best_params[1])
    model = train_GMM_HMM(model,X_digits[i])
    models.append(model)

"""### Step 12: Digit Recognition - Testing

This implementation is part of hmm.py and can be also be seen above, specifically in predict_GMM_HMM().

### Step 13: Confusion Matrix for dev and test set & Accuracy
"""

predictions_dev = predict_GMM_HMM(models, X_dev)
predictions_test = predict_GMM_HMM(models, X_test)
conf_matrix_dev = confusion_matrix(y_dev, predictions_dev)
conf_matrix_test = confusion_matrix(y_test, predictions_test)
classes = [i for i in range(10)]

print("Confusion matrix for dev set")
plot_confusion_matrix(conf_matrix_dev,classes)
print("Confusion matrix for test set")
plot_confusion_matrix(conf_matrix_test,classes)
print("Classification report for test set")
print(classification_report(y_test,predictions_test))

"""### Step 14: RNN & LSTM Training"""

input_dim     = len(X_train[0][0])
hidden_size   = 256
output_dim    = 10
num_layers    = 1
bidirectional = False
dropout       = 0.2

lstm = BasicLSTM(input_dim, hidden_size, output_dim, num_layers, bidirectional, dropout)

vectorized_labels = np.zeros((len(y_train),10))
for i, label in enumerate(y_train):
    vectorized_labels[i][label] = 1

batch_size = 128
lr         = 5e-2
epochs     = 75
momentum   = 0.73
weight_regularization = 0.5
early_stopping        = True

best_params, training_loss, validation_loss = train(
                                                    lstm,
                                                    X_train, y_train,
                                                    X_dev, y_dev,
                                                    early_stopping = early_stopping,
                                                    batch_size = batch_size,
                                                    learning_rate = lr,
                                                    epochs = epochs,
                                                    momentum = momentum)


p_file = 'params_sgd_early_not.pickle'
# p_file = 'params_sgd1e3.pickle'
# p_file = 'params_sgd7mom.pickle'

ff = open(p_file, 'wb')

training_file = 'training_early_not.pickle'
# training_file = 'training_sgd1e3.pickle'
# training_file = 'training_sgd7mom.pickle'
training = open(training_file, 'wb')

pickle.dump(best_params, ff)
pickle.dump((training_loss, validation_loss), training)
# best_params = pickle.load(ff)
# training_loss, validation_loss = pickle.load(training)

lstm.load_state_dict(best_params)

test_dataset    = FrameLevelDataset(X_test, y_test)
_, y_pred, true_labels = eval_dataset(lstm, test_dataset)

print(f'Accuracy score of LSTM: {accuracy_score(true_labels, y_pred)}')

plt.figure(figsize=(10,8))
plt.title(f'LSTM Training, SGD with learning rate {lr:.4f} and {momentum:.2f} momentum')
plt.grid()
plt.plot(list(range(len(training_loss))), training_loss)
plt.plot(list(range(len(validation_loss))), validation_loss)
plt.legend(['Training Loss', 'Validation Loss'])
plt.show()

conf_matrix = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(conf_matrix, list(range(10)), title = 'LSTM Confusion matrix')

input_dim     = len(X_train[0][0])
hidden_size   = 256
output_dim    = 10
num_layers    = 1
bidirectional = False
dropout       = 0.2

lstm = BasicLSTM(input_dim, hidden_size, output_dim, num_layers, bidirectional, dropout)

vectorized_labels = np.zeros((len(y_train),10))
for i, label in enumerate(y_train):
    vectorized_labels[i][label] = 1

batch_size = 128
lr         = 5e-2
epochs     = 75
momentum   = 0.73
weight_regularization = 0.5
early_stopping        = True

best_params, training_loss, validation_loss = train(
                                                    lstm,
                                                    X_train, y_train,
                                                    X_dev, y_dev,
                                                    early_stopping = early_stopping,
                                                    batch_size = batch_size,
                                                    learning_rate = lr,
                                                    epochs = epochs,
                                                    momentum = momentum)


# p_file = 'params_sgd_early_not.pickle'
p_file = 'params_sgd1e3.pickle'
# p_file = 'params_sgd7mom.pickle'

ff = open(p_file, 'wb')

# training_file = 'training_early_not.pickle'
training_file = 'training_sgd1e3.pickle'
# training_file = 'training_sgd7mom.pickle'
training = open(training_file, 'wb')

pickle.dump(best_params, ff)
pickle.dump((training_loss, validation_loss), training)
# best_params = pickle.load(ff)
# training_loss, validation_loss = pickle.load(training)

lstm.load_state_dict(best_params)

test_dataset    = FrameLevelDataset(X_test, y_test)
_, y_pred, true_labels = eval_dataset(lstm, test_dataset)

print(f'Accuracy score of LSTM: {accuracy_score(true_labels, y_pred)}')

plt.figure(figsize=(10,8))
plt.title(f'LSTM Training, SGD with learning rate {lr:.4f} and {momentum:.2f} momentum')
plt.grid()
plt.plot(list(range(len(training_loss))), training_loss)
plt.plot(list(range(len(validation_loss))), validation_loss)
plt.legend(['Training Loss', 'Validation Loss'])
plt.show()

conf_matrix = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(conf_matrix, list(range(10)), title = 'LSTM Confusion matrix')