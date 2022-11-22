
# PATTERN RECOGNITION
# LAB PROJECT #1
# SUBJECT: Optical Digit Recognition

##########################################################################################
# Libraries & necessary packets
##########################################################################################

import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from collections import Counter
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold,learning_curve
from sklearn.ensemble import BaggingClassifier, VotingClassifier

##########################################################################################
# Implementation of various functions that are used in patrec_lab1.py
##########################################################################################

def show_sample(X, index):
    '''Takes a dataset (e.g. X_train) and imshows the digit at the corresponding index

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        index (int): index of digit to show
    '''
    plt.imshow(np.reshape(X[index],(16,16)))
    plt.show()

##########################################################################################

def find_digit_index(X,y):
    digits = {}
    for i in range(10):
        digits[i]=[]
        #gathering all the indices for each digit
    for i in range(len(y)):
        digits[y[i]].append(i)
    return digits

##########################################################################################

def plot_digits_samples(X, y):
    '''Takes a dataset and selects one example from each label and plots it in subplots

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
    '''
    labels = [set() for i in range(10)]

    for i, d in enumerate(y):
      labels[int(d)].add(i)

    random_digit_indices = [random.sample(s, 1)[0] for s in labels]

    random_digits = [X[i] for i in random_digit_indices]
    random_digits = list(map(lambda x: np.reshape(x,(16,16)), random_digits))

    fig, axs = plt.subplots(5,2, figsize=(8,25))
    for d,x in enumerate(random_digits):
        i = d // 2
        j = d % 2
        axs[i,j].imshow(x)
    plt.show()

##########################################################################################

def digit_mean_at_pixel(X, y, digit, pixel=(10, 10)):
    '''Calculates the mean for all instances of a specific digit at a pixel location

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select
        pixels (tuple of ints): The pixels we need to select.

    Returns:
        (float): The mean value of the digits for the specified pixels
    '''
    digits = find_digit_index(X,y)
    all_pixels = []
    for i in digits[digit]:
        temp = np.reshape(X[i],(16,16))
        pixel_val = temp[pixel[0],pixel[1]]
        all_pixels.append(pixel_val)
    mean = np.mean(all_pixels)
    return mean

##########################################################################################

def digit_variance_at_pixel(X, y, digit, pixel=(10, 10)):
    '''Calculates the variance for all instances of a specific digit at a pixel location

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select
        pixels (tuple of ints): The pixels we need to select

    Returns:
        (float): The variance value of the digits for the specified pixels
    '''
    digits = find_digit_index(X,y)
    all_pixels = []
    for i in digits[digit]:
        temp = np.reshape(X[i],(16,16))
        pixel_val = temp[pixel[0],pixel[1]]
        all_pixels.append(pixel_val)
    var = np.var(all_pixels)
    return var

##########################################################################################

def digit_mean(X, y, digit):
    '''Calculates the mean for all instances of a specific digit

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select

    Returns:
        (np.ndarray): The mean value of the digits for every pixel
    '''
    means = np.zeros((16,16))
    for i in range(16):
        for j in range(16):
            mean = digit_mean_at_pixel(X,y,digit,(i,j))
            means[i][j] = mean
    return means

##########################################################################################

def digit_variance(X, y, digit):
    '''Calculates the variance for all instances of a specific digit

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select

    Returns:
        (np.ndarray): The variance value of the digits for every pixel
    '''
    varss = np.zeros((16,16))
    for i in range(16):
        for j in range(16):
            var = digit_variance_at_pixel(X,y,digit,(i,j))
            varss[i][j] = var
    return varss

##########################################################################################

def euclidean_distance(s, m):
    '''Calculates the euclidean distance between a sample s and a mean template m

    Args:
        s (np.ndarray): Sample (nfeatures)
        m (np.ndarray): Template (nfeatures)

    Returns:
        (float) The Euclidean distance between s and m
    '''
    return np.linalg.norm(s-m, ord = 2)

##########################################################################################

def euclidean_distance_classifier(X, X_mean):
    '''Classifies based on the euclidean distance between samples in X and template vectors in X_mean

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        X_mean (np.ndarray): Digits data (n_classes x nfeatures)

    Returns:
        (np.ndarray) predictions (nsamples)
    '''
    predictions = []
    for x in X:
        x = np.reshape(x, (1,256))
        distances = np.array([euclidean_distance(x, np.reshape(X_mean[d],(1,256))) for d in range(10)])
        idx = int(np.argmin(distances))
        predictions.append(idx)
    return predictions

##########################################################################################

class EuclideanClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.X_mean_ = None


    def fit(self, X, y):
        samples_per_digit = [[] for i in range(10)]
        number_of_samples = len(y)
        for i in range(number_of_samples):
            sample = X[i]
            samples_per_digit[y[i]].append(sample)
        samples_per_digit = np.array(samples_per_digit)
        self.X_mean = [np.mean(samples_per_digit[i], axis=0) for i in range(10)]
        return self

    def predict(self, X):
        predictions = []
        for x in X:
            norms = np.zeros(10)
            for i in range(10):
                norms[i] = np.linalg.norm(x - self.X_mean[i])
            predictions.append(np.argmin(norms))
            #predict using min euclidean distance
        return predictions

    def score(self, X, y):
        predictions = self.predict(X)
        test_size = len(y)
        cor = 0
        #count where labels and predictions are same
        for i in range(test_size):
            if (y[i] == predictions[i]):
                cor += 1
        return cor/test_size

##########################################################################################

def evaluate_classifier(clf, X, y, folds=5):
        scores = []
        cv = KFold(n_splits=folds,random_state=0, shuffle=True)
        for train_index, test_index in cv.split(X):
            X_train_split, X_test_split = X[train_index], X[test_index]
            y_train_split, y_test_split = y[train_index], y[test_index]
            clf.fit(X_train_split, y_train_split)
            scores.append(clf.score(X_test_split, y_test_split))
        return np.mean(scores)

##########################################################################################

def calculate_priors(X, y): #we dont really need X -- OK Boomer
    occurencies = Counter(y)
    pre_prob = dict((x,occurencies[x]/len(y)) for x in sorted(occurencies))
    aprioris = np.zeros(10)
    for i in pre_prob.keys():
        aprioris[i] = pre_prob[i]
    return aprioris

##########################################################################################

def calc_gaussian(x, m, v):
    if (v == 0):
        v = 1e-10 # Guarantee that the following fraction is well defined
    c = 1/np.sqrt(2*np.pi*v)
    return c*np.exp(-np.power(x - m, 2)/(2 * v))

##########################################################################################

class CustomNBClassifier(BaseEstimator, ClassifierMixin):
    """Naive Bayesian Classifier"""
    def __init__(self, use_unit_variance=False):
        self.X_mean_ = None
        self.use_unit_variance = use_unit_variance

    def fit(self, X, y):
        self.p = calculate_priors(X,y)
        samples_per_digit = [[] for i in range(10)]
        for i in range(len(X)):
            sample = X[i]
            samples_per_digit[y[i]].append(sample)
        samples_per_digit = np.array(samples_per_digit)
        self.X_mean = [np.mean(samples_per_digit[i],axis=0) for i in range(10)]
        if (self.use_unit_variance == False):
            self.X_var = [np.var(samples_per_digit[i],axis=0) for i in range(10)]
        else:
            self.X_var = [[1]*256 for i in range(10)]

    def predict(self, X):
        predictions = []
        for x in X:
            test_sample = x
            log_likelihood = np.log(self.p)
            for digit in range(10):
                for i in range(len(test_sample)):
                        mean_x = self.X_mean[digit]
                        var_x = self.X_var[digit]
                        likelihood = calc_gaussian(test_sample[i],
                                                mean_x[i],
                                                var_x[i])
                        if (likelihood == 0):
                            likelihood = 10**(-10)
                        log_likelihood[digit] += np.log(likelihood)
            predictions.append(np.argmax(log_likelihood)) # Predict based on the maximum likelihood
        return predictions

    def score(self, X, y):
        predictions = self.predict(X)
        error = predictions - y
        return (1 - np.count_nonzero(error)/len(X))

##########################################################################################

class DigitsDataset(Dataset):
    def __init__(self, X, y, trans = None):
        self.data = list(zip(X, y))
        self.trans = trans

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.trans is not None:
            return self.trans(self.data[idx])
        else:
            return self.data[idx]

##########################################################################################

class LinearWActivation(nn.Module): # always inherit from nn.Module
  def __init__(self, in_features, out_features, activation='sigmoid'):
      super(LinearWActivation, self).__init__()
      self.f = nn.Linear(in_features, out_features)
      if activation == 'sigmoid':
          self.a = nn.Sigmoid()
      else:
          self.a = nn.ReLU()

  def forward(self, x): # the forward pass of info through the net
      return self.a(self.f(x))

##########################################################################################

class CustomNN(nn.Module):
    def __init__(self, layers, n_features, n_classes, activation='sigmoid'):
        super(CustomNN, self).__init__()
        layers_in = [n_features] + layers # list concatenation
        layers_out = layers + [n_classes]
        # loop through layers_in and layers_out lists
        self.f = nn.Sequential(*[
            LinearWActivation(in_feats, out_feats, activation=activation)
            for in_feats, out_feats in zip(layers_in, layers_out)
        ])
        self.clf = nn.Linear(n_classes, n_classes)

    def forward(self, x): # again the forwrad pass
        y = self.f(x)
        return self.clf(y)

##########################################################################################

class PytorchNNModel(BaseEstimator, ClassifierMixin):
    def __init__(self, *args, **kwargs):
        self.model = CustomNN([300, 80], 256, 10, 'ReLu')
        self.model.double()
        self.criterion = nn.CrossEntropyLoss()
        ETA = 2 * 1e-2
        self.optimizer = optim.SGD(self.model.parameters(), lr=ETA)

    def fit(self, X, y):
        BATCH_SZ = 32
        EPOCHS = 40
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
        self.train_data = DigitsDataset(X, y)
        # self.test_data  = DigitsDataset(X_test, y_test)
        train_loader = DataLoader(self.train_data, batch_size=BATCH_SZ)
        # test_loader = DataLoader(test_data, batch_size=BATCH_SZ)

        self.model.train()
        for epoch in range(EPOCHS):
            running_average_loss = 0
            for i, data in enumerate(train_loader):
                X_batch, y_batch = data
                self.optimizer.zero_grad()
                out = self.model(X_batch)
                loss = self.criterion(out, y_batch)
                loss.backward()
                self.optimizer.step()

                running_average_loss += loss.detach().item()
                if i % 100 == 0:
                    print("Epoch: {} \t Batch: {} \t Loss {}".format(epoch, i, float(running_average_loss) / (i + 1)))

    def predict(self, X):
        BATCH_SZ = 32
        test_data = DigitsDataset(X, np.zeros(len(X)))

        predictions = np.array([])

        test_loader = DataLoader(test_data, batch_size=BATCH_SZ)
        with torch.no_grad(): # no gradients required!! eval mode, speeds up computation
            for i, data in enumerate(test_loader):
                X_batch, _ = data # test data and labels
                out = self.model(X_batch) # get net's predictions
                val, y_pred = out.max(1) # argmax since output is a prob distribution
                predictions = np.concatenate([predictions, y_pred.numpy()])
        return predictions

    def score(self, X, y):
        BATCH_SZ = 32
        test_data = DigitsDataset(X, y)
        test_loader = DataLoader(test_data, batch_size=BATCH_SZ)

        acc = 0
        n_samples = 0
        with torch.no_grad(): # no gradients required!! eval mode, speeds up computation
            for i, data in enumerate(test_loader):
                X_batch, y_batch = data # test data and labels
                out = self.model(X_batch) # get net's predictions
                val, y_pred = out.max(1) # argmax since output is a prob distribution
                acc += (y_batch == y_pred).sum().detach().item() # get accuracy
                n_samples += BATCH_SZ

        return (acc / n_samples)

##########################################################################################

def evaluate_linear_svm_classifier(X, y, folds=5):
    clf = SVC(kernel='linear',gamma='auto')
    score = evaluate_classifier(clf,X,y,folds)
    return score

##########################################################################################

def evaluate_rbf_svm_classifier(X, y, folds=5):
    clf = SVC(kernel='rbf',gamma='auto')
    score = evaluate_classifier(clf,X,y,folds)
    return score

##########################################################################################

def evaluate_knn_classifier(X, y, folds=5):
    scores = []
    for neighbors in range(1,11):
        clf = KNeighborsClassifier(n_neighbors = neighbors)
        scores.append(evaluate_classifier(clf,X,y,folds))
    return scores

##########################################################################################

def evaluate_sklearn_nb_classifier(X, y, folds=5):
    clf = GaussianNB()
    score = evaluate_classifier(clf,X,y,folds)
    return score

##########################################################################################

def evaluate_custom_nb_classifier(X, y, folds=5):
    clf = CustomNBClassifier()
    score = evaluate_classifier(clf,X,y,folds)
    return score

##########################################################################################

def evaluate_euclidean_classifier(X, y, folds=5):
    clf = EuclideanClassifier()
    score = evaluate_classifier(clf,X,y,folds)
    return score

##########################################################################################

def evaluate_nn_classifier(X, y, folds=5):
    """ Create a pytorch nn classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    clf = PytorchNNModel()
    score = evaluate_classifier(clf, X, y, folds)

    return score

##########################################################################################

def evaluate_voting_classifier(X, y, folds=5):
    """ Create a voting ensemble classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    clf1 = GaussianNB()
    clf2 = SVC(kernel='linear',gamma='auto', probability=True)
    clf3 = KNeighborsClassifier(n_neighbors = 2)
    clf4 = EuclideanClassifier()

    vclf = VotingClassifier(estimators=[('gnb', clf1), ('lsvm', clf2), ('2nn', clf3)], voting='hard')

    # vclf = VotingClassifier(estimators=[('gnb', clf1), ('lsvm', clf2), ('euclidean', clf4)], voting='hard')

    # vclf  = VotingClassifier(estimators=[('gnb', clf1), ('lsvm', clf2), ('2nn', clf3)], voting='soft')

    score = evaluate_classifier(vclf, X, y, folds)

    return score

##########################################################################################

def evaluate_bagging_classifier(X, y, folds=5):
    """ Create a bagging ensemble classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    baggingClf = BaggingClassifier(base_estimator=SVC(kernel='linear'),n_estimators=10)

    # baggingClf = BaggingClassifier(base_estimator=EuclideanClassifier(),n_estimators=10)

    # baggingClf = BaggingClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=10)

    score = evaluate_classifier(baggingClf, X, y, folds)
    return score
