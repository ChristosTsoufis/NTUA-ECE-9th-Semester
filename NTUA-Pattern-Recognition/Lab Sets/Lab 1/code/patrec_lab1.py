
# PATTERN RECOGNITION
# LAB PROJECT #1
# SUBJECT: Optical Digit Recognition

##########################################################################################
# Libraries & necessary packets
##########################################################################################

import numpy as np
import random
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold,learning_curve
from sklearn.naive_bayes import GaussianNB

from lib import *

import sys
import warnings
warnings.filterwarnings('ignore')

##########################################################################################
# Implementation of various functions that are used below
##########################################################################################

def read_data(path, X, y):
    with  open(path, "r") as f:
        line = f.readline().split()
        line = [float(i) for i in line]
        X.append(line[1:])
        y.append(int(line[0]))
        while line:
            line = f.readline().split()
            if line:
                line = [float(i) for i in line]
                X.append(line[1:])
                y.append(int(line[0]))
    return np.array(X),np.array(y)

##########################################################################################

def plot_clf(clf,X,y,labels):
    fig, ax = plt.subplots() #Initialize plot
    y = [ int(x) for x in y ] #Get the labels in int form (form float)
    X0, X1 = X[:, 0], X[:, 1] #Get the first and second component
    x_min, x_max = X0.min() - 1, X0.max() + 1 #Get their min and max (for each component)
    y_min, y_max = X1.min() - 1, X1.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .05),
                         np.arange(y_min, y_max, .05))
    Z = np.array(clf.predict(np.c_[xx.ravel(), yy.ravel()])) #Perform prediction for all in the matrix
    Z = Z.reshape(xx.shape)
    y = list( map( int, y ) )
    for i in range(10):
        x0, x1 = [], []
        for j in range( len( y ) ):
            if y[ j ] == i:
                x0.append( X0[ j ] )
                x1.append( X1[ j ] )
        zeros = ax.scatter(
            x0, x1,
             label=i,
            s=60, alpha=0.9, edgecolors='k')
    colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired
    colorst = [colormap(i) for i in np.linspace(0, 0.9,len(ax.collections))]
    for t,j1 in enumerate(ax.collections):
        j1.set_color(colorst[t])
    ax.set_xticks(())
    ax.set_yticks(())
    ax.legend()
    plt.show()

##########################################################################################

def plot_decision_regions(clf,X,y,labels):
    fig, ax = plt.subplots() #Initialize plot
    y = [ int(x) for x in y ] #Get the labels in int form (form float)
    title = ('Decision surface of Estimator') #GEt a title
    X0, X1 = X[:, 0], X[:, 1] #Get the first and second component
    x_min, x_max = X0.min() - 1, X0.max() + 1 #Get their min and max (for each component)
    y_min, y_max = X1.min() - 1, X1.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .05),
                         np.arange(y_min, y_max, .05))
    Z = np.array(clf.predict(np.c_[xx.ravel(), yy.ravel()])) #Perform prediction for all in the matrix
    unique, counts = np.unique(Z, return_counts=True)
    Z = Z.reshape(xx.shape)
    plt.imshow(Z,cmap = "viridis")
    plt.show()

##########################################################################################

def my_learning_curve( clf, x_train, y_train, x_test, y_test ):
    #train on datasets of increasing size and plot accuracy of classifier
    X, Y, Y2 = [], [], []
    for ntests in range( 100, len( x_train ), 50 ):
        clf.fit( x_train[:ntests ], y_train[:ntests ] )
        X.append( ntests )
        Y.append( clf.score( x_train , y_train) )
        Y2.append( clf.score(  x_test, y_test) )
    plt.figure()
    plt.plot(X, Y, color='red', label="Training score")
    plt.plot(X, Y2, color='green', label="Validation score")
    plt.xlabel("Training examples")
    plt.grid(True)
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.show()

##########################################################################################

def my_error_curve( clf, x_train, y_train, x_test, y_test ):
    #train on datasets of increasing size and plot errors of classifier
    X, Y, Y2 = [], [], []
    for ntests in range( 100, len( x_train ), 100 ):
        clf.fit( x_train[:ntests ], y_train[:ntests ] )
        X.append( ntests )
        Y.append( 1 - clf.score( x_train , y_train) )
        Y2.append( 1 - clf.score( x_test, y_test) )
    plt.figure()
    plt.plot(X,Y,color='red',label="Training error")
    plt.plot(X,Y2,color='green',label="Validation error")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid(True)
    plt.legend(loc="best")
    plt.show()

##########################################################################################

def calculate_mispredictions_per_label(y_pred, y):
    mispredictions = np.zeros(10)

    for pred, actual in zip(y_pred, y):
        if pred != actual:
            mispredictions[actual] += 1

    return mispredictions

##########################################################################################

def plot_mispredictions(clfs, X, y, names, rows, cols, type="confusion_matrix"):
    fig, axs = plt.subplots(rows, cols)
    for i, clf in enumerate(clfs):
        ii = i // cols
        jj = i % cols
        clf.fit(X,y)
        y_pred = clf.predict(X)
        axs[ii, jj].set_title(names[i])
        if type == "confusion_matrix":
            matrix = confusion_matrix(y_pred, y)
            axs[ii, jj].imshow(matrix)
        else:
            errors = calculate_mispredictions_per_label(y_pred, y)
            axs[ii, jj].bar(list(range(10)), errors)
            axs[ii, jj].set_xticks(list(range(10)))
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
# PRE-LAB
##########################################################################################

X_train = []
y_train = []
X_test = []
y_test = []

##########################################################################################
# Step 1
##########################################################################################
print("Step 01: Successful Data reading")
print()
X_train, y_train = read_data("train.txt",X_train,y_train)
X_test, y_test = read_data("test.txt",X_test,y_test)
print("==================================================================")
print()

##########################################################################################
# Step 2
##########################################################################################
print("Step 02: Showing sample #131")
print()
plt.title("Sample 131")
show_sample(X_train, 130)
print("==================================================================")
print()

##########################################################################################
# Step 3
##########################################################################################
print("Step 03: Showing one random sample from all digits")
print()
plot_digits_samples(X_train, y_train)
print("==================================================================")
print()

##########################################################################################
# Step 4
##########################################################################################
print("Step 04: Mean value of all 0 samples")
print()
mean_0 = digit_mean_at_pixel(X_train,y_train,0)
print("Mean value of all 0 samples at pixel (10, 10) is", mean_0)
print()
print("==================================================================")
print()

##########################################################################################
# Step 5
##########################################################################################
print("Step 05: Variance of all 0 samples")
print()
var_0 = digit_variance_at_pixel(X_train,y_train,0)
print("Variance of all 0 samples at pixel (10, 10) is", var_0)
print()
print("==================================================================")
print()

##########################################################################################
# Step 6
##########################################################################################
print("Step 06: Mean value & Variance of all 0 samples")
print()
means_0 = digit_mean(X_train, y_train, 0)
varss_0 = digit_variance(X_train, y_train, 0)

# print("Step 6: Mean value of all 0 samples is", means_0)
# print("Step 6: Variance of all 0 samples is", varss_0)

print("==================================================================")
print()

##########################################################################################
# Step 7
##########################################################################################
print("Step 07: Digit 0 mean")
print()
plt.imshow(means_0)
plt.title("Mean pixel values from Samples labeled as $0$")
plt.show()
print("==================================================================")
print()

##########################################################################################
# Step 8
##########################################################################################
print("Step 08: Digit 0 variance")
print()
plt.imshow(varss_0)
plt.title("Variance of the pixels from Samples labeled as $0$")
plt.show()
print("==================================================================")
print()

##########################################################################################
# Step 9
##########################################################################################
print("Step 09: Mean value & Variance of all samples")
print()

X_mean = []
X_var = []

for h in range(10):
    temp_mean = digit_mean(X_train,y_train,h)
    X_mean.append(temp_mean)
plot_digits_samples(X_mean, list(range(10)))

# for h in range(10):
#     temp_var = digit_variance(X_train,y_train,h)
#     X_var.append(temp_var)
# plot_digits_samples(X_var, list(range(10)))

print("==================================================================")
print()

##########################################################################################
# Step 10
##########################################################################################
print("Step 10: Showing digit #101 from test dataset")
print()

digit_101 = np.reshape(X_test[100],(16,16))
plt.imshow(digit_101)
plt.title("Digit 101")
plt.show()
norms_101 = np.array([euclidean_distance(digit_101,X_mean[d]) for d in range(10)])
idx = int(np.argmin(norms_101))
print("Test digit #101 is classified as", idx)
print()
print("==================================================================")
print()

##########################################################################################
# Step 11
##########################################################################################
print("Step 11: Accuracy of Euclidean classifier")
print()
predictions = euclidean_distance_classifier(X_test,X_mean)
fail = np.count_nonzero( [ e1 - e2 for e1, e2 in zip( predictions, y_test ) ] )
print("Accuracy of euclidean classifier %.5f" %(1-fail/len(predictions)))
print()
print("==================================================================")
print()

##########################################################################################
# Step 12
##########################################################################################
print("Step 12: Accuracy for Euclidean Distance Classifier")
print()
euclideanClassifier = EuclideanClassifier()
euclideanClassifier.fit(X_train, y_train)
print()
print("Accuracy for Euclidean Distance Classifier is %.5f" %euclideanClassifier.score(X_test, y_test))
print()
print("==================================================================")
print()

##########################################################################################
# Step 13 (a)
##########################################################################################
print("Step 13: (a) Mean score of 5-fold-cross-validatio")
print()
cv_X = np.vstack((X_train,X_test))
cv_y = np.array([*y_train,*y_test])
cv_score = evaluate_classifier(euclideanClassifier,cv_X,cv_y)
print("Mean score of 5-fold-cross-validation: %.5f" %cv_score)
print()

##########################################################################################
# Step 13 (b)
##########################################################################################
print("Step 13: (b) Decision Region of Euclidean Classifier")
print()
pca = PCA(n_components=2) #Initialize PCA from 256 features to 2
X_train_PCA = pca.fit_transform(X_train)
euclideanClassifier.fit(X_train_PCA,y_train)
# plot_clf does the scatter of all samples on 2d space
plot_clf(euclideanClassifier,X_train_PCA,y_train,list(range(10)))
# plot decision regions plots the actual decision regions
plot_decision_regions(euclideanClassifier,X_train_PCA,y_train,list(range(10)))
print()

##########################################################################################
# Step 13 (c)
##########################################################################################
print("Step 13: (c) Learning Curve of Euclidean Classifier")
print()
my_learning_curve(euclideanClassifier,X_train,y_train,X_test,y_test)
my_error_curve(euclideanClassifier,X_train,y_train,X_test,y_test)

print("==================================================================")
print()

##########################################################################################
# END OF PRE-LAB
##########################################################################################

##########################################################################################
# MAIN LAB 
##########################################################################################

##########################################################################################
# Step 14
##########################################################################################
print("Step 14: a-priori probabilities")
print()
aprioris = calculate_priors(X_train,y_train)
print(aprioris)
print()
print("==================================================================")
print()

##########################################################################################
# Step 15
##########################################################################################
print("Step 15: Gaussian Naive Bayes Classifier & Sklearn's GaussianNB Classifier")
print()
gaussian_naive_bayes = CustomNBClassifier()
gaussian_naive_bayes.fit(X_train,y_train)
print()
print("Gaussian Naive Bayes accuracy score: %.5f" % gaussian_naive_bayes.score(X_test,y_test))
clf = GaussianNB()
clf.fit(X_train,y_train)
print()
print("Sklearn's GaussianNB accuracy is: %.5f" % clf.score(X_test,y_test))
print()
print("==================================================================")
print()

##########################################################################################
# Step 16
##########################################################################################
print("Step 16: Gaussian Naive Bayes Classifier & Sklearn's GaussianNB Classifier for Variance = 1")
print()
uniform_naive_bayes = CustomNBClassifier(True)
uniform_naive_bayes.fit(X_train,y_train)
print()
print("Gaussian Naive Bayes with unit variance accuracy score: %.5f" % uniform_naive_bayes.score(X_test,y_test))
print()
print("==================================================================")
print()

##########################################################################################
# Step 17
##########################################################################################
print("Step 17: Comparison between Naive Bayes, Nearest Neighbors & SVMs")
print()
cv_X = np.vstack((X_train,X_test))
cv_y = np.array([*y_train, *y_test])
score_sklearn_gnb = evaluate_sklearn_nb_classifier(cv_X,cv_y)
print("Sklearn's Gaussian Naive Bayes cross validation score: %.5f" % score_sklearn_gnb)
print()
score_custom_gnb = evaluate_custom_nb_classifier(cv_X,cv_y)
print()
print("Custom Gaussian Naive Bayes cross validation score: %.5f" % score_custom_gnb)
print()
score_knn = evaluate_knn_classifier(cv_X,cv_y)
print("K-Nearest neighbors cross validation score for number of neighbors in range [1,10]", score_knn)
print()
score_linear_svm = evaluate_linear_svm_classifier(cv_X,cv_y)
print("Linear SVM cross validation score: %.5f" % score_linear_svm)
print()
score_rbf_svm = evaluate_rbf_svm_classifier(cv_X,cv_y)
print("RBF SVM cross validation score: %.5f" % score_rbf_svm)
print()
score_euclidean = evaluate_euclidean_classifier(cv_X,cv_y)
print()
print("Euclidean cross validation score: %.5f" % score_euclidean)
print()
print("==================================================================")
print()

##########################################################################################
# Step 18 (a)
##########################################################################################
print("Step 18: (a) Voting Classifier")
print()

clfs = [GaussianNB(), EuclideanClassifier(), SVC(kernel='linear',gamma='auto'), SVC(kernel='rbf',gamma='auto')]
names = ['GaussianNB', 'EuclideanClassifier', 'Linear SVM', 'RBF SVM']

cv_X = np.vstack((X_train,X_test))
cv_y = np.array([*y_train, *y_test])

plot_mispredictions(clfs, X_train, y_train, names, 2, 2, 'barplot')

clfs = []
names = []
for k in range(2,11):
    clfs.append(KNeighborsClassifier(n_neighbors = k))
    names.append(f'Nearest neighbors, k = {k}')

plot_mispredictions(clfs, X_train, y_train, names, 3, 3, 'barplot')

clf1 = GaussianNB()
clf2 = SVC(kernel='linear',gamma='auto', probability=True)
clf3 = KNeighborsClassifier(n_neighbors = 2)
clf4 = EuclideanClassifier()

vclf1 = VotingClassifier(estimators=[('gnb', clf1), ('lsvm', clf2), ('2nn', clf3)], voting='hard')
vclf1.fit(X_train, y_train)
score = evaluate_classifier(vclf1, X_test, y_test, 5)
print()
print(f'Voting classifier 1 with hard voting: {score}')
print()

vclf2 = VotingClassifier(estimators=[('gnb', clf1), ('lsvm', clf2), ('2nn', clf4)], voting='hard')
vclf2.fit(X_train, y_train)
score = evaluate_classifier(vclf2, X_test, y_test, 5)
print()
print(f'Voting classifier 2 with hard voting: {score}')

score = evaluate_voting_classifier(X_test, y_test)
print()
print(f'Voting classifier 1 with soft voting: {score}')
print()

print("==================================================================")
print()

##########################################################################################
# Step 18 (b)
##########################################################################################
print("Step 18: (b) Voting Classifier")
print()

baggingClf = BaggingClassifier(base_estimator=SVC(kernel='linear'),n_estimators=10).fit(X_train, y_train)
score = evaluate_classifier(baggingClf, X_test, y_test, 5)
print()
print(f'Bagging classifier with Linear SVM: {score}')
print()

baggingClf2 = BaggingClassifier(base_estimator=EuclideanClassifier(),n_estimators=10).fit(X_train, y_train)
score = evaluate_classifier(baggingClf2, X_test, y_test, 5)
print()
print(f'Bagging classifier with EuclideanClassifier: {score}')

baggingClf3 = BaggingClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=10).fit(X_train, y_train)
score = evaluate_classifier(baggingClf3, X_test, y_test, 5)
print()
print(f'Random Forest: {score}')
print()

print("==================================================================")
print()

##########################################################################################
# Step 19 (Bonus)
##########################################################################################
print("Step 19: Introduction to Neural Networks & PyTorch")
print()

cv_X = np.vstack((X_train,X_test))
cv_y = np.array([*y_train, *y_test])
cv_score = evaluate_nn_classifier(cv_X, cv_y)
print(f'NN 5-fold average accuracy: {cv_score}')
print()
print("==================================================================")
print()

##########################################################################################
