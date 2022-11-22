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
