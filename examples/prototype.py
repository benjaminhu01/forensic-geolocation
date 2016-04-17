from __future__ import print_function

import os
os.chdir('/Users/nsgrantham/Documents/deepspace/')

import imp
geolocator = imp.load_source('deepspace', 'deepspace/geolocator.py')
classifier = imp.load_source('deepspace', 'deepspace/classifier.py')

import numpy as np
import pandas as pd
np.random.seed(881)  # for reproducibility

from geopy.point import Point
from geopy.distance import distance

def main():
    states = np.loadtxt('data/homes1000/home-covariates.csv', delimiter=',', skiprows=1, usecols=(2,), dtype=str)
    states = np.array([state.replace('"', '') for state in states])
    usa = np.loadtxt('data/homes1000/grid-lonlat.csv', delimiter=',', skiprows=1, usecols=(1, 0))
    domain = [Point(coord) for coord in usa]
    del usa
    X = np.loadtxt('data/homes1000/fungi-OTU-presence-small.csv', delimiter=',', skiprows=1, dtype=int)
    ids = X[:, 0]
    X = np.delete(X, 0, 1)  # remove home id column
    n, p = X.shape
    homes = np.loadtxt('data/homes1000/home-lonlat.csv', delimiter=',', skiprows=1, usecols=(2, 1))
    origins = np.array([Point(coord) for coord in homes])
    del homes
    K_min, K_max = (30, 60)

    N = 10  # number of nets to train
    batch_size = 64
    # set up cross-validation
    nfold = 10
    to_fold = np.random.randint(low=0, high=nfold, size=n)
    for f in xrange(nfold):
        test = to_fold == f
        ntest = sum(test)
        id_test = ids[test]
        train = np.logical_not(test)
        X_train = X[train, :]
        origins_train = origins[train]
        X_test = X[test, :]
        origins_test = origins[test]

        weights = weight_by_state(states[train])
        classifiers = [classifier.Classifier(get_seeds(domain, K_min, K_max)) for _ in xrange(N)]
        for j, c in enumerate(classifiers):
            print('Training classifier %d of %d in fold %d...' % (j+1, N, f+1))
            c.fit(X_train, origins_train, sample_weight=weights)
        geo = geolocator.Geolocator(domain, classifiers)
        predictions = geo.predict(X_test)
        errors = pd.Series([distance(*s).kilometers for s in zip(origins_test, predictions)])
        print(errors.describe())
        quit()

def get_seeds(domain, low, high):
    return np.random.choice(domain, np.random.randint(low, high + 1, 1))

def weight_by_state(states):
    uspop = pd.read_csv('data/homes1000/US-states.csv', header=None, usecols=(1, 2))
    uspop = uspop.set_index(1)[2].to_dict()
    total = float(sum(uspop.values()))
    for key, value in uspop.iteritems():
        uspop[key] = value / total
    unique, counts = np.unique(states, return_counts=True)
    obs_per_state = dict(zip(unique, counts))
    return np.array([uspop[state] / obs_per_state[state] for state in states])

if __name__ == '__main__':
    main()
