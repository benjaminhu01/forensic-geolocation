from __future__ import print_function

import os
import random
import time

import numpy as np
import pandas as pd
import deepspace as ds

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD

from sklearn.ensemble import RandomForestClassifier

np.random.seed(881)  # for reproducibility

def main():
    # Load example data from the data directory
    domain = load_domain()
    data = load_homes1000()
    n = data.shape[0]
    
    # Make training and testing data
    assignment = [np.random.choice(['train', 'test'], p=[0.8, 0.2]) for _ in range(n)]
    grouped = data.groupby(assignment)
    s_test, X_test = split_s_and_X(grouped.get_group('test'))
    s_train, X_train = split_s_and_X(grouped.get_group('train'))
    p = X_train.shape[1]

    # Weight sample points proportionally by state population
    weights = scale_by_population(grouped.get_group('train')[['state']])

    # Fit deep neural network (DNN) ensemble of Voronoi classifiers
    n_cells_min, n_cells_max = (30, 60)
    n_classifiers = 3
    ensemble = []
    for i in range(n_classifiers):
        print('Training classifier %d of %d' % (i+1, n_classifiers))
        n_cells = np.random.randint(n_cells_min, n_cells_max)
        model = Sequential()
        model.add(Dense(512, input_shape=(p, )))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(n_cells))
        model.add(Activation('softmax'))
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        clf = ds.VoronoiClassifier(model, domain, n_cells)
        start = time.clock()
        clf.fit(X_train.as_matrix(), s_train, binary_matrix=True,
                nb_epoch=20, batch_size=64, sample_weight=weights.as_matrix().flatten())
        end = time.clock()
        print(end - start)
        ensemble.append(clf)

    # Predict origin of testing data
    geo = ds.Geolocator(ensemble, domain)
    predictions = geo.predict(X_test)
    errors = ds.distance(s_test, predictions)
    print(errors.describe())
    region_90 = geo.predict_regions(X_test, quantile=0.9)
    
    # Fit random forest ensemble of Voronoi classifiers
    ensemble = []
    for i in range(n_classifiers):
        print('Training classifier %d of %d' % (i+1, n_classifiers))
        n_cells = np.random.randint(n_cells_min, n_cells_max)
        model = RandomForestClassifier(n_estimators=100)
        clf = ds.VoronoiClassifier(model, domain, n_cells)
        start = time.clock()
        clf.fit(X_train.as_matrix(), s_train,
                sample_weight=weights.as_matrix().flatten())
        end = time.clock()
        print(end - start)
        ensemble.append(clf)

    # Predict origin of testing data
    geo = ds.Geolocator(ensemble, domain)
    predictions = geo.predict(X_test)
    errors = ds.distance(s_test, predictions)
    print(errors.describe())
    region_90 = geo.predict_regions(X_test, quantile=0.9)

def get_data_dir():
    return '/Users/nsgrantham/Code/deepspace/data/'

def load_domain():
    data_dir = get_data_dir()
    domain = pd.read_csv(os.path.join(data_dir, 'homes1000/grid-lonlat.csv'))
    domain.index.set_names('domain', inplace=True)
    return domain[['lat', 'lon']]

def load_homes1000():
    homes1000_dir = os.path.join(get_data_dir(), 'homes1000')
    locations_file = os.path.join(homes1000_dir, 'home-lonlat.csv')
    locations = pd.read_csv(locations_file, index_col=['ids'])
    covariates_file = os.path.join(homes1000_dir, 'home-covariates.csv')
    covariates = pd.read_csv(covariates_file, index_col=['ids'])
    states = covariates['State'].rename('state')
    values_file = os.path.join(homes1000_dir, 'fungi-OTU-presence-small.csv')
    values = pd.read_csv(values_file, index_col=['ids'])
    return pd.concat([locations, states, values], axis=1)

def scale_by_population(states):
    data_dir = get_data_dir()
    uspop = pd.read_csv(os.path.join(data_dir, 'homes1000/US-states.csv'), 
            header=None, usecols=(1, 2))
    uspop.columns = ['state', 'pop']
    uspop.set_index('state', inplace=True)
    scale_by_size = lambda x: x / len(x)
    weights = (states.join(uspop, on='state')
                     .groupby('state')
                     .transform(scale_by_size))
    return weights / weights.mean()

def split_s_and_X(data):
    s = data[['lat', 'lon']]
    X = data[[col for col in data.columns if col.startswith('OTU')]]
    return (s, X)

if __name__ == '__main__':
    main()
