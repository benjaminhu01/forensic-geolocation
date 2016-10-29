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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

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
    weights = weights.as_matrix().flatten() 

    # Fit deep neural network (DNN) ensemble of Voronoi classifiers
    n_cells_min, n_cells_max = (30, 60)
    n_classifiers = 10
    ensemble_nnc = []
    ensemble_knc = []
    ensemble_svc = []
    ensemble_rfc = []
    for i in range(n_classifiers):
        print('Training classifier %d of %d' % (i+1, n_classifiers))
        n_cells = np.random.randint(n_cells_min, n_cells_max)
        nnc = Sequential()
        nnc.add(Dense(512, input_shape=(p, )))
        nnc.add(Activation('relu'))
        nnc.add(Dropout(0.2))
        nnc.add(Dense(512))
        nnc.add(Activation('relu'))
        nnc.add(Dropout(0.2))
        nnc.add(Dense(512))
        nnc.add(Activation('relu'))
        nnc.add(Dropout(0.2))
        nnc.add(Dense(512))
        nnc.add(Activation('relu'))
        nnc.add(Dropout(0.2))
        nnc.add(Dense(n_cells))
        nnc.add(Activation('softmax'))
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        nnc.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        knc = KNeighborsClassifier(n_neighbors=20)
        svc = SVC(probability=True)
        rfc = RandomForestClassifier(n_estimators=100)
        clf_nnc = ds.VoronoiClassifier(nnc, domain, n_cells)
        clf_knc = ds.VoronoiClassifier(knc, domain, n_cells)
        clf_svc = ds.VoronoiClassifier(svc, domain, n_cells)
        clf_rfc = ds.VoronoiClassifier(rfc, domain, n_cells)
        clf_nnc.fit(X_train.as_matrix(), s_train, sample_weight=weights, 
                binary_matrix=True, nb_epoch=30, batch_size=32)
        clf_knc.fit(X_train.as_matrix(), s_train)
        clf_svc.fit(X_train.as_matrix(), s_train, sample_weight=weights)
        clf_rfc.fit(X_train.as_matrix(), s_train, sample_weight=weights)
        ensemble_nnc.append(clf_nnc)
        ensemble_knc.append(clf_knc)
        ensemble_svc.append(clf_svc)
        ensemble_rfc.append(clf_rfc)

    # Predict origin of testing data
    ensemble = ensemble_nnc + ensemble_knc + ensemble_svc + ensemble_rfc
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
