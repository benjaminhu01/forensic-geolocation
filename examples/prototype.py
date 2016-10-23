from __future__ import print_function

import os

import deepspace as ds
import numpy as np
import pandas as pd
np.random.seed(881)  # for reproducibility

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils

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

if __name__ == '__main__':
    domain = load_domain()
    data = load_homes1000()
    n = data.shape[0]
    n_cells_min, n_cells_max = (25, 75)
    n_classifiers = 25
    n_folds = 10
    fold_ids = np.random.randint(low=0, high=n_folds, size=n)
    for fold in range(n_folds):
        assignment = ['test' if fold_id == fold else 'train' for fold_id in fold_ids]
        grouped = data.groupby(assignment)
        test = grouped.get_group('test')
        s_test = test[['lat', 'lon']]
        X_test = test[[col for col in test.columns if col.startswith('OTU')]]
        train = grouped.get_group('train')
        s_train = train[['lat', 'lon']]
        X_train = train[[col for col in train.columns if col.startswith('OTU')]]
        p = X_train.shape[1]
        weights = scale_by_population(train[['state']])
        ensemble = []
        for j in range(n_classifiers):
            print('Training classifier %d/%d in fold %d/%d...' % (j, n_classifiers, fold, n_folds))
            n_cells = np.random.randint(n_cells_min, n_cells_max)
            model = Sequential()
            model.add(Dense(256, input_shape=(p, )))
            model.add(Activation('relu'))
            model.add(Dropout(0.2))
            model.add(Dense(256))
            model.add(Activation('relu'))
            model.add(Dropout(0.2))
            model.add(Dense(256))
            model.add(Activation('relu'))
            model.add(Dropout(0.2))
            model.add(Dense(256))
            model.add(Activation('relu'))
            model.add(Dropout(0.2))
            model.add(Dense(n_cells))
            model.add(Activation('softmax'))
            sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
            model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
            clf = ds.VoronoiClassifier(model, domain, n_cells)
            clf.fit(X_train.as_matrix(), s_train, to_categorical=True, 
                    sample_weight=weights.as_matrix().flatten(), 
                    nb_epoch=50, batch_size=64, verbose=2)
            ensemble.append(clf)
        geo = ds.Geolocator(ensemble, domain)
        predictions = geo.predict(X_test)
        errors = ds.distance(s_test, predictions)
        print(errors.describe())
        region_50 = geo.predict_regions(X_test, quantile=0.5)
        region_75 = geo.predict_regions(X_test, quantile=0.75)
        region_90 = geo.predict_regions(X_test, quantile=0.9)
        quit()
