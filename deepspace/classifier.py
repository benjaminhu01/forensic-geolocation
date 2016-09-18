"""
:class:`.Classifier` base object from which other classifiers are templated.
"""
import numpy as np

import imp
utils = imp.load_source('deepspace', 'deepspace/utils.py')

# from geopy.distance import distance

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils

class Classifier(object):
    """
    Template for supervised classifier.
    """

    def __init__(self, seeds):
        self.model = []
        self.seeds = seeds.set_index(pd.Series(range(seeds.shape[0])))

    def fit(self, data, nb_epoch=20, batch_size=32, verbose=0, sample_weight=None):
        y_train = self.nearest_seeds(locations(data))
        Y_train = np_utils.to_categorical(y_train, len(self.seeds))
        X_train = biomes(data).as_matrix()
        self.model = self.keras(input_dim=X_train.shape[1], output_dim=len(self.seeds))
        self.model.fit(X_train, Y_train, nb_epoch=nb_epoch, batch_size=batch_size,
          sample_weight=sample_weight, verbose=verbose, show_accuracy=True)

    def predict(self, X):
        return self.model.predict(X)

    def keras(self, input_dim, output_dim):
        """
        Build a four-layer deep classification network with ReLU neurons and dropout.
        """
        model = Sequential()
        model.add(Dense(512, input_shape=(input_dim,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(output_dim))
        model.add(Activation('softmax'))
        # sgd = SGD(lr=0.01, momentum=0.9, decay=0.0)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        return model

    def nearest_seeds(self, s):
        dist = utils.pairwise_distances(s, self.seeds)
        idx = dist.index.get_level_values(level=0).unique()
        return np.array([a[1] for a in dist.groupby(level=0).idxmin().reindex(idx)])

    # def nearest_seed(self, s):
    #     return np.argmin([distance(s, seed).kilometers for seed in self.seeds])
