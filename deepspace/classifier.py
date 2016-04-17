"""
:class:`.Classifier` base object from which other classifiers are templated.
"""
import numpy as np

from geopy.distance import distance

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

class Classifier(object):
    """
    Template for supervised classifier.
    """

    def __init__(self, seeds):
        self.model = []
        self.seeds = seeds

    def fit(self, X_train, S_train, nb_epoch=20, batch_size=32, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones(X_train.shape[0])
        y_train = np.asarray([self.nearest_seed(s) for s in S_train])
        Y_train = np_utils.to_categorical(y_train, len(self.seeds))
        self.model = self.keras(input_dim=X_train.shape[1], output_dim=len(self.seeds))
        self.model.fit(X_train, Y_train, nb_epoch=20, batch_size=batch_size,
            sample_weight=sample_weight, verbose=1, show_accuracy=True)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def keras(self, input_dim, output_dim):
        """
        Build a four-layer deep classification network with ReLU neurons and dropout.
        """
        model = Sequential()
        model.add(Dense(128, input_shape=(input_dim,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(output_dim))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        return model

    def nearest_seed(self, s):
        return np.argmin([distance(s, seed).kilometers for seed in self.seeds])
