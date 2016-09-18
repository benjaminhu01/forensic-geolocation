from __future__ import print_function

import os
os.chdir('/Users/nsgrantham/Documents/deepspace/')

import imp
utils = imp.load_source('deepspace', 'deepspace/utils.py')
geolocator = imp.load_source('deepspace', 'deepspace/geolocator.py')
classifier = imp.load_source('deepspace', 'deepspace/classifier.py')

import numpy as np
import pandas as pd
np.random.seed(881)  # for reproducibility

if __name__ == '__main__':
    domain = utils.load_domain()
    full_data = utils.load_data_precise()
    K = (50, 100)
    N = 200  # number of nets to train
    data = utils.partition(full_data, folds=10)
    for f in xrange(len(data)):
        test, train = utils.split(data, f)
        # weights = utils.sample_weight(train)
        classifiers = [classifier.Classifier(domain.sample(np.random.randint(*K))) for _ in xrange(N)]
        for j, c in enumerate(classifiers):
            print('Training classifier %d of %d in fold %d...' % (j+1, N, f+1))
            c.fit(train, nb_epoch=10, batch_size=32, verbose=0)#, sample_weight=weights)
        geo = geolocator.Geolocator(domain, classifiers)
        print(geo.error(test).describe())
        quit()
