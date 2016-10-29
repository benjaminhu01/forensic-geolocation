import numpy as np

from .utils import pairwise_distances
from .utils import to_categorical

class VoronoiClassifier(object):
    """
    Template for supervised classifier.
    """

    def __init__(self, model, domain, n_cells):
        self.model = model
        self.seeds = domain.sample(n_cells).reset_index(drop=True)
        self.seeds.index.set_names('seeds', inplace=True)
        self.n_cells = n_cells

    def fit(self, X, s, binary_matrix=False, **kwargs):
        y = self._assign_cells(s)
        if binary_matrix:
            y = to_categorical(y, self.n_cells)
        self.model.fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        return self.model.predict(X, **kwargs)

    def predict_proba(self, X, **kwargs):
        return self.model.predict_proba(X, **kwargs)
    
    def _assign_cells(self, s):
        dist = pairwise_distances(s, self.seeds)
        cells = (dist.ix[dist.groupby(level=0).idxmin()]
                     .index.get_level_values(level=1))
        return cells
