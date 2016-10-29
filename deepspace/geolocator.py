import numpy as np
import pandas as pd

from .utils import distance

class Geolocator(object):

    def __init__(self, ensemble, domain):
        self.ensemble = ensemble
        self.domain = domain

    def evaluate_likelihood(self, X, **kwargs):
        idx = pd.MultiIndex.from_product([X.index, self.domain.index],
                                         names=['ids', 'domain'])
        likelihood = pd.Series(0, index=idx, name='likelihood', dtype=float)
        scale_by_size = lambda x: x / len(x)
        for clf in self.ensemble:
            cells = clf._assign_cells(self.domain)
            cells = pd.Series(cells, index=self.domain.index, name='cell')
            proba = clf.predict_proba(X.as_matrix(), **kwargs).T
            if hasattr(clf.model, "classes_"):
                classes = clf.model.classes_
            else:
                classes = clf.seeds.index
            proba = pd.DataFrame(proba, index=classes, columns=X.index)
            points = (cells.to_frame()
                           .join(proba, on='cell')
                           .groupby('cell')
                           .transform(scale_by_size)
                           .unstack()
                           .rename('likelihood'))
            likelihood = likelihood.add(points)
        return likelihood

    def predict(self, X, **kwargs):
        likelihood = self.evaluate_likelihood(X, **kwargs)
        idx = likelihood.groupby(level='ids').idxmax()
        preds = (likelihood.ix[idx]
                           .to_frame()
                           .join(self.domain, how='inner'))
        return preds

    def predict_regions(self, X, quantile, **kwargs):
        likelihood = self.evaluate_likelihood(X, **kwargs)
        probs = likelihood.groupby(level='ids').transform(lambda x: x / x.sum())
        preds = self.threshold(probs, quantile)
        return preds

    def score(self, X, s, **kwargs):
        preds = self.predict(X, **kwargs)
        errors = distance(s, preds)
        return errors
    
    def threshold(self, probs, quantile):
        assert 0. <= quantile <= 1., "Quantile must belong to [0, 1]."
        cum_probs = (probs.groupby(level='ids')
                          .transform(lambda x: x.sort_values(ascending=False).cumsum()))
        preds = (cum_probs.groupby(level='ids')
                .apply(lambda x: x[(x < quantile).shift(1).fillna(x[0] < quantile)]))
        return preds
