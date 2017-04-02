import numpy as np
import pandas as pd

from .utils import distance

class Geolocator(object):

    def __init__(self, ensemble, domain):
        self.ensemble = ensemble
        self.domain = domain

    def evaluate_likelihood(self, X, normalize=False, **kwargs):
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
        if normalize:
            likelihood = (likelihood.groupby(level='ids').transform(lambda x: x / x.sum())
                            .rename(index='probability', inplace=True))
        return likelihood

    def predict(self, X, **kwargs):
        likelihood = self.evaluate_likelihood(X, **kwargs)
        idx = likelihood.groupby(level='ids').idxmax()
        preds = (likelihood.ix[idx]
                           .to_frame()
                           .join(self.domain, how='inner'))
        return preds

    def predict_regions(self, X, quantile, **kwargs):
        probs = self.evaluate_likelihood(X, normalize=True, **kwargs)
        preds = self.threshold(probs, quantile)
        return preds

    def score(self, X, s, **kwargs):
        preds = self.predict(X, **kwargs)
        errors = distance(s, preds)
        return errors
    
    def threshold(self, probs, quantile):
        idx = probs.index
        cum_probs = (probs.groupby(level='ids')
                    .apply(lambda x: x.sort_values(ascending=False).cumsum()))
        cum_probs.index = cum_probs.index.droplevel(0)
        def determine_region(x):
            reg = pd.Series(1.0, index=x.index, dtype=float)
            for q in sorted(quantile)[::-1]:
                assert 0. <= q <= 1., "Quantile must belong to [0, 1]."
                is_within_region = (x < q).shift(1).fillna(x[0] < q)
                reg[is_within_region] = q
            return reg
        preds = (cum_probs.groupby(level='ids')
                .apply(determine_region)
                .sort_index(level=0)
                .rename('prob_region'))
        return preds
