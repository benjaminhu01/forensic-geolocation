import numpy as np
import pandas as pd

class Geolocator(object):

    def __init__(self, ensemble, domain):
        self.ensemble = ensemble
        self.domain = domain

    def score(self, X, normalize=False):
        idx = pd.MultiIndex.from_product([X.index, self.domain.index],
                                         names=['ids', 'domain'])
        scores = pd.Series(0, index=idx, name='score', dtype=float)
        scale_by_size = lambda x: x / len(x)
        for clf in self.ensemble:
            cells = clf._assign_cells(self.domain)
            cells = pd.Series(cells, index=self.domain.index, name='cell')
            proba = clf.predict_proba(X.as_matrix()).T
            proba = pd.DataFrame(proba, index=clf.seeds.index, columns=X.index)
            points = (cells.to_frame()
                           .join(proba, on='cell')
                           .groupby('cell')
                           .transform(scale_by_size)
                           .unstack()
                           .rename('score'))
            scores = scores.add(points)
        if normalize:
            scores = scores.groupby(level='ids').transform(lambda x: x / x.sum())
        return scores

    def predict(self, X):
        scores = self.score(X)
        idx = scores.groupby(level='ids').idxmax()
        preds = (scores.ix[idx]
                       .to_frame()
                       .join(self.domain, how='inner'))
        return preds

    def predict_regions(self, X, quantile):
        scores = self.score(X, normalize=True)
        preds = self.threshold(scores, quantile)
        return preds
    
    def threshold(self, scores, quantile):
        assert 0. <= quantile <= 1., "Quantile must belong to [0, 1]."
        cum_scores = (scores.groupby(level='ids')
                            .transform(lambda x: x.sort_values(ascending=False).cumsum()))
        preds = (cum_scores.groupby(level='ids')
                .apply(lambda x: x[(x < quantile).shift(1).fillna(x[0] < quantile)]))
        print(preds.groupby(level='ids').max())
        return preds
