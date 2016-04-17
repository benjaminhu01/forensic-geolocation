
import numpy as np

class Geolocator(object):

    def __init__(self, domain, classifiers):
        self.domain = np.array(domain)
        self.classifiers = classifiers

    def score_locations(self, X):
        scores = np.zeros((X.shape[0], len(self.domain)))
        for c in self.classifiers:
            probs = c.predict(X)
            labels = [c.nearest_seed(s) for s in self.domain]
            counts = np.bincount(labels, minlength = len(c.seeds))
            scores += probs[:, labels] / counts[labels][np.newaxis, :]
        scores /= np.sum(scores, 1)[:, np.newaxis]
        return scores

    def predict(self, X):
        scores = self.score_locations(X)
        points = self.domain[np.apply_along_axis(np.argmax, 1, scores)]
        return points

    def predict_region(self, X, q):
        scores = self.score_locations(X)
        points = self.envelope(scores, q)
        return points

    def envelope(self, scores, q):
        assert 0. <= q <= 1., "Quantile value must belong to [0, 1]."
        points = []
        for score in scores:
            order = np.argsort(score)
            in_region = np.cumsum(score[order]) >= 1. - q
            if not np.any(in_region):
                # scores may not sum perfectly to 1 due to machine precision
                # so make sure to force inclusion of the final point
                in_region[-1] = np.ones(1, dtype = bool)
            points.append(self.domain[order[in_region]])
        return points

    # def envelope(self, scores, q):
    #     assert q >= 0. and q <= 1., "Quantile value must belong to [0, 1])."
    #     order = np.argsort(scores)
    #     in_region = np.cumsum(scores[order]) >= 1. - q
    #     if not np.any(in_region):
    #         # scores may not sum perfectly to 1 due to rounding error
    #         # so make sure to force inclusion of the final point
    #         in_region[-1] = np.ones(1, dtype = bool)
    #     return self.domain[order[in_region]]
