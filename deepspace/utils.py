
import os
import copy

import numpy as np
import pandas as pd

def get_data_dir():
    return '/Users/nsgrantham/Documents/deepspace/data/'

def load_domain():
    data_dir = get_data_dir()
    domain = pd.read_csv(os.path.join(data_dir, 'homes1000/grid-lonlat.csv'))
    return domain[['lat', 'lon']]

def load_data(minimal=True):
    data_dir = get_data_dir()
    path = 'homes1000/fungi-OTU-presence-small.csv' if minimal else 'homes1000/fungi-OTU-presence-large.csv'
    states = pd.read_csv(os.path.join(data_dir, 'homes1000/home-covariates.csv'), index_col=['ids'])['State']
    origins = pd.read_csv(os.path.join(data_dir, 'homes1000/home-lonlat.csv'), index_col=['ids'])
    biomes = pd.read_csv(os.path.join(data_dir, path), index_col=['ids'])
    data = pd.concat([states, origins, biomes], axis=1)
    return data

def load_data_precise(minimal=True):
    data_dir = get_data_dir()
    path = 'homes1000/fungi-OTU-presence-small.csv' if minimal else 'homes1000/fungi-OTU-presence-large.csv'
    origins = pd.read_csv(os.path.join(data_dir, 'homes1000/home-lonlat-full.csv'), index_col=['ids'])
    biomes = pd.read_csv(os.path.join(data_dir, path), index_col=['ids'])
    data = pd.concat([origins, biomes], axis=1, join='inner')
    return data

def partition(data, folds):
    to_fold = np.random.randint(low=0, high=folds, size=data.shape[0])
    return [data.loc[to_fold == f, :] for f in xrange(folds)]

def split(data, fold):
    d = copy.copy(data)
    return d.pop(fold), pd.concat(d)

def biomes(data):
    return data.filter([col for col in data.columns if col.startswith('OTU')])

def locations(data):
    return data[['lat', 'lon']]

def states(data):
    return data['State']

def pairwise_distances(coord1, coord2=None):
    if coord2 is None:
        coord2 = coord1
    idx = pd.MultiIndex.from_product([coord1.index, coord2.index],
                                     names=['origin', 'other'])
    pairs = pd.concat([coord1.add_suffix('1').reindex(idx, level='origin'),
                       coord2.add_suffix('2').reindex(idx, level='other')],
                       axis=1)
    dist = great_circle(pairs['lat1'], pairs['lon1'], pairs['lat2'], pairs['lon2'])
    return dist

def pairwise_distance_matrix(coord1, coord2=None):
    dist = pairwise_distances(coord1, coord2)
    return dist.unstack(level=1)

def great_circle(lat1, lon1, lat2, lon2, miles=False):
    '''
    Calculate great circle distance.
    http://www.johndcook.com/blog/python_longitude_latitude/

    Parameters
    ----------
    lat1, lon1, lat2, lon2: float or array of float

    Returns
    -------
    distance:
      distance from ``(lat1, lon1)`` to ``(lat2, lon2)`` in kilometers
      or miles (if miles kwarg is True).
    '''
    phi1 = np.deg2rad(90 - lat1)
    phi2 = np.deg2rad(90 - lat2)

    theta1 = np.deg2rad(lon1)
    theta2 = np.deg2rad(lon2)

    cos = (np.sin(phi1) * np.sin(phi2) * np.cos(theta1 - theta2) +
           np.cos(phi1) * np.cos(phi2))
    arc = np.arccos(np.clip(cos, -1, 1))
    k = 3960 if miles else 6373 # kilometers
    return arc * k

def sample_weight(data):
    uspop = pd.read_csv('data/homes1000/US-states.csv', header=None, usecols=(1, 2))
    uspop = uspop.set_index(1)[2].to_dict()
    total = float(sum(uspop.values()))
    for key, value in uspop.iteritems():
        uspop[key] = value / total
    states = utils.states(data)
    unique, counts = np.unique(states, return_counts=True)
    obs_per_state = dict(zip(unique, counts))
    return np.array([uspop[state] / obs_per_state[state] for state in states])
