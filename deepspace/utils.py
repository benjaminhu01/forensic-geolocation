import numpy as np
import pandas as pd

def pairwise_distances(coord1, coord2=None):
    if coord2 is None:
        coord2 = coord1.copy()
    idx = pd.MultiIndex.from_product([coord1.index, coord2.index])
    pairs = pd.concat([coord1.add_suffix('1').reindex(idx, level=0),
                       coord2.add_suffix('2').reindex(idx, level=1)],
                       axis=1)
    dist = great_circle(pairs['lat1'], pairs['lon1'], pairs['lat2'], pairs['lon2'])
    return dist

def distance(coord1, coord2):
    coord2 = coord2.set_index(coord1.index)
    pairs = pd.concat([coord1.add_suffix('1'), coord2.add_suffix('2')], axis=1)
    dist = great_circle(pairs['lat1'], pairs['lon1'], pairs['lat2'], pairs['lon2'])
    return dist

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
    arc_length = 3960 if miles else 6373 # kilometers
    return arc * arc_length
