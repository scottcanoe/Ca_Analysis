import time

import intake
from intake.source.discovery import drivers, load_plugins_from_module
import weakref
from time import perf_counter as clock
from typing import List, Mapping

import dask.array as da
import h5py
import matlab.engine
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.animation import FFMpegWriter
import napari
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from main import *
from processing import *
from ca_analysis.plot import get_smap




sessions = [
    # no scopolamine
    # open_session('Thy1_0093', '2023-04-19', '1', fs=0),
    # open_session('Thy1_0093', '2023-04-19', '2', fs=0),
    open_session('M110', '2023-04-21', '1', fs=0),
    open_session('M110', '2023-04-21', '2', fs=0),
    open_session('M115', '2023-04-21', '1', fs=0),
    open_session('M115', '2023-04-21', '2', fs=0),
#     open_session('M150', '2023-04-21', '1', fs=0),
#     open_session('M150', '2023-04-21', '2', fs=0),
#     open_session('M152', '2023-04-21', '1', fs=0),
#     open_session('M152', '2023-04-21', '2', fs=0),
#     # open_session('M153', '2023-04-21', '1', fs=0),
#     # open_session('M153', '2023-04-21', '2', fs=0),
#
#     # scopolamine
#     open_session('Thy1_0093', '2023-04-25', '1', fs=0),
    # open_session('Thy1_0093', '2023-04-25', '2', fs=0),
    open_session('M110', '2023-04-25', '1', fs=0),
    open_session('M110', '2023-04-25', '2', fs=0),
    open_session('M115', '2023-04-25', '1', fs=0),
    open_session('M115', '2023-04-25', '2', fs=0),
]


s = sessions[0]

# arr = load_splits(s, 'ach', 'right')
# mov = arr.mean('trial')
# with h5py.File(s.fs.getsyspath('mean_mov.h5'), 'w') as f:
#     f.create_dataset('ach', data=mov)

with h5py.File(s.fs.getsyspath('mean_mov.h5'), 'r') as f:
    data = f['ach'][:]
mov = xr.DataArray(data, dims=('time', 'y', 'x'))
im_shape = mov.shape[1:]

mat = np.stack([im.flatten() for im in mov.data])
pca = PCA()
tformed = pca.fit_transform(mat)
comps = pca.components_
M = []
for c in pca.components_:
    im = c.reshape(im_shape)
    M.append(im)
