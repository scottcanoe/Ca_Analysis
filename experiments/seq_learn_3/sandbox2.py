
from types import SimpleNamespace
from typing import (
    Any,
    Callable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import dask.array as da
import fs as pyfs
import fs.errors

import h5py
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np

import xarray as xr
from scipy.stats import ks_2samp
from sklearn import manifold
from sklearn.metrics import euclidean_distances
from ca_analysis import *
from ca_analysis.plot import *

from main import *
from utils import *


def add_2d_scatter(ax, points, points_color, title=None):
    x, y = points.T
    ax.scatter(x, y, c=points_color, s=50, alpha=0.8)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())

def plot_2d(points, points_color, title):
    fig, ax = plt.subplots(figsize=(3, 3), facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=16)
    add_2d_scatter(ax, points, points_color)
    plt.show()

def plot_3d(points, points_color, title):
    x, y, z = points.T

    fig, ax = plt.subplots(
        figsize=(6, 6),
        facecolor="white",
        tight_layout=True,
        subplot_kw={"projection": "3d"},
    )
    fig.suptitle(title, size=16)
    col = ax.scatter(x, y, z, c=points_color, s=50, alpha=0.8)
    ax.view_init(azim=-60, elev=9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.colorbar(col, ax=ax, orientation="horizontal", shrink=0.6, aspect=60, pad=0.01)
    plt.show()


n_components = 2
day = 5
event = 'ABCD'

roi_filter = 'all'

# create distance matrix
sessions = get_sessions(day=day)
apply_roi_filter(sessions, roi_filter)
arrays = flex_split(sessions, event)
data = arrays.mean('trial').transpose('time', ...)

# create colors for plotting
smap = get_smap('inferno', vlim=[0, 1])
S_colors = []
for x in np.linspace(0, 1, data.sizes['time']):
    S_colors.append(smap([x]))

mds = manifold.MDS(
    n_components=n_components,
    random_state=0,
)
S_sc = mds.fit_transform(data)
if n_components == 2:
    plot_2d(S_sc, S_colors, f'{event} day {day}')
else:
    plot_3d(S_sc, S_colors, f'{event} day {day}')

