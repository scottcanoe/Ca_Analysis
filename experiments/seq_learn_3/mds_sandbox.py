
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
# matplotlib.use('TkAgg')
import matplotlib.gridspec as gridspec
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


def plot_2d(ax, points, points_color):
    add_2d_scatter(ax, points, points_color)



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


# def pairwise_distances(arr: xr.DataArray) -> np.ndarray:
#     out = np.zeros(arr.sizes['time'] - 1)
#     for i in range(len(out)):
#         a, b = arr.isel(time=i), arr.isel(time=i+1)
#         out[i] = np.linalg.norm(a - b)
#     return out

def pairwise_distances(arr: xr.DataArray) -> np.ndarray:
    out = np.zeros(arr.sizes['time'] - 1)
    for i in range(len(out)):
        a, b = arr.isel(time=i), arr.isel(time=i+1)
        # a = a / np.linalg.norm(a)
        # b = b / np.linalg.norm(b)
        # out[i] = pearsonr(a, b).statistic
        out[i] = np.linalg.norm(a - b)
    return out


event = 'ACBD'
roi_filter = 'all'
n_components = 2
drop_gray = False
savedir = Path.home() / "plots/seq_learn_3/MDS"

sessions_0 = get_sessions(day=0)
sessions_5 = get_sessions(day=5)
apply_roi_filter(sessions_0 + sessions_5, roi_filter)

is_seq = event in ('ABCD', 'ABBD', 'ACBD')

# sessions_0 = sessions_0[0:1]
# sessions_5 = sessions_5[0:1]

# load data
if event == "gray":
    data_0 = split_all_gray(sessions_0).mean('trial').transpose('time', ...)
    data_5 = split_all_gray(sessions_5).mean('trial').transpose('time', ...)
else:
    data_0 = flex_split(sessions_0, event, drop_gray=drop_gray).mean('trial').transpose('time', ...)
    data_5 = flex_split(sessions_5, event, drop_gray=drop_gray).mean('trial').transpose('time', ...)


# do MDS
mds_kw = {
    'n_components': n_components,
    'random_state': 0,
}
mds_0 = manifold.MDS(**mds_kw)
points_0 = mds_0.fit_transform(data_0)

mds_5 = manifold.MDS(**mds_kw)
points_5 = mds_5.fit_transform(data_5)

# create colors for plotting
smap = get_smap('inferno', vlim=[0, 1])
colors = []
for x in np.linspace(0, 1, data_0.sizes['time']):
    colors.append(smap([x]))

# make MDS plot
# fig, axes = plt.subplots(1, 2, figsize=(9, 4))
fig = plt.figure(tight_layout=True)
gs = gridspec.GridSpec(2, 2)

ax = fig.add_subplot(gs[0, 0])
ax.set_aspect('equal')
points = points_0
add_2d_scatter(ax, points, colors)
ax.set_title('day 0')
if is_seq:
    starts = points[[0, 8, 16, 24]]
    x, y = starts.T
    ax.scatter(x, y, c='white', s=50, marker='+')

ax = fig.add_subplot(gs[0, 1])
ax.set_aspect('equal')
points = points_5
add_2d_scatter(ax, points, colors)
ax.set_title('day 5')
if is_seq:
    starts = points[[0, 8, 16, 24]]
    x, y = starts.T
    ax.scatter(x, y, c='white', s=50, marker='+')


a = pairwise_distances(data_0)
b = pairwise_distances(data_5)
# fig, ax = plt.subplots(figsize=(10, 3))
ax = fig.add_subplot(gs[1, :])
ax.plot(a, color='black', label='day 0')
ax.plot(b, color='red', label='day 5')
ax.set_xlim([0, len(a) - 1])
ax.legend()
if is_seq:
    last = None if drop_gray else '-'
    annotate_onsets(ax, data_0.coords['event'], last=last)
else:
    ax.set_xticks([0, 7, 14, 21])
    ax.set_xticklabels(['0', '250', '500', '250'])
    ax.set_xlabel('msec')

ax.set_title('pattern similarity (neighbor distances)')
ax.set_ylabel('euclidean distance')
ax.set_xlabel('time')
fig.suptitle(event)
plt.show()
fig.savefig(savedir / f"MDS_{event}.png")
