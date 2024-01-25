import copy
import datetime
import logging
import dataclasses
from numbers import Number
import os
from pathlib import Path
import shutil
from time import perf_counter as clock
from types import SimpleNamespace
from typing import (
    Any, Callable,
    List,
    Mapping,
    NamedTuple, Optional,
    Sequence,
    Tuple,
    Union,
)

import dask.array as da
import h5py
import ndindex as nd
import pandas as pd
import matplotlib
from matplotlib.figure import Figure
matplotlib.use('Qt5Agg')
import matplotlib.colors as mpc
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import *
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from statsmodels.distributions.empirical_distribution import ECDF
import xarray as xr

from ca_analysis import *
from ca_analysis.stats import *

from processing import *
from cycler import cycler
import matplotlib.pyplot as plt
from processing import *
import seaborn as sns
from sklearn.decomposition import PCA

from ca_analysis import *
from main import *
from utils import *

"""
See how well the decoder can tell which block an event came from.
"""


def get_corrs(s: Session, event: str) -> np.ndarray:


    arr = s.spikes.split(event)
    arr = arr.mean('time')

    n_trials = arr.sizes['trial']
    mat = np.zeros([n_trials, n_trials])
    for i in range(n_trials - 1):
        a = arr[i]
        # a = a / np.linalg.norm(a)
        for j in range(i, n_trials):
            b = arr[j]
            score = pearsonr(a, b).statistic
            mat[i, j] = score

    corrs = np.zeros(n_trials - 1)
    for i in range(n_trials - 1):
        corrs[i] = np.diagonal(mat, i).mean()
    return corrs

"""
Look at drift over 1 sec periods using ABCD.
"""
# day = 5
# sessions = get_sessions(day=day, fs=0)
#
# dct = {}
# arr = flex_split(sessions, 'ABCD')
# arr = arr.mean('time')
#
# n_trials = arr.sizes['trial']
# mat = np.zeros([n_trials, n_trials])
# for i in range(n_trials - 1):
#     a = arr[i]
#     for j in range(i, n_trials):
#         b = arr[j]
#         score = pearsonr(a, b).statistic
#         mat[i, j] = score
#
# corrs = np.zeros(n_trials - 1)
# for i in range(n_trials - 1):
#     corrs[i] = np.diagonal(mat, i).mean()
#
# fig, ax = plt.subplots()
# Y = corrs[1:]
# X = np.arange(len(Y)) + 1
# ax.plot(X, Y)
# ax.set_xlabel('distance between presentation')
# ax.set_ylabel('correlation')
# plt.show()

"""
Pooling across mice, across events
"""

# day = 5
# sessions = get_sessions(day=day, fs=0)
# dct = {}
# events = schema.events[1:]
# events = [ev for ev in events if ev.name.endswith('_')]
# for ev in events:
#
#     arr = flex_split(sessions, ev)
#     arr = arr.mean('time')
#
#     n_trials = arr.sizes['trial']
#     mat = np.zeros([n_trials, n_trials])
#     for i in range(n_trials - 1):
#         a = arr[i]
#         for j in range(i, n_trials):
#             b = arr[j]
#             score = pearsonr(a, b).statistic
#             mat[i, j] = score
#
#     corrs = np.zeros(n_trials - 1)
#     for i in range(n_trials - 1):
#         corrs[i] = np.diagonal(mat, i).mean()
#     dct[ev.name] = corrs
#
# mat = np.stack([val for val in dct.values()])
# savedir = Path(__file__).parent / 'representational_drift'
# savepath = savedir / f'day{day}_gray.npy'
# np.save(savepath, mat)

savedir = Path(__file__).parent / 'representational_drift'
day0 = np.load(savedir / 'day0_gratings.npy')
day5 = np.load(savedir / 'day5_gratings.npy')

fig, ax = plt.subplots()

data = day0[:, 1:]
mean = data.mean(axis=0)
std = np.std(data, axis=0)
N = np.arange(499, 1, -1)
sem = std / np.sqrt(N)
ymin = mean - sem
ymax = mean + sem
X = np.arange(len(mean)) + 1
ax.plot(X, mean, color='black', label='day 0')
ax.fill_between(X, ymin,  ymax, color='black', alpha=0.2)

data = day5[:, 1:]
mean = data.mean(axis=0)
std = np.std(data, axis=0)
N = np.arange(499, 1, -1)
sem = std / np.sqrt(N)
ymin = mean - sem
ymax = mean + sem
X = np.arange(len(mean)) + 1
ax.plot(X, mean, color='red', label='day 5')
ax.fill_between(X, ymin,  ymax, color='red', alpha=0.2)

ax.legend()
ax.set_xlabel('distance between presentation')
ax.set_ylabel('correlation')
plt.show()

