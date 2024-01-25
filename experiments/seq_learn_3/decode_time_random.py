import copy
import datetime
import logging
import dataclasses
from numbers import Number
import os
from pathlib import Path
import shutil
import time
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
from natsort import natsorted
import ndindex as nd
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.colors as mpc
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import *
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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from processing import *
from sklearn.decomposition import PCA

from ca_analysis import *
from main import *
from utils import *


def load_cmats(event: str, day: int) -> np.ndarray:
    datadir = Path(__file__).parent / f"decode_time/{event}_data/day{day}"
    paths = natsorted([p.name for p in datadir.glob('*.npy')])
    paths = [datadir / p for p in paths]
    cmats = np.stack([np.load(p) for p in paths]).astype(float)
    return cmats


def get_accuracy(cm: np.ndarray) -> np.ndarray:

    n = cm.shape[0]
    out = np.zeros(n)
    for i in range(n):
        row = cm[i]
        score = 0
        # if i - 2 >= 0:
        #     score += 0.25 * row[i - 1]
        if i - 1 >= 0:
            score += 0.5 * row[i - 1]
        score += row[i]
        if i + 1 <= n - 1:
            score += 0.5 * row[i + 1]
        # if i + 2 <= n - 1:
        #     score += 0.25 * row[i + 1]
        out[i] = score / row.sum()
    return out



event = "gray"
savedir = Path(__file__).parent / "figures/figures"
cmats_0 = load_cmats(event, 0)
cmats_5 = load_cmats(event, 5)

cm_0 = cmats_0.mean(axis=0) / 600
cm_5 = cmats_5.mean(axis=0) / 600

pct_0 = get_accuracy(cm_0)
pct_5 = get_accuracy(cm_5)

"""
For gray: 24 timepoints

make random confusion matrices (600 trials)
"""
n_iters = 100
n_timepoints = 24
n_trials = 600
all_middle = np.array([])
all_edges = np.array([])
for k in range(10):
    cmats = np.zeros([n_iters, n_timepoints, n_timepoints])
    for iter in range(n_iters):
        for i in range(n_timepoints):
            preds = np.random.choice(n_timepoints, n_trials, replace=True)
            for p in preds:
                cmats[iter, i, p] = cmats[iter, i, p] + 1

    cm = cmats.mean(axis=0) / 600
    pct = get_accuracy(cm)
    edges = np.array([pct[0], pct[-1]])
    middle = pct[1:-1]
    all_middle = np.r_[all_middle, middle]
    all_edges = np.r_[all_edges, edges]


mean = all_middle.mean()
low, high = np.percentile(all_middle, [2.5, 97.5])
print(f'middle: mean={mean}, low={low}, high={high}')

mean = all_edges.mean()
low, high = np.percentile(all_edges, [2.5, 97.5])
print(f'edges: mean={mean}, low={low}, high={high}')

# fig, ax = plt.subplots()
# ax.plot(pct, color='black')
# plt.show()
