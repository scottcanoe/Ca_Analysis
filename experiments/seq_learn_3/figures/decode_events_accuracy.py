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

from cycler import cycler
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.decomposition import PCA

from ca_analysis import *
from seq_learn_3.main import *
from seq_learn_3.utils import *


def load_cmats(day: int) -> np.ndarray:
    datadir = Path(__file__).parent.parent / f"decode_events/day{day}"
    paths = natsorted([p.name for p in datadir.glob('*.npy')])
    paths = [datadir / p for p in paths]
    cmats = np.stack([np.load(p) for p in paths]).astype(float)
    return cmats


def get_accuracy(cm: np.ndarray) -> np.ndarray:
    return np.diag(cm)


plt.rcParams['font.size'] = 8


savedir = Path(__file__).parent / "figures"
cmats_0 = load_cmats(0)
cmats_5 = load_cmats(5)

cm_0 = cmats_0.mean(axis=0) / 250
cm_5 = cmats_5.mean(axis=0) / 250

pct_0 = get_accuracy(cm_0)
pct_5 = get_accuracy(cm_5)


"""
Compute accuracy with confidence intervals
"""

cmats = load_cmats(0)
cmats = normalize_confusion_matrices(cmats)
# acc = np.stack([np.diag(m) for m in cmats])
# mean = acc.mean()
# low, high = np.percentile(acc, [2.5, 97.5])
# print(f'day 0: mean={mean}, low={low}, high={high}, std={acc.std()}')

diags = np.stack([np.diag(cm) for cm in cmats])
mus = diags.mean(axis=1)
mu = mus.mean()
std = mus.mean()
# mu = diags.mean()
# std = diags.std() / 10
print(f'day 0: mu={mu}, std={std}')

cmats = load_cmats(5)
cmats = normalize_confusion_matrices(cmats)
# acc = np.stack([np.diag(m) for m in cmats])
# mean = acc.mean()
# low, high = np.percentile(acc, [2.5, 97.5])
# print(f'day 5: mean={mean}, low={low}, high={high}, std={acc.std()}')

diags = np.stack([np.diag(cm) for cm in cmats])
# mus = diags.mean(axis=1)
mu = diags.mean()
std = diags.std() / 10
print(f'day 0: mu={mu}, std={std}')

n_iters = 100
n_timepoints = 15
n_trials = 250
cmats = np.zeros([n_iters, n_timepoints, n_timepoints])
for iter in range(n_iters):
    for i in range(n_timepoints):
        preds = np.random.choice(n_timepoints, n_trials, replace=True)
        for p in preds:
            cmats[iter, i, p] = cmats[iter, i, p] + 1

cmats = normalize_confusion_matrices(cmats)
diags = np.stack([np.diag(cm) for cm in cmats])
mus = diags.mean(axis=1)
mu = mus.mean()
std = mus.std()
# mu = diags.mean()
# std = diags.std()
print(f'random: mu={mu}, std={std}')
low, high = np.percentile(mus, [2.5, 97.5])
print(f'random: mean={mu}, low={low}, high={high}')
