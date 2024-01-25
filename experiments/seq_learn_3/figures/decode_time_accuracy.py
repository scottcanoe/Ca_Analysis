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


def load_cmats(event: str, day: int) -> np.ndarray:
    datadir = Path(__file__).parent.parent / f"decode_time/{event}_data/day{day}"
    paths = natsorted([p.name for p in datadir.glob('*.npy')])
    paths = [datadir / p for p in paths]
    cmats = np.stack([np.load(p) for p in paths]).astype(float)
    return cmats


# def get_accuracy(cm: np.ndarray) -> np.ndarray:
#
#     n = cm.shape[0]
#     out = np.zeros(n)
#     for i in range(n):
#         row = cm[i]
#         score = 0
#         # if i - 2 >= 0:
#         #     score += 0.25 * row[i - 1]
#         if i - 1 >= 0:
#             score += 0.5 * row[i - 1]
#         score += row[i]
#         if i + 1 <= n - 1:
#             score += 0.5 * row[i + 1]
#         # if i + 2 <= n - 1:
#         #     score += 0.25 * row[i + 1]
#
#         out[i] = score / row.sum()
#     return out


def get_accuracy(cm: np.ndarray) -> np.ndarray:
    return np.diag(cm)

"""
32 timepoints for sequences.
24 timepoints for gray.
"""

def get_diags(event, day):
    cmats = load_cmats(event, day)
    cmats = normalize_confusion_matrices(cmats)
    diags = np.stack([np.diag(cm) for cm in cmats])
    return diags

day = 5
ABCD = get_diags('ABCD', day).mean(axis=1)
ABBD = get_diags('ABBD', day).mean(axis=1)
ACBD = get_diags('ACBD', day).mean(axis=1)
mus = np.concatenate([ABCD, ABBD, ACBD])
mu = mus.mean() * 100
std = mus.std() * 100
print(f'mu={mu}, std={std}')
#
# event = "ABCD"
#
# cmats_0 = load_cmats(event, 0)
# cmats_5 = load_cmats(event, 5)
#
# cmats_0 = normalize_confusion_matrices(cmats_0)
# cmats_5 = normalize_confusion_matrices(cmats_5)
#
# cmats = cmats_0
# diags = np.stack([np.diag(cm) for cm in cmats])
# mus = diags.mean(axis=1)
# mu = mus.mean()
# std = mus.std()


n_iters = 100
n_timepoints = 32
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


