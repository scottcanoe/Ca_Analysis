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
import ndindex as nd
import pandas as pd
import matplotlib.colors as mpc
import matplotlib.pyplot as plt
from natsort import natsorted
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
from processing import *
from sklearn.decomposition import PCA

from ca_analysis import *
from main import *
from utils import *

savedir = Path(__file__).parent / "decode_time"
savedir.mkdir(exist_ok=True)
datadir = savedir / 'data'
datadir.mkdir(exist_ok=True)
plotdir = savedir / 'plots'
plotdir.mkdir(exist_ok=True)

day = 0
datadir = Path(__file__).parent / f"decode_time/gray_data/day{day}"
paths = natsorted([p.name for p in datadir.glob('*.npy')])
paths = [datadir / p for p in paths]
mats = np.stack([np.load(p) for p in paths]).astype(float)
cm = mats.mean(axis=0)
# sessions = get_sessions(day=day)
#
# ABCD_ = flex_split(sessions, 'ABCD._')
# ABBD_ = flex_split(sessions, 'ABBD._')
# ACBD_ = flex_split(sessions, 'ACBD._')
# data = xr.concat([ABCD_, ABBD_, ACBD_], 'trial')

#
# group_1_trial_inds = np.arange(0, 1200, 2)
# group_2_trial_inds = group_1_trial_inds + 1
#
# group_1 = data.isel(trial=group_1_trial_inds)
# group_2 = data.isel(trial=group_2_trial_inds)
#
#
# X, y = [], []
# for i in range(group_1.sizes['time']):
#     chunk = group_1.isel(time=i)
#     n_trials = chunk.sizes['trial']
#     X.append(chunk)
#     y.append(np.full(n_trials, i))
# X_train = xr.concat(X, 'trial')
# y_train = np.hstack(y)
#
# X, y = [], []
# for i in range(group_2.sizes['time']):
#     chunk = group_2.isel(time=i)
#     n_trials = chunk.sizes['trial']
#     X.append(chunk)
#     y.append(np.full(n_trials, i))
# X_test = xr.concat(X, 'trial')
# y_test = np.hstack(y)
#
#
#
# clf = SVC(random_state=0)
# clf.fit(X_train, y_train)
# predictions = clf.predict(X_test)
#
# cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
#
fig, ax = plt.subplots()
ax.imshow(cm, cmap='gray')
ax.set_title(f"gray day {day}")
ax.set_xlabel('predicted')
ax.set_ylabel('true')
fig.tight_layout(pad=2)
plt.show()
#
# # fig.savefig(f'plots/cm_{event}_day{day}')
# path = plotdir / f"{fname}.png"
# fig.savefig(path)
#
# path = datadir / f"{fname}.npy"
# np.save(path, cm)
#
