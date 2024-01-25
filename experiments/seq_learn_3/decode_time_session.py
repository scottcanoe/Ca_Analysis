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
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
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


sessions_0 = get_sessions(day=0)
sessions_5 = get_sessions(day=5)

sessions = sessions_5[0:1]

event = 'ABCD'
drop_gray = False

data = flex_split(sessions, event, drop_gray=drop_gray)
group_1_trial_inds = np.arange(0, 400, 2)
group_2_trial_inds = group_1_trial_inds + 1
group_1 = data.isel(trial=group_1_trial_inds)
group_2 = data.isel(trial=group_2_trial_inds)

X, y = [], []
for i in range(group_1.sizes['time']):
    chunk = group_1.isel(time=i)
    n_trials = chunk.sizes['trial']
    X.append(chunk)
    y.append(np.full(n_trials, i))
X_train = xr.concat(X, 'trial')
y_train = np.hstack(y)

X, y = [], []
for i in range(group_2.sizes['time']):
    chunk = group_2.isel(time=i)
    n_trials = chunk.sizes['trial']
    X.append(chunk)
    y.append(np.full(n_trials, i))
X_test = xr.concat(X, 'trial')
y_test = np.hstack(y)

clf = SVC()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

cm = confusion_matrix(y_test, predictions, labels=clf.classes_)

fig, ax = plt.subplots()
im = ax.imshow(cm, cmap='inferno')
ax.set_title(event)
ax.set_xlabel('predicted')
ax.set_ylabel('true')
divider = make_axes_locatable(ax)
ax_cb = divider.append_axes("right", size="5%", pad=0.05)
fig.add_axes(ax_cb)
plt.colorbar(im, cax=ax_cb)
ax_cb.yaxis.tick_right()
ax_cb.yaxis.set_tick_params(labelright=True)

if event in ("ABCD", "ABBD", "ACBD"):
    ticks = [0, 8, 16, 24]
    ticklabels = [r'$\Delta$'] * 4
    if not drop_gray:
        ticks.append(32)
        ticklabels.append('-')
else:
    ticks = [0, 8, 16, 24]
    ticklabels = ['0', '250', '500', '800']

ax.set_xticks(ticks)
ax.set_xticklabels(ticklabels)
ax.set_yticks(ticks)
ax.set_yticklabels(ticklabels)

fig.tight_layout(pad=1)
plt.show()

