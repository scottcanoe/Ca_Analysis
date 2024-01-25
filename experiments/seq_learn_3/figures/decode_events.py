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
from experiments.seq_learn_3.main import *
from experiments.seq_learn_3.utils import *


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

# with h5py.File('../data/decode_events.h5', 'w') as f:
#     cm = cmats_0.astype(int)
#     f.create_dataset('day_0', data=cm)
#     cm = cmats_5.astype(int)
#     f.create_dataset('day_5', data=cm)



cm_0 = cmats_0.mean(axis=0) / 250
cm_5 = cmats_5.mean(axis=0) / 250

pct_0 = get_accuracy(cm_0)
pct_5 = get_accuracy(cm_5)


#-------------------------------------------------------------------------------
# Compute accuracy

fig, ax = plt.subplots(figsize=(3.5, 1.75))
X1 = np.arange(15)
X2 = X1 + 0.5
ax.bar(X1, pct_0, color='black', label='day 0', width=0.4)
ax.bar(X2, pct_5, color='red', label='day 5', width=0.4)
ax.legend(frameon=False, loc='upper right')

ax.set_xlabel('time')
ax.set_ylabel('fraction correct')
labels = [
    'ABCD.A',
    'ABCD.B',
    'ABCD.C',
    'ABCD.D',
    'ABCD._',
    'ABBD.A',
    'ABBD.B1',
    'ABBD.B2',
    'ABBD.D',
    'ABBD._',
    'ACBD.A',
    'ACBD.C',
    'ACBD.B',
    'ACBD.D',
    'ACBD._',
]
ax.set_xticks(X1 + 0.25)
ax.set_xticklabels(labels, rotation=60)
# ax.set_xticklabels([r'$\Delta$'] * 4)
# ax.set_xlim([0, 1000])

ax.set_ylim([0.4, 1])
fig.tight_layout(pad=0.8)

# plt.show()
fig.savefig(savedir / f"decode_events_accuracy.eps")

#-------------------------------------------------------------------------------

plt.rcParams['font.size'] = 6

fig, axes = plt.subplots(1, 3, figsize=(7, 2.25))

ax, mat = axes[0], cm_0
im = ax.imshow(mat, cmap='inferno')
ax.set_title(f"day 0")
ax.set_xlabel('predicted')
ax.set_ylabel('true')
divider = make_axes_locatable(ax)
ax_cb = divider.append_axes("right", size="5%", pad=0.05)
fig.add_axes(ax_cb)
plt.colorbar(im, cax=ax_cb)
ax_cb.yaxis.tick_right()
ax_cb.yaxis.set_tick_params(labelright=True)

ax, mat = axes[1], cm_5
im = ax.imshow(mat, cmap="inferno")
ax.set_title(f"day 5")
ax.set_xlabel('predicted')
ax.set_ylabel('true')
divider = make_axes_locatable(ax)
ax_cb = divider.append_axes("right", size="5%", pad=0.05)
fig.add_axes(ax_cb)
plt.colorbar(im, cax=ax_cb)
ax_cb.yaxis.tick_right()
ax_cb.yaxis.set_tick_params(labelright=True)

ax, mat = axes[2], cm_5 - cm_0
v_extreme = 0.1
im = ax.imshow(mat, 'coolwarm', vmin=-v_extreme, vmax=v_extreme)
ax.set_title(f"day 5 - day 0")
ax.set_xlabel('predicted')
ax.set_ylabel('true')
divider = make_axes_locatable(ax)
ax_cb = divider.append_axes("right", size="5%", pad=0.05)
fig.add_axes(ax_cb)
cbar = plt.colorbar(im, cax=ax_cb)
ax_cb.yaxis.tick_right()
ax_cb.yaxis.set_tick_params(labelright=True)

# if event in ("ABCD", "ABBD", "ACBD"):
#     ticks = [0, 8, 16, 24]
#     ticklabels = [r'$\Delta$'] * 4
# else:
#     ticks = [0, 6, 12, 18, 23]
#     ticklabels = ['0', '200', '400', '600', '800']
ticks = np.arange(len(labels))
for ax in axes:
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)

fig.tight_layout(pad=0.7)
# plt.show()
fig.savefig(savedir / f"decode_events_cm.eps")



# Compute accuracy with confidence intervals

cmats = load_cmats(0)
cmats = normalize_confusion_matrices(cmats)
acc = np.stack([np.diag(m) for m in cmats])
mean = acc.mean()
low, high = np.percentile(acc, [2.5, 97.5])
print(f'day 0: mean={mean}, low={low}, high={high}')

cmats = load_cmats(5)
cmats = normalize_confusion_matrices(cmats)
acc = np.stack([np.diag(m) for m in cmats])
mean = acc.mean()
low, high = np.percentile(acc, [2.5, 97.5])
print(f'day 5: mean={mean}, low={low}, high={high}')

