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


plt.rcParams['font.size'] = 8


event = "gray"

cmats_0 = load_cmats(event, 0)
cmats_5 = load_cmats(event, 5)

cm_0 = cmats_0.mean(axis=0) / 600
cm_5 = cmats_5.mean(axis=0) / 600

pct_0 = get_accuracy(cm_0)
pct_5 = get_accuracy(cm_5)


X = 1000 * np.arange(cm_0.shape[0]) / 30

#-------------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(2.75, 1.5))

cmats = cmats_0
cmats = normalize_confusion_matrices(cmats)
acc = np.stack([get_accuracy(m) for m in cmats])
low = np.percentile(acc, 2.5, axis=0)
high = np.percentile(acc, 97.5, axis=0)
ax.fill_between(X, low, high, color='gray', alpha=0.2)

cmats = cmats_5
cmats = normalize_confusion_matrices(cmats)
acc = np.stack([get_accuracy(m) for m in cmats])
low = np.percentile(acc, 2.5, axis=0)
high = np.percentile(acc, 97.5, axis=0)
ax.fill_between(X, low, high, color='red', alpha=0.2)


ax.plot(X, pct_0, color='black', label='day 0')
ax.plot(X, pct_5, color='red', label='day 5')
ax.legend(frameon=False, loc='upper right')
ax.set_title(event)
ax.set_xlabel('time (msec)')
ax.set_ylabel('fraction correct')
if event in ('ABCD', 'ABBD', 'ACBD'):
    ax.set_xticks([0, 250, 500, 750])
    ax.set_xticklabels([r'$\Delta$'] * 4)
    ax.set_xlim([0, 1000])
else:
    ax.set_xticks([0, 200, 400, 600, 800])
    ax.set_xlim([0, 766])
ax.set_ylim([0, 1])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

fig.tight_layout(pad=0.8)

plt.show()
fig.savefig(f"figures/decode_time_{event}_accuracy.eps")

#-------------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(7, 2.25))

ax, mat = axes[0], cm_0
im = ax.imshow(mat, cmap='inferno', vmin=0.0, vmax=0.5)
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
im = ax.imshow(mat, cmap="inferno", vmin=0.0, vmax=0.5)
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

if event in ("ABCD", "ABBD", "ACBD"):
    ticks = [0, 8, 16, 24]
    ticklabels = [r'$\Delta$'] * 4
else:
    ticks = [0, 6, 12, 18, 23]
    ticklabels = ['0', '200', '400', '600', '800']
for ax in axes:
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticklabels)

fig.tight_layout(pad=1)
fig.suptitle(event)
plt.show()

fig.savefig(f"figures/decode_time_{event}_cm.eps")
