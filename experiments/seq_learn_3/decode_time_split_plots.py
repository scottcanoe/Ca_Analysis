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


def load_cmats(event: str, day: int, front: bool)-> np.ndarray:
    datadir = Path(__file__).parent / f"decode_time/split/{event}_data/day{day}"
    datadir = datadir / 'front' if front else datadir / 'back'
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
        if i - 2 >= 0:
            score += 0.25 * row[i - 1]
        if i - 1 >= 0:
            score += 0.5 * row[i - 1]
        score += row[i]
        if i + 1 <= n - 1:
            score += 0.5 * row[i + 1]
        if i + 2 <= n - 1:
            score += 0.25 * row[i + 1]

        out[i] = score / row.sum()
    return out

def norm_mat(cm):
    sums = cm.sum(axis=1)
    for i in range(cm.shape[0]):
        cm[i] = cm[i] / sums[i]
    return cm


event = "ACBD"
day = 0
savedir = Path.home() / "plots/seq_learn_3/decode_time/split"
cmats_front = load_cmats(event, day, True)
cmats_back = load_cmats(event, day, False)

cm_front = cmats_front.mean(axis=0)
cm_front = norm_mat(cm_front)

cm_back = cmats_back.mean(axis=0)
cm_back = norm_mat(cm_back)

pct_front = get_accuracy(cm_front)
pct_back = get_accuracy(cm_back)


X = 1000 * np.arange(cm_front.shape[0]) / 30

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(X, pct_front, color='black', label='front')
ax.plot(X, pct_back, color='red', label='back')
ax.legend()
ax.set_xlabel('time (msec)')
ax.set_ylabel('fraction correct')
if event in ('ABCD', 'ABBD', 'ACBD'):
    ax.set_xticks([0, 250, 500, 750])
    ax.set_xticklabels([r'$\Delta$'] * 4)
ax.set_ylim([0, 1])
fig.tight_layout(pad=3)

plt.show()
# fig.savefig(savedir / f"decode_time_{event}_accuracy.png")


fig, axes = plt.subplots(1, 3, figsize=(8, 3))

ax, mat = axes[0], cm_front
im = ax.imshow(mat, cmap='inferno')
ax.set_title(f"front")
ax.set_xlabel('predicted')
ax.set_ylabel('true')
divider = make_axes_locatable(ax)
ax_cb = divider.append_axes("right", size="5%", pad=0.05)
fig.add_axes(ax_cb)
plt.colorbar(im, cax=ax_cb)
ax_cb.yaxis.tick_right()
ax_cb.yaxis.set_tick_params(labelright=True)

ax, mat = axes[1], cm_back
im = ax.imshow(mat, cmap="inferno")
ax.set_title(f"back")
ax.set_xlabel('predicted')
ax.set_ylabel('true')
divider = make_axes_locatable(ax)
ax_cb = divider.append_axes("right", size="5%", pad=0.05)
fig.add_axes(ax_cb)
plt.colorbar(im, cax=ax_cb)
ax_cb.yaxis.tick_right()
ax_cb.yaxis.set_tick_params(labelright=True)

ax, mat = axes[2], cm_back - cm_front
v_extreme = 0.1
im = ax.imshow(mat, 'coolwarm', vmin=-v_extreme, vmax=v_extreme)
ax.set_title(f"back - front")
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
    ticks = [0, 8, 16, 24]
    ticklabels = ['0', '250', '500', '800']
for ax in axes:
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticklabels)

fig.tight_layout(pad=1)
fig.suptitle(event)
plt.show()


# fig.savefig(savedir / f"decode_time_{event}_cmats.png")
