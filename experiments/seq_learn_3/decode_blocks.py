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
from sklearn.decomposition import PCA

from ca_analysis import *
from main import *
from utils import *

"""
See how well the decoder can tell which block an event came from.
"""

# def decode(event, day) -> np.ndarray:


def make_plot(event, day):

    sessions = get_sessions(day=day, fs=0)
    apply_roi_filter(sessions, 'visual')

    ev = schema.get(event=event)
    arr = flex_split(sessions, ev).mean('time')

    X_train, X_test = [], []
    y_train, y_test = [], []
    for i in range(0, 500, 100):
        block = arr.isel(trial=slice(i, i + 100))
        train_inds = np.random.choice(100, 50, replace=False)
        test_inds = np.setdiff1d(np.arange(100), train_inds)
        X_train.append(block.isel(trial=train_inds))
        y_train.append(np.full(50, i))
        X_test.append(block.isel(trial=test_inds))
        y_test.append(np.full(50, i))

    X_train = xr.concat(X_train, 'trial')
    y_train = np.concatenate(y_train)
    X_test = xr.concat(X_test, 'trial')
    y_test = np.concatenate(y_test)

    clf = SVC()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    # confusion matrix
    cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
    fig, ax = plt.subplots(figsize=(5.75, 5))
    im = ax.imshow(cm, cmap='gray')
    ticks = np.arange(5)
    tick_labels = ['1', '2', '3', '4', '5']
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel('predicted block')
    ax.set_ylabel('true block')
    fig.suptitle(str(event))
    plt.colorbar(im)
    plt.tight_layout(pad=1)
    plt.show()

    return fig


def get_cmat(day, event, kernel='rbf'):

    sessions = get_sessions(day=day, fs=0)
    # apply_roi_filter(sessions, 'visual')
    ev = schema.get(event=event)
    arr = flex_split(sessions, ev)
    arr = arr.isel(time=slice(-4, None))
    arr = arr.mean('time')

    X_train, X_test = [], []
    y_train, y_test = [], []
    for i in range(0, 500, 100):
        block = arr.isel(trial=slice(i, i + 100))
        train_inds = np.random.choice(100, 50, replace=False)
        test_inds = np.setdiff1d(np.arange(100), train_inds)
        X_train.append(block.isel(trial=train_inds))
        y_train.append(np.full(50, i))
        X_test.append(block.isel(trial=test_inds))
        y_test.append(np.full(50, i))

    X_train = xr.concat(X_train, 'trial')
    y_train = np.concatenate(y_train)
    X_test = xr.concat(X_test, 'trial')
    y_test = np.concatenate(y_test)

    clf = SVC(kernel=kernel)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    # confusion matrix
    cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
    cm = normalize_confusion_matrix(cm)
    return cm

#
# day = 0
# event = 'ABCD.A'
#
# sessions = get_sessions(day=day, fs=0)
# apply_roi_filter(sessions, 'visual')
#
# ev = schema.get(event=event)
# arr = flex_split(sessions, ev).mean('time')
# for i in range(arr.shape[0]):
#     trial = arr[i]
#     arr[i] = trial / np.linalg.norm(trial)
#
# X_train, X_test = [], []
# y_train, y_test = [], []
# for i in range(0, 500, 100):
#     block = arr.isel(trial=slice(i, i + 100))
#     train_inds = np.random.choice(100, 50, replace=False)
#     test_inds = np.setdiff1d(np.arange(100), train_inds)
#     X_train.append(block.isel(trial=train_inds))
#     y_train.append(np.full(50, i))
#     X_test.append(block.isel(trial=test_inds))
#     y_test.append(np.full(50, i))
#
# X_train = xr.concat(X_train, 'trial')
# y_train = np.concatenate(y_train)
# X_test = xr.concat(X_test, 'trial')
# y_test = np.concatenate(y_test)
#
# clf = SVC()
# clf.fit(X_train, y_train)
# predictions = clf.predict(X_test)
#
# # confusion matrix
# cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
# fig, ax = plt.subplots(figsize=(5.75, 5))
# im = ax.imshow(cm, cmap='gray')
# ticks = np.arange(5)
# tick_labels = ['1', '2', '3', '4', '5']
# ax.set_xticks(ticks)
# ax.set_xticklabels(tick_labels)
# ax.set_yticks(ticks)
# ax.set_yticklabels(tick_labels)
# ax.set_xlabel('predicted block')
# ax.set_ylabel('true block')
# fig.suptitle(str(event))
# plt.colorbar(im)
# plt.tight_layout(pad=1)
# plt.show()

day = 0
events = schema.events[1:]
events = [ev for ev in events if not ev.name.endswith('_')]
cmats = []
for ev in events:
    cmats.append(get_cmat(day, ev))
cmats = np.stack(cmats)
cm = cmats.mean(axis=0)
fig, ax = plt.subplots(figsize=(5.75, 5))
im = ax.imshow(cm, cmap='gray')
ticks = np.arange(5)
tick_labels = ['1', '2', '3', '4', '5']
ax.set_xticks(ticks)
ax.set_xticklabels(tick_labels)
ax.set_yticks(ticks)
ax.set_yticklabels(tick_labels)
ax.set_xlabel('predicted block')
ax.set_ylabel('true block')
# fig.suptitle(str(event))
plt.colorbar(im)
plt.tight_layout(pad=1)
plt.show()

