import copy
import datetime
import logging
import dataclasses
from numbers import Number
import os
from pathlib import Path
import shutil
import time
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
from processing import *
from sklearn.decomposition import PCA

from ca_analysis import *
from main import *
from utils import *



# set up parameters
n_iters = 100


# do decoding
for day in (0, 5):

    # load data
    sessions = get_sessions(day=day)
    ABCD_ = flex_split(sessions, 'ABCD._')
    ABBD_ = flex_split(sessions, 'ABBD._')
    ACBD_ = flex_split(sessions, 'ACBD._')
    data = xr.concat([ABCD_, ABBD_, ACBD_], 'trial')
    for front in (True, False):
        savedir = Path(__file__).parent / f'decode_time/split/gray_data/day{day}'
        savedir = savedir / 'front' if front else savedir / 'back'
        savedir.mkdir(exist_ok=True, parents=True)

        for iter in range(n_iters):
            t_start = clock()

            # split groups
            if front:
                all_trial_inds = np.arange(200)
                group_1_trial_inds = np.random.choice(all_trial_inds, 100, replace=False)
                group_2_trial_inds = np.setdiff1d(all_trial_inds, group_1_trial_inds)
                group_1 = data.isel(trial=group_1_trial_inds)
                group_2 = data.isel(trial=group_2_trial_inds)
            else:
                all_trial_inds = np.arange(1300, 1500)
                group_1_trial_inds = np.random.choice(all_trial_inds, 100, replace=False)
                group_2_trial_inds = np.setdiff1d(all_trial_inds, group_1_trial_inds)
                group_1 = data.isel(trial=group_1_trial_inds)
                group_2 = data.isel(trial=group_2_trial_inds)

            # - training data
            X, y = [], []
            for i in range(group_1.sizes['time']):
                chunk = group_1.isel(time=i)
                n_trials = chunk.sizes['trial']
                X.append(chunk)
                y.append(np.full(n_trials, i))
            X_train = xr.concat(X, 'trial')
            y_train = np.hstack(y)

            # - testing data
            X, y = [], []
            for i in range(group_2.sizes['time']):
                chunk = group_2.isel(time=i)
                n_trials = chunk.sizes['trial']
                X.append(chunk)
                y.append(np.full(n_trials, i))
            X_test = xr.concat(X, 'trial')
            y_test = np.hstack(y)

            # train model
            clf = SVC(random_state=0)
            clf.fit(X_train, y_train)
            predictions = clf.predict(X_test)

            # save confusion matrix
            cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
            np.save(savedir / f'{iter}.npy', cm)

            t_tot = clock() - t_start
            print(f'day {day}, iter {iter} completed in {format_timedelta(t_tot)}')
