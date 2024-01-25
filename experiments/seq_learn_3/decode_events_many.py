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

import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

import xarray as xr

from ca_analysis import *


from ca_analysis import *
from main import *
from utils import *



n_iters = 100

for day in (0, 5):
    sessions = get_sessions(day=day, fs=0)
    apply_roi_filter(sessions, 'all')

    for it in range(n_iters):
        print(it)
        n_trials = 500
        train_ids = np.random.choice(n_trials, n_trials // 2, replace=False)
        test_ids = np.setdiff1d(np.arange(n_trials, dtype=int), train_ids)

        X_train, X_test = [], []
        y_train, y_test = [], []
        events = schema.events[1:]
        for ev in events:
            arr = flex_split(sessions, schema.get(event=ev))
            arr = arr.isel(time=slice(2, None)).mean('time')
            train = arr.isel(trial=train_ids)
            X_train.append(train)
            y_train.append(np.full(train.sizes['trial'], ev.id))

            test = arr.isel(trial=test_ids)
            X_test.append(test)
            y_test.append(np.full(test.sizes['trial'], ev.id))

        X_train = xr.concat(X_train, 'trial')
        y_train = np.concatenate(y_train)
        X_test = xr.concat(X_test, 'trial')
        y_test = np.concatenate(y_test)

        clf = SVC()
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)

        # confusion matrix
        cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
        savedir = Path(__file__).parent / f'decode_events/day{day}'
        fname = savedir / f'{it}.npy'
        np.save(fname, cm)

