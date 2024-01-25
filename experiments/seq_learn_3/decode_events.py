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


def run_decoder(day: int) -> None:


    sessions = get_sessions(day=day, fs=0)
    apply_roi_filter(sessions, 'visual')

    n_trials = 500
    train_ids = np.random.choice(n_trials, n_trials // 2, replace=False)
    test_ids = np.setdiff1d(np.arange(n_trials, dtype=int), train_ids)

    X_train, X_test = [], []
    y_train, y_test = [], []
    events = schema.events[1:]
    for ev in events:
        arr = flex_split(sessions, schema.get(event=ev))
        # arr = arr.isel(time=7)
        arr = arr.isel(time=slice(-4, None)).mean('time')
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
    cm = normalize_confusion_matrix(cm)

    labels = [ev.name for ev in events]

    ax = sns.heatmap(cm, xticklabels=labels, yticklabels=labels, vmin=0, vmax=1, square=True)
    ax.set_title(f'day {day}')
    fig = ax.figure
    fig.tight_layout(pad=1.2)
    plt.show()

    savedir = Path(__file__).parent / "decode_events"
    np.save(savedir / f'day{day}.npy', cm)
    fig.savefig(savedir / f'day{day}.png')


run_decoder(5)
# run_decoder(5)
