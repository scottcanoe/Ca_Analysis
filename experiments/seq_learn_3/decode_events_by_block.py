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



day = 0

sessions = get_sessions(day=day, fs=0)
apply_roi_filter(sessions, 'visual')
accuracy = []
for block in range(1):
    start = block * 100
    stop = start + 100
    n_trials = 100
    train_ids = np.random.choice(n_trials, n_trials // 2, replace=False)
    test_ids = np.setdiff1d(np.arange(n_trials, dtype=int), train_ids)

    X_train, X_test = [], []
    y_train, y_test = [], []
    events = schema.events[1:]
    for ev in events:
        arr = flex_split(sessions, schema.get(event=ev))
        arr = arr.isel(trial=slice(start, stop))
        # arr = arr.mean('time')
        # arr = arr.isel(time=7)
        arr = arr.isel(time=slice(4, 8)).mean('time')
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
    fig, ax = plt.subplots(figsize=(10.5, 8))
    im = ax.imshow(cm, cmap='gray')
    ticks = np.arange(len(events))
    tick_labels = [ev.name for ev in events]
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, rotation=-65)  # , ha='left')
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel('predicted')
    ax.set_ylabel('true')
    fig.suptitle(f'day {day}, block {block}')
    plt.colorbar(im)
    plt.tight_layout(pad=1)
    plt.show()
    savedir = Path.home() / 'plots/seq_learn_3/decode_events_by_block'
    savedir.mkdir(exist_ok=True)
    path = savedir / f'block_{block}.png'
    fig.savefig(path)
    # acc = get_soft_decoder_accuracy(cm)
    acc = np.diag(cm)
    accuracy.append(acc)

acc = np.stack(accuracy)

# fig, ax = plt.subplots()
# for block in range(5):
#     ax.plot(acc[block], label=str(block))
# ax.set_xticks(ticks)
# ax.set_xticklabels(tick_labels, rotation=-65)
# ax.legend()
# plt.show()
