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


savedir = Path(__file__).parent / 'decode_events/separate_blocks'


def run(day, train_block_id, test_block_id):

    npy_dir = Path(__file__).parent / 'decode_events/separate_blocks'
    npy_dir.mkdir(exist_ok=True, parents=True)
    npy_path = npy_dir / f'day_{day}_train_{train_block_id}_test_{test_block_id}.npy'

    sessions = get_sessions(day=day, fs=0)
    apply_roi_filter(sessions, 'all')

    train_ids = np.arange(100) + 100 * train_block_id
    test_ids = np.arange(100) + 100 * test_block_id

    X_train, X_test = [], []
    y_train, y_test = [], []
    events = schema.events[1:]
    for ev in events:
        arr = flex_split(sessions, schema.get(event=ev))
        # arr = arr.mean('time')
        arr = arr.isel(time=slice(4, None)).mean('time')
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
    np.save(npy_path, cm)

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
    fig.suptitle(f'day {day}, train block {train_block_id}, test block {test_block_id}')
    plt.colorbar(im)
    plt.tight_layout(pad=1)
    plt.show()

    plot_dir = Path.home() / 'plots/seq_learn_3/separate_blocks'
    plot_dir.mkdir(exist_ok=True, parents=True)
    plot_path = plot_dir / f'day_{day}_train_{train_block_id}_test_{test_block_id}.png'
    fig.savefig(plot_path)


def heatmap(cm):
    events = schema.events[1:]
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
    # fig.suptitle(f'day {day}, train block {train_block_id}, test block {test_block_id}')
    plt.colorbar(im)
    plt.tight_layout(pad=1)
    plt.show()
    return fig


def load_cmat(day, train_block_id , test_block_id):
    npy_dir = Path(__file__).parent / 'decode_events/separate_blocks'
    npy_path = npy_dir / f'day_{day}_train_{train_block_id}_test_{test_block_id}.npy'
    return np.load(npy_path)


def load_cmats(day):
    cmats = []
    for train in range(5):
        for test in range(5):
            if train == test:
                continue
            if test - train != 1:
                continue
            cmats.append(load_cmat(day, train, test))
    return np.stack(cmats)


# for day in (0, 5):
#     for train in range(5):
#         for test in range(5):
#             if train == test:
#                 continue
#             run(day, train, test)

cmats_0 = load_cmats(0).mean(axis=0)
cmats_5 = load_cmats(5).mean(axis=0)

fig_0 = heatmap(cmats_0)
fig_0.suptitle('day 0')

fig_5 = heatmap(cmats_5)
fig_5.suptitle('day 5')

cm = cmats_5 - cmats_0
fig, ax = plt.subplots(figsize=(10.5, 8))
events = schema.events[1:]
vmax = max(abs(cm.min()), abs(cm.max()))
im = ax.imshow(cm, cmap='coolwarm', vmin=-vmax, vmax=vmax)
ticks = np.arange(len(events))
tick_labels = [ev.name for ev in events]
ax.set_xticks(ticks)
ax.set_xticklabels(tick_labels, rotation=-65)  # , ha='left')
ax.set_yticks(ticks)
ax.set_yticklabels(tick_labels)
ax.set_xlabel('predicted')
ax.set_ylabel('true')
plt.colorbar(im)
plt.tight_layout(pad=1)
plt.show()
