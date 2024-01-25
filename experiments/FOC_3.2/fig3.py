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
from scipy.stats import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from statsmodels.distributions.empirical_distribution import ECDF
import xarray as xr

from ca_analysis import *
from ca_analysis.stats import *

from main import *
from processing import *


transition_to_code = {
    "AC": 1,
    "BC": 2,
    "BD": 3,
    "CD": 4,
    "CE": 5,
    "DE": 6,
    "DA": 7,
    "EA": 8,
    "EB": 9,
    "AB": 10,
}
code_to_transition = {val: key for key, val in transition_to_code.items()}


def balance_data(X, y):
    y = y.flatten()
    in_play = np.unique(y)
    n_keep = np.min([y[y == val].sum() for val in in_play])
    X_lst, y_lst = [], []
    for val in in_play:
        inds = argwhere(y == val)
        inds = shuffled(inds)
        inds = inds[:n_keep]
        X_lst.append(X[inds])
        y_lst.append(y[inds])
    X_out = np.vstack(X_lst)
    y_out = np.hstack(y_lst)
    return X_out, y_out


def reshape(*arrays):
    out = []
    for arr in arrays:
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        out.append(arr)
    return out


def pool_histogram_scores(
    ses: Union[Session, List[Session]],
    block: IndexLike = slice(None),
    events: Sequence = ("A", "B", "C", "D", "E"),
    score_fn: Callable = lambda pre, post: post - pre,
) -> Tuple[Mapping, Mapping]:

    ses = [ses] if isinstance(ses, Session) else ses
    T, pairs = get_transitions(), get_transition_pairs()
    for tr in itertransitions(ses):
        pre, post = tr.data
        if block is not None or block != slice(None):
            block = [block] if is_int(block) else block
            block_ids = np.arange(pre.block_id.max().item() + 1)
            valid_blocks = block_ids[block]
            tf = np.in1d(pre.block_id, valid_blocks)
            pre, post = pre[tf], post[tf]

        pre, post = pre.mean("time"), post.mean("time")
        pre, post = pre.mean("trial"), post.mean("trial")
        tr.scores = score_fn(pre, post)
        T[tr.name].scores.append(tr.scores)

    for tr in T.values():
        tr.scores = pool_flat(tr.scores)

    for ev in events:
        for s in ses:
            p = s.data.get_pair(ev, "post")
            high, low = p.high, p.low
            p.scores = low.scores - high.scores
            pairs[ev].scores.append(p.scores)
        pairs[ev].scores = pool_flat(pairs[ev].scores)

    return T, pairs



def compute_scores(df: pd.DataFrame, event_ids: np.array) -> np.array:
    scores = np.zeros(len(event_ids))
    for i, ev in enumerate(event_ids):
        subdf = df[df.event_id == ev]
        scores[i] = subdf["spikes"].sum()
    return scores / scores.sum()


def permute_df(df: pd.DataFrame):
    out = df.copy()
    out["event_id"] = shuffled(df["event_id"])
    return out

def generate_confusion_matrix(s: Session, use_codes: ArrayLike) -> np.ndarray:


    T, pairs = get_transitions(), get_transition_pairs()
    all_codes = []
    all_scores = []
    trial_average = False
    time_slice = slice(2, None)

    spikes = s.data.get('spikes')
    means = spikes.mean('time')
    cutoff = np.percentile(means, 50)
    keep_rois = argwhere(means > cutoff)
    for tr in itertransitions(s):
        lst = [tr.data[0], tr.data[1]]
        # lst = [arr.isel(roi=keep_rois) for arr in lst]
        # optionally don't use all frames
        # lst = [arr[np.isin(arr.block_id, [3, 4])] for arr in lst]
        if time_slice is not None:
            lst = [arr.isel(time=time_slice) for arr in lst]

        # average over frames in each presentation, and put trials in
        # zeroth dimension.
        lst = [arr.mean("time").transpose("trial", "roi") for arr in lst]

        # optionally average over trials
        if trial_average:
            lst = [arr.mean("trial") for arr in lst]

        pre, post = lst
        scores = post
        # scores = pre
        # scores = post - pre
        codes = np.full(post.sizes["trial"], transition_to_code[tr.name])

        tr.scores = scores
        T[tr.name].scores.append(scores)
        all_scores.append(scores)
        all_codes.append(codes)

    X = np.vstack(all_scores)
    y = np.hstack(all_codes)
    # X, y = balance_data(X, y)

    tf = np.isin(y, use_codes)
    X, y = X[tf], y[tf]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.5, shuffle=True,
    )
    X_train, y_train = balance_data(X_train, y_train)
    X_test, y_test = balance_data(X_test, y_test)
    y_train, y_test = reshape(y_train, y_test)

    clf = SVC(random_state=0)
    clf.fit(
        X_train,
        y_train,
    )
    predictions = clf.predict(X_test)
    cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
    return cm, clf


if __name__ == "__main__":

    logger = get_logger(logging.INFO)

    sessions = get_sessions()
    if sessions[0].data.transitions["AC"].data[0] is None:
        load_spikes(sessions, block=slice(None))

    use_codes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # use_codes = [1, 2, 3, 4, 9, 10]
    info = [generate_confusion_matrix(s, use_codes) for s in sessions]
    mats = [tup[0] for tup in info]
    clf = info[0][1]
    mats = np.stack(mats)
    cm = mats.sum(axis=0)
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    im = ax.imshow(cm, cmap="inferno")
    labels = [
        "AC",
        "BC",
        "BD",
        "CD",
        "CE",
        "DE",
        "DA",
        "EA",
        "EB",
        "AB",
    ]
    ticks = np.arange(len(use_codes))
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    cbar = plt.colorbar(im)
    plt.show()
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm,
    #                               display_labels=clf.classes_)
    # disp.display_labels = [code_to_transition[code] for code in use_codes]
    # disp.plot()
    # plt.show()
    PLOTDIR = Path.home() / "plots/cosyne"
    fig.savefig(PLOTDIR / "cmat.pdf")
    fig.savefig(PLOTDIR / "cmat.eps")