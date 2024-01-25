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




def pool_flat(lst):
    return np.hstack([np.asarray(mat).reshape(-1) for mat in lst])


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



if __name__ == "__main__":


    sessions = get_sessions("3.2")

    s = sessions[1]
    schema = s.events.schema
    events = [schema.get(event=ev_name) for ev_name
              in ["gray", "A", "B", "C", "D", "E", "F"]]
    n_rois = s.data.n_cells

    # split master spike array by event
    spikes = {}
    for i in range(1, 8):
        ev = schema.get(event=i)
        spikes[ev.name] = s.data.split(ev.name, blocks=1)

    # split gray into median number of trials.
    med_trials = int(np.median([arr.sizes["trial"] for arr in spikes.values()]))
    trial_len = int(np.median([arr.sizes["time"] for arr in spikes.values()]))
    gray = spikes['gray'].transpose("trial", "time", "roi").squeeze()
    n_trials = gray.sizes["time"] // trial_len
    n_timepoints = n_trials * trial_len
    mat = gray.data[:n_timepoints]
    mat = mat.reshape(n_trials, trial_len, gray.sizes["roi"])
    if n_trials < med_trials:
        inds = np.random.choice(n_trials, replace=True)
        mat = mat[inds]
    else:
        mat = mat[-med_trials:]
    gray = xr.DataArray(mat, dims=("trial", "time", "roi"), coords=gray.coords)
    spikes["gray"] = gray

    # subsample "F" to get median number of trials.
    F = spikes["F"]
    inds = np.random.choice(F.sizes["trial"], size=med_trials, replace=False)
    spikes["F"] = F.isel(trial=inds)

    # average over time for all spike arrays
    for key, arr in spikes.items():
        spikes[key] = arr.mean("time")


    all_info = []
    alpha = 0.05
    for i in range(n_rois):
        obj = SimpleNamespace(roi=i)
        arrays = [spikes[ev.name].isel(roi=i) for ev in events]
        obj.is_normal = all([normaltest(arr).pvalue <= alpha for arr in arrays])
        if obj.is_normal:
            obj.ftest = f_oneway(*arrays)
            if obj.ftest.pvalue <= alpha:
                # compute scores
                scores = np.array([arr.sum() for arr in arrays], dtype=float)
                scores = scores / scores.sum()
                obj.scores = xr.DataArray(scores, dims=('event',), coords=dict(event=events))

        all_info.append(obj)

    info = [obj for obj in all_info if obj.is_normal and obj.ftest.pvalue < alpha]
    maxs = np.array([obj.scores.max().item() for obj in info])
    thresh = 2 * 1 / len(events)
    good = []
    for obj in info:
        if np.any(obj.scores >= thresh):
            loc = np.argmax(obj.scores.data)
            obj.event = events[loc]
            obj.max = obj.scores[loc].item()

            good.append(obj)
    n_all = len(all_info)
    n_anova = len(info)
    n_thresh = len(good)

    for obj in good:
        print(f"{obj.event.name}: {obj.max}")