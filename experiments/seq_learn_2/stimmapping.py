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

import pandas as pd
import matplotlib.colors as mpc
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import *
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



def stimmap():

    sessions = get_sessions(day=5)

    s = sessions[0]

    schema = s.events.schema
    events = [schema.get(event=ev_name) for ev_name in ("A", "B", "C", "D", "gray")]
    n_cells = s.data.n_cells

    # split master spike array by event
    spikes = {}
    for i in range(1, 6):
        ev = schema.get(event=i)
        spikes[ev.name] = s.data.split(ev.name)

    # split gray into median number of trials.
    med_trials = int(np.median([arr.sizes["trial"] for arr in spikes.values()]))
    for name, arr in spikes.items():
        n_trials = arr.sizes["trial"]
        if n_trials == med_trials:
            continue
        arr = arr.transpose("trial", "time", "roi").squeeze()
        replace = n_trials < med_trials
        inds = np.random.choice(n_trials, size=(med_trials,), replace=replace)
        # inds = np.random.choice(med_trials, replace=arr.sizes["trial"] < med_trials)
        # mat = arr.data[inds]
        new_arr = arr.isel(trial=inds)
        spikes[name] = new_arr

    # average over time for all spike arrays
    for key, arr in spikes.items():
        spikes[key] = arr.mean("time")

    all_info = []
    alpha = 0.05
    for i in range(n_cells):
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

    print('finished')


def compute_visual_selectivity(s: Session):
    schema = s.events.schema
    spikes = s.data.get("spikes")
    rois = spikes.coords["roi"]
    n_cells = len(rois)

    frames = s.events.tables["frame"]
    gray_inds, stim_inds = [], []
    for ev in schema.events:
        df = frames[frames['event'] == ev.id]
        locs = np.array(df.index)
        if ev.stimulus == 0:
            gray_inds.append(locs)
        else:
            stim_inds.append(locs)

    gray_inds, stim_inds = np.concatenate(gray_inds), np.concatenate(stim_inds)

    major_chunk = 8
    minor_chunk = 4

    inds = gray_inds
    n, rem = divmod(len(inds), major_chunk)
    if rem > 0:
        inds = inds[:-rem]
    inds = inds.reshape(n, major_chunk)
    inds = inds[:, :minor_chunk]
    samples = []
    for j in range(inds.shape[0]):
        obj = spikes.isel(time=inds[j]).mean("time").data
        samples.append(obj)
    mat = np.vstack(samples).T
    gray_data = xr.DataArray(mat, dims=('roi', 'time'), coords=dict(roi=rois))

    inds = stim_inds
    n, rem = divmod(len(inds), major_chunk)
    if rem > 0:
        inds = inds[:-rem]
    inds = inds.reshape(n, major_chunk)
    inds = inds[:, :minor_chunk]
    samples = []
    for j in range(inds.shape[0]):
        obj = spikes.isel(time=inds[j]).mean("time").data
        samples.append(obj)
    mat = np.vstack(samples).T
    stim_data = xr.DataArray(mat, dims=('roi', 'time'), coords=dict(roi=rois))

    all_info = []
    alpha = 0.05
    for i in range(len(rois)):
        roi_id = rois[i].item()
        obj = SimpleNamespace(roi=i)
        stim_array = stim_data.sel(roi=roi_id)
        gray_array = gray_data.sel(roi=roi_id)
        stat = ks_2samp(stim_array, gray_array)
        obj.pvalue = stat.pvalue
        obj.significant = stat.pvalue < alpha
        obj.visual = False
        if obj.significant:
            obj.visual = gray_array.mean().item() < stim_array.mean().item()
        all_info.append(obj)

    significant = [info for info in all_info if info.significant]
    visual = [info for info in significant if info.visual]
    n_sig = len(significant)
    n_vis = len(visual)
    print(s)
    print('significant: {} / {}  ({:.2f}%)'.format(n_sig, n_cells, 100 * (n_sig / n_cells)))
    print('visual: {} / {}  ({:.2f}%)'.format(n_vis, n_cells, 100 * (n_vis / n_cells)))
    print('')
    ids = np.array([info.roi for info in visual])
    np.save(s.fs.getsyspath("visual.npy"), ids)


# sessions = get_sessions(day=5)
sessions = [
    open_session("36662-1", "2022-09-22", "1", fs=0),
    open_session("36662-2", "2022-09-22", "2", fs=0),
]
for s in sessions:
    compute_visual_selectivity(s)

