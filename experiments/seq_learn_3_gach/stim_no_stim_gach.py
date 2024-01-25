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

from ca_analysis.processing.utils import finalize_alignment

# sessions = get_sessions(day=5)
sessions = [
    # open_session("36662-2", "2022-09-22", "2", fs="ssd"),
    # open_session("58175-1", "2022-11-03", "1", fs="ssd"),
    # open_session("58181-1", "2022-11-03", "1", fs="ssd"),
    # open_session("58181-2", "2022-11-03", "1", fs="ssd"),
    # open_session("58181-3", "2022-11-04", "1", fs="ssd"),
]

s = open_session("55708-3", "2022-10-14", "1", fs='ca-nas')
schema = s.events.schema
frame_df = s.events['frame']
ev_df = s.events['event']
fps = s.events.get_fps()

G = np.load(s.fs.getsyspath("G.npy"))
G = xr.DataArray(G, dims=('time',), coords=dict(time=frame_df['time']))

gray_inds, stim_inds = [], []
for ev in schema.events:
    df = frame_df[frame_df['event'] == ev.id]
    locs = np.array(df.index)
    if ev.stimulus == 0:
        gray_inds.append(locs)
    else:
        stim_inds.append(locs)

gray_inds, stim_inds = np.concatenate(gray_inds), np.concatenate(stim_inds)
gray_data = G.isel(time=gray_inds)
stim_data = G.isel(time=stim_inds)
gray_data = gray_data[::10]
stim_data = stim_data[::10]
ks = kstest(gray_data, stim_data)

fig, ax = plt.subplots()
X, Y = gaussian_kde(gray_data)
ax.plot(X, Y, label='gray')
X, Y = gaussian_kde(stim_data)
ax.plot(X, Y, label='stim')
ax.legend()
plt.show()
