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

"""
--------------------------------------------------------------------------------
- utils
"""

"""
--------------------------------------------------------------------------------
- plots
"""


def plot_heatmap_and_histogram(
    s: Session,
    trials: Optional[slice] = None,
    heatmap_path=None,
    histogram_path=None,
):
    """

    Parameters
    ----------
    s: Session
    trials: slice (optional)
        Provide a slice with start/stop values to use a subset of trials. start/stop values
    are normalized (i.e., slice(1/3, 2/3) would use only middle third of trials).
    heatmap_path: path-like (optional)
        Where to save heatmap.
    histogram_path: path-like (optional)
        Where to save histogram.

    Returns
    -------
    None
    """

    # construct trial-averaged arrays
    s.transitions = get_transitions(s)
    for T in s.transitions.values():
        data = T.arr
        n_trials = data.sizes["trial"]
        if trials is not None or trials != slice(None):
            start = round(trials.start * n_trials) if trials.start else None
            stop = round(trials.stop * n_trials) if trials.stop else None
            data = data.isel(trial=slice(start, stop))
        T.plot_arr = data.mean("trial")
        T.plot_arr = T.plot_arr.transpose("roi", "time")

    # Initialize figure and axes.
    chunklen = 6

    nrows, ncols = len(s.transitions) // 2, 3
    width, height = 9, 3 * nrows

    # preprocess plot data (sorting by oddball score, making difference data, etc.)
    specs = [("BC", "AC"), ("CD", "BD"), ("DE", "CE"), ("EA", "DA"), ("AB", "EB")]
    rows = []

    for i in range(nrows):
        sp = specs[i]
        # grab transitions, and build "difference transition"
        high = s.transitions[sp[0]].plot_arr  # roi x time (trial averaged)
        high.name = sp[0]
        high.attrs["P"] = 0.9 - 0.1 * i
        high.attrs["Ptype"] = "HP"

        low = s.transitions[sp[1]].plot_arr
        low.name = sp[1]
        low.attrs["P"] = 0.1 + 0.1 * i
        low.attrs["Ptype"] = "LP"

        diff = low - high
        diff.name = f"{low.name} - {high.name}"
        diff.coords["event"] = high.coords["event"]
        diff.attrs["P"] = high.attrs["P"] - low.attrs["P"]
        diff.attrs["Ptype"] = "diff"

        # grab subregions of transitions for computing oddball score
        labels = [int(ev) for ev in high.coords["event"]]
        ind = np.squeeze(np.argwhere(np.ediff1d(labels, to_begin=0) != 0)).item()
        chunk_1 = diff.isel(time=slice(max(0, ind - chunklen), ind))  # before onset
        chunk_2 = diff.isel(time=slice(ind, ind + chunklen))  # after onset

        # compute oddball score: total absolute deflection (subtractive, not ratio)
        c1 = chunk_1.mean("time")
        c2 = chunk_2.mean("time")
        scores = c2 - c1

        # sort based on oddball score
        sorting_scores = chunk_2.mean("time")
        sort_order = np.flipud(np.squeeze(np.argsort(sorting_scores)))

        row = {
            "high": high,
            "low": low,
            "diff": diff,
            "scores": scores,
            "sort_order": sort_order,
        }
        rows.append(row)

    sort_order = rows[0]["sort_order"]
    for i, row in enumerate(rows):
        row["high"] = row["high"].isel(roi=sort_order)
        row["low"] = row["low"].isel(roi=sort_order)
        row["diff"] = row["diff"].isel(roi=sort_order)

    # build colormaps from pooled data
    reg_pooled, diff_pooled = [], []
    for i, row in enumerate(rows):
        reg_pooled.append(row["high"].data.flatten())
        reg_pooled.append(row["low"].data.flatten())
        diff_pooled.append(row["diff"].data.flatten())
    reg_pooled = np.hstack(reg_pooled)
    reg_vlim = np.percentile(reg_pooled, (2.5, 97.5))
    reg_norm = mpc.Normalize(vmin=reg_vlim[0], vmax=reg_vlim[1])
    reg_cmap = "inferno"

    diff_pooled = np.hstack(diff_pooled)
    diff_vlim = np.percentile(diff_pooled, (2.5, 97.5))
    diff_norm = mpc.Normalize(vmin=diff_vlim[0], vmax=diff_vlim[1])
    diff_cmap = "coolwarm"

    fig = plt.figure(figsize=(width, height))
    fig.tight_layout(pad=0)
    gs = fig.add_gridspec(nrows, ncols)

    def plot_one(arr, row, col, cmap, norm):
        ax = fig.add_subplot(gs[row, col])
        im = ax.imshow(
            arr,
            cmap=cmap,
            norm=norm,
            aspect="auto",
            interpolation="hanning",
        )

        midpoint = np.mean(ax.get_xlim())
        ax.axvline(midpoint, color="gray", ls="--")
        ax.set_xticks([])
        ax.set_yticks([])
        Ptype = arr.attrs.get("Ptype", None)
        if Ptype in {"HP", "LP"}:
            P = arr.attrs["P"]
            title = "$" + f"{Ptype}_{{" + "{:.1f}".format(P) + "}$" + f" ({arr.name})"
            ax.set_title(title)
        elif Ptype == "diff":
            P = arr.attrs["P"]
            title = "$\Delta P" + f"_{{" + "{:.1f}".format(P) + "}$" + f" ({arr.name})"
            ax.set_title(title)

        arr.attrs["ax"] = ax
        arr.attrs["im"] = im

    for i, row in enumerate(rows):
        high, low, diff = row["high"], row["low"], row["diff"]
        plot_one(high, i, 0, reg_cmap, reg_norm)
        plot_one(low, i, 1, reg_cmap, reg_norm)
        plot_one(diff, i, 2, diff_cmap, diff_norm)

    plt.show()
    if heatmap_path is not None:
        fig.savefig(heatmap_path)

    plt.rcParams.update({"font.size": 8})
    fig = plt.figure(figsize=(10, 8))
    fig.tight_layout()
    ax = fig.add_subplot()
    colors = ["black", "red", "teal", "blue", "gray"]
    linestyles = ["-", "-", "-", "-", "--"]
    for i, row in enumerate(rows):
        scores = row["scores"]
        X, Y = gaussian_kde(scores)
        arr = row["diff"]
        P = arr.attrs["P"]
        label = "$\Delta P" + f"_{{" + "{:.1f}".format(P) + "}$" + f" ({arr.name})"
        ax.plot(X, Y, color=colors[i], ls=linestyles[i], label=label)

    ax.legend()
    ax.set_xlim([-6, 6])
    ax.set_xlabel('$r$')
    ax.set_ylabel('density')
    plt.show()
    if histogram_path is not None:
        fig.savefig(histogram_path)

    # import scipy.stats
    # print("Rank Sum Tests (nov. excess ratios)\n------------------------")
    # chance_scores = rows[-1]["scores"]
    # for i in range(4):
    #     test_scores = rows[i]["scores"]
    #     pvalue = scipy.stats.ranksums(test_scores, chance_scores).pvalue
    #     name = rows[i]["diff"].name
    #     print(f"{name}: pvalue = {pvalue}")
    #
    # for i, row in enumerate(rows):
    #     scores = row["scores"]
    #     sc = scores.mean()
    #     print(f"{row['diff'].name} : {sc}")


def pooling_sandbox():
    sessions = [
        open_session("12488-1", "2022-02-09", "1", fs="ssd"),
        open_session("12489-1", "2022-02-09", "1", fs="ssd"),
        open_session("19071-1", "2022-02-09", "1", fs="ssd"),
        open_session("19075-3", "2022-02-09", "1", fs="ssd"),
        open_session("19088-2", "2022-02-09", "1", fs="ssd"),
        open_session("19089-1", "2022-02-09", "1", fs="ssd"),
    ]
    # s = sessions[0]
    # for s in sessions:
    #     path1 = f"/home/scott/newplots/{s.mouse}_heatmap.pdf"
    #     path2 = f"/home/scott/newplots/{s.mouse}_histogram.pdf"
    #     plot_heatmap_and_histogram(s, path1, path2)

    # get trial-averaged responses for each transition, across sessions.
    # put result in Transition object as Transition.lst.
    pool = get_transitions()
    for p in pool.values():
        p.data = []  # each transition gets a list

    for s in sessions:
        # filter 1 here: i.e., remove cells with low firing rates by
        # settings s.cells[i].iscell = False.

        # collect transitions for remaining cells.
        s.transitions = get_transitions(s)

        # get trial-averaged responses for each transition, and add to master list.
        for tname, T in s.transitions.items():
            data = T.arr
            ntrials = data.sizes["trial"]
            # data = data.isel(trial=slice(0, 10))  # front third
            # data = data.isel(trial=slice(0, ntrials // 3))  # front third
            # data = data.isel(trial=slice(ntrials // 3, 2* ntrials // 3))  # middle third
            data = data.isel(trial=slice(2 * ntrials // 3, None))  # back third

            pool[tname].data.append(data.mean("trial"))

    # concatenate master lists.
    for p in pool.values():
        p.data = xr.concat(p.data, dim="roi")  # this is n_timepoints in transition by n_rois.

    specs = [("BC", "AC"), ("CD", "BD"), ("DE", "CE"), ("EA", "DA"), ("AB", "EB")]

    chunklen = 6
    info = []
    for i, sp in enumerate(specs):
        high = pool[sp[0]].data
        high.name = sp[0]

        low = pool[sp[1]].data
        low.name = sp[1]

        diff = low - high
        diff.name = f"{low.name} - {high.name}"
        diff.coords["event"] = high.coords["event"]
        # diff.attrs["P"] = high.attrs["P"] - low.attrs["P"]

        # grab subregions of transitions for computing oddball score
        labels = [int(ev) for ev in high.coords["event"]]
        ind = np.squeeze(np.argwhere(np.ediff1d(labels, to_begin=0) != 0)).item()
        chunk_1 = diff.isel(time=slice(max(0, ind - chunklen), ind))  # before onset
        chunk_2 = diff.isel(time=slice(ind, ind + chunklen))  # after onset

        # compute oddball score: total absolute deflection (subtractive, not ratio)
        c1 = chunk_1.mean("time")
        c2 = chunk_2.mean("time")
        scores = c2 - c1

        # sort based on oddball score
        sorting_scores = chunk_2.mean("time")
        sort_order = np.flipud(np.squeeze(np.argsort(sorting_scores)))

        d = {"name": diff.name, "scores": scores}
        info.append(d)

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ["black", "red", "teal", "blue", "gray"]
    linestyles = ["-", "-", "-", "-", "--"]
    labels = ["AC/BC (0.1/0.8)", "BD/CD (0.2/0.7)", "CE/DE (0.3/0.6)", "DA/EA (0.4/0.5)", "(EB/AB) 0.5/0.9"]
    for i, d in enumerate(info):
        sc = d["scores"]
        X, Y = gaussian_kde(d["scores"])
        label = d["name"]
        ax.plot(X, Y, color=colors[i], ls=linestyles[i], label=labels[i])
    ax.legend()
    # ax.set_xlim([-5, 5])
    ax.set_xlim([0, 5])
    ax.set_xlabel("$\Delta r$")
    ax.set_ylabel("density")
    plt.show()

    fig.savefig("/home/scott/newplots/combined_histogram.pdf")


def compute_oddball_scores(
    s: Session,
    event: str,
    mode: str = "post",
    **kw,
) -> xr.DataArray:
    pair = TransitionPair(event, mode)
    high = s.data.split("spikes", pair[0].high, **kw)[1]
    low = s.data.split("spikes", pair[1].low, **kw)[1]
    high = high.mean("time").mean("trial")
    low = low.mean("time").mean("trial")
    scores = low - high
    return scores


def pooled(
    fn: Callable,
    sessions: List[Session],
    *args,
    **kw,
) -> Any:
    lst = []
    for s in sessions:
        res = fn(s, *args, **kw)
        lst.append(res)
    out = lst
    return out


def violinplot(sessions, **kw) -> "Figure":
    pairs = {}
    block = kw.pop("block", kw.pop("block_id", None))
    for event in ("A", "B", "C", "D", "E"):
        high, low = get_transition_pair(event, "post")
        p = SimpleNamespace(name=event, high=high, low=low)
        p.high_data = []
        p.low_data = []
        lst = []
        for s in sessions:
            high = s.data.split("spikes", p.high.events, block=block)[1]
            low = s.data.split("spikes", p.low.events, block=block)[1]
            high = high.mean("time")
            low = low.mean("time")
            p.high_data.append(high.data.flatten())
            p.low_data.append(low.data.flatten())
        p.high_data = np.concatenate(p.high_data)
        p.low_data = np.concatenate(p.low_data)
        pairs[event] = p

    fig, ax = plt.subplots()
    styles = {
        "C": dict(color="black", lw=3),
        "D": dict(color="blue", lw=2),
        "E": dict(color="red", lw=2),
        "A": dict(color="gray", lw=2),
        "B": dict(color="black", lw=1, ls="--"),
    }
    likes = {
        "C": 0.8 / 0.1,
        "D": 0.7 / 0.2,
        "B": 0.9 / 0.5,
        "A": 0.5 / 0.4,
        "E": 0.6 / 0.5,
    }

    datasets = []
    xticklabels = []
    for event in pairs:
        p = pairs[event]
        datasets += [p.high_data, p.low_data]
        xticklabels += [p.high.name, p.low.name]
    positions = np.arange(len(datasets))

    showextrema = kw.pop("showextrema", False)
    points = kw.pop("points", 500)
    ax.violinplot(
        datasets,
        positions,
        showextrema=showextrema,
        points=points,
        **kw,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(xticklabels)
    ax.set_ylim([0, 10])

    plt.show()
    return fig


def prepare_oddball_scores(
    ses: Union[Session, List[Session]],
    block=None,
    events=("A", "B", "C", "D", "E"),
) -> Mapping:
    def prepare_one(s: Session):
        s.data.cell_mode = True
        s.pairs = {}
        for ev in events:
            s.pairs[ev] = TransitionPair(ev, parent=s)
            high = s.data.split("spikes", p.high.events, block=block)[1]
            low = s.data.split("spikes", p.low.events, block=block)[1]
            arrays = [high, low]
            for i, mat in enumerate(arrays):
                mat = mat.mean("time")
                mat = mat.mean("trial")
                arrays[i] = mat
            high, low = arrays
            arr = low - high
            s.pairs[ev].scores = arr

    # Initialize sessions` transitions pairs
    ses = [s] if isinstance(ses, Session) else ses
    for s in ses:
        s.data.cell_mode = True
        s.pairs = {}
        for ev in events:
            p = s.pairs[ev] = TransitionPair(ev, parent=s)
            high_raw = s.data.split("spikes", p.high.events, block=block)[1]
            low_raw = s.data.split("spikes", p.low.events, block=block)[1]
            arrays = [high_raw, low_raw]
            for i, mat in enumerate(arrays):
                mat = mat.mean("time")
                mat = mat.mean("trial")
                arrays[i] = mat
            high_raw, low_raw = arrays
            arr = low_raw - high_raw
            s.pairs[ev].scores = arr

    # Initialize pooled transitions pairs
    pairs = {}
    for ev in events:
        pairs[ev] = TransitionPair(ev, parent=ses)

    # Do pooling
    for ev in events:
        for s in ses:
            high_raw = s.data.split("spikes", p.high.events, block=block)[1]
            low_raw = s.data.split("spikes", p.low.events, block=block)[1]
            arrays = [high_raw, low_raw]
            for i, mat in enumerate(arrays):
                mat = mat.mean("time")
                mat = mat.mean("trial")
                arrays[i] = mat
            high_raw, low_raw = arrays
            arr = low_raw - high_raw
            s.pairs[ev].scores = arr
            pairs[ev].scores.append(arr)
        pairs[ev].flatten_scores()

    return pairs


_SESSION_ARGS = [
    ("12488-1", "2022-02-09", "1"),
    ("12489-1", "2022-02-09", "1"),
    ("19071-1", "2022-02-09", "1"),
    ("19075-3", "2022-02-09", "1"),
    ("19088-2", "2022-02-09", "1"),
    ("19089-1", "2022-02-09", "1"),
]


def get_session(num, cell_mode: bool = True, fs="ssd"):
    args = _SESSION_ARGS[num]
    s = open_session(*args, fs=fs)
    s.data.cell_mode = cell_mode
    s.pairs = {}
    return s


def pool_scores(
    ses: Union[Session, List[Session]],
    block: IndexLike = slice(None),
    mode: str = "post",
    events: Sequence = ("A", "B", "C", "D", "E"),
) -> Mapping:
    # Initialize sessions` transitions pairs
    ses = [ses] if isinstance(ses, Session) else ses
    for s in ses:
        s.data.cell_mode = True
        s.pairs = {}
        for ev in events:
            p = TransitionPair(ev, mode=mode)
            high_raw = s.data.split("spikes", p.high.events, block=block)[1]
            low_raw = s.data.split("spikes", p.low.events, block=block)[1]
            p.high_raw = high_raw
            p.low_raw = low_raw
            arrays = [high_raw, low_raw]
            for i, mat in enumerate(arrays):
                mat = mat.mean("time")
                mat = mat.mean("trial")
                arrays[i] = mat
            p.high, p.low = arrays
            arr = p.low - p.high
            p.scores = arr
            s.pairs[ev] = p

    # Pool transitions pairs
    pairs = {}
    for ev in events:
        p = TransitionPair(ev)
        lst = [s.pairs[ev].scores for s in ses]
        p.scores = np.hstack(lst)
        pairs[ev] = p

    return pairs


def make_fig(ses, mode: str = "post"):
    pair_to_style = {
        "A": dict(color="gray", ls="--"),
        "B": dict(color="blue", ls="-"),
        "C": dict(color="black", ls="-"),
        "D": dict(color="red", ls="-"),
        "E": dict(color="lightblue", ls="-"),
    }

    fig = plt.figure(figsize=(4, 12))
    axes = [fig.add_subplot(4, 1, i + 1) for i in range(4)]
    for plotnum, block in enumerate([2, 3, 4, slice(None)]):
        ax = axes[plotnum]
        pairs = pool_scores(ses, block=block, mode=mode)
        for key, p in pairs.items():
            # compute surprise and use for legend label
            high, low = p.high, p.low
            surprise = np.log(high.P / low.P)
            p.surprise = surprise
            label = f"{p.key}: {high.P} / {low.P}"
            label += ": {:.2f}".format(p.surprise)

            # get style
            style = pair_to_style[p.key]

            # plot curves
            arr = p.scores
            arr = np.abs(arr)
            x, y = gaussian_kde(arr)
            ax.plot(x, y, label=label, **style)

            # etc.
            title = f"block: {block}" if is_int(block) else "all blocks"
            ax.set_title(title)
            ax.legend()
            # ax.set_xlim([-4, 4])
            ax.set_xlim([0, 2])

    fig.tight_layout()
    return fig


def pool_histogram_scores(
    ses: Union[Session, List[Session]],
    block: IndexLike = slice(None),
    events: Sequence = ("A", "B", "C", "D", "E"),
) -> Mapping:
    ses = [ses] if isinstance(ses, Session) else ses
    for s in ses:
        s.pairs = {}
        s.transitions = get_transitions()
        for name, tr in s.data.transitions.items():
            pre, post = s.data.split("spikes", [tr.pre, tr.post], block=block)
            tr.pre_data: xr.DataArray = pre.mean("trial")
            tr.post_data: xr.DataArray = post.mean("trial")

        s.pairs = {}
        for ev in events:
            pair = TransitionPair(ev, mode="post")
            high, low = pair.high, pair.low
            trans = [high, low]
            for tr in trans:
                pre, post = s.data.split("spikes", [tr.pre, tr.post], block=block)
                tr.pre_data: xr.DataArray = pre.mean("trial")
                tr.post_data: xr.DataArray = post.mean("trial")

            # method one: subtract trial averaged data, then time average.
            diff_pre = low.pre_data - high.pre_data
            diff_post = low.post_data - high.post_data
            c1 = diff_pre.mean("time")
            c2 = diff_post.mean("time")
            scores = c2 - c1
            pair.scores = scores.data
            s.pairs[ev] = pair

    # Pool transitions pairs
    pairs = {}
    for ev in events:
        p = TransitionPair(ev)
        lst = [s.pairs[ev].scores for s in ses]
        p.scores = np.hstack(lst)
        pairs[ev] = p

    return pairs



def pool_flat(lst):
    return np.hstack([np.asarray(mat).reshape(-1) for mat in lst])


def compute_empirical_statistics():
    T = get_transitions()
    dct = {}
    for name, tr in get_transitions().items():
        pre, post = tr.pre, tr.post
        key = (int(pre), int(post))
        dct[key] = tr

    s = sessions[0]
    df = s.events.tables["event"]
    df = df[df["block_id"] > 1]
    ids = df["event_id"].values
    n_trans = len(ids)
    counts = np.zeros([len(ids), 7, 7], dtype=int)
    for i in range(1, len(ids)):
        counts[i] = counts[i - 1]
        cur_id, prev_id = ids[i], ids[i - 1]
        if cur_id < 2 or prev_id < 2:
            continue
        counts[i, prev_id, cur_id] += 1
        row = counts[i, cur_id]
        nobs = row.sum()
        tot = row.sum()


def plot_pair(s: Session, roi: int, pair: str, axes: "Axes"):
    pair = s.data.get_pair(pair)
    high, low = pair.high, pair.low

    dset, ax = pair.high, axes[0]
    pre = dset.data[0].isel(roi=roi)
    post = dset.data[1].isel(roi=roi)
    Y = xr.concat([pre, post], "time")
    n_trials = Y.sizes["trial"]
    X = np.arange(16)
    ax.axvline(7.5)
    for i in range(n_trials):
        ax.plot(X, Y.isel(trial=i).values, color='black', alpha=0.5)
    ax.set_title(dset.name)

    dset, ax = pair.low, axes[1]
    pre = dset.data[0].isel(roi=roi)
    post = dset.data[1].isel(roi=roi)
    Y = xr.concat([pre, post], "time")
    n_trials = Y.sizes["trial"]
    X = np.arange(16)
    ax.axvline(7.5)
    for i in range(n_trials):
        ax.plot(X, Y.isel(trial=i).values, color='black', alpha=0.5)
    ax.set_xlabel('time')
    ax.set_title(dset.name)

    ymax = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    for ax in axes:
        ax.set_ylim([0, ymax])


def cdf_plot():
    logger = get_logger(logging.INFO)

    sessions = get_sessions()
    if sessions[0].data.transitions["AC"].data[0] is None:
        load_spikes(sessions)

    """
    Look at variability between elements
    """

    T, pairs = get_transitions(), get_transition_pairs()
    for tr in itertransitions(sessions):
        pre = tr.data[0].mean("time")
        post = tr.data[1].mean("time")
        pre = pre.mean("trial")
        post = post.mean("trial")
        tr.scores = post - pre
        # tr.scores = post
        T[tr.name].scores.append(tr.scores)

    for tr in T.values():
        tr.scores = pool_flat(tr.scores)

    PLOTDIR = Path.home() / f"plots/cdfs"
    PLOTDIR.mkdir(exist_ok=True, parents=True)

    T1, T2 = T["AC"], T["BC"]
    S1, S2 = T1.scores, T2.scores
    n = len(S1) // 2
    IDs = shuffled(np.arange(len(S1)))
    IDs1, IDs2 = IDs[:n], IDs[n:]
    fig = plt.figure(figsize=(4, 12))
    for i, p in enumerate(pairs.values()):
        ax = fig.add_subplot(5, 1, i + 1)
        T1, T2 = T[p.high_name], T[p.low_name]
        S1, S2 = T1.scores, T2.scores
        S1 = S1[IDs1]
        S2 = S2[IDs2]

        ecdf1 = ECDF(S1)
        ecdf2 = ECDF(S2)

        xlim = np.array([-5, 5])
        X = np.linspace(xlim[0], xlim[1], 1000)

        # x, y = gaussian_kde(S1, X)
        # ax.plot(x, y, label=T1.name)
        ax.plot(ecdf1.x, ecdf1.y, label=T1.name)
        # x, y = gaussian_kde(S2, X)
        # ax.plot(x, y, label=T2.name)
        ax.plot(ecdf2.x, ecdf2.y, label=T2.name)

        ax.legend()
        ax.set_xlim(xlim)

        stat = kstest(S1, S2)
        title = f"ks-test: p={stat.pvalue}"
        ax.set_title(title)

    fname = PLOTDIR / "pooled_nov_excess.pdf"
    fig.tight_layout()
    fig.savefig(fname)


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



# function to calculate Cohen's d for independent samples
def cohend(d1, d2):
    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    # calculate the pooled standard deviation
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = np.mean(d1), np.mean(d2)
    # calculate the effect size
    delta = abs(u1 - u2)
    return delta / s


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


def make_histogram_figure(
    ses: Union[Session, List[Session]],
    score_fn: Callable = lambda pre, post: post - pre,
) -> "Figure":

    pair_to_style = {
        "A": dict(color="blue", ls="-"),
        "B": dict(color="gray", ls="--"),
        "C": dict(color="black", ls="-"),
        "D": dict(color="red", ls="-"),
        "E": dict(color="gray", ls="-"),
    }

    xlim = [-3, 3]
    X = np.linspace(xlim[0], xlim[1], 1000)

    fig = plt.figure(figsize=(4, 12))
    axes = [fig.add_subplot(4, 1, i + 1) for i in range(4)]
    for plot_num, block in enumerate([2, 3, 4, slice(2, None)]):
        ax = axes[plot_num]
        T, pairs = pool_histogram_scores(ses, block=block, score_fn=score_fn)
        for key in ["C", "D", "E", "A", "B"]:
            # for key in "ABCDE":
            p = pairs[key]
            # compute surprise and use for legend label
            high, low = p.high, p.low
            label = f"{low.name} - {high.name}"

            # get style
            style = pair_to_style[p.key]

            # plot curves
            arr = p.scores
            # arr = np.abs(arr)
            x, y = gaussian_kde(arr, X)
            ax.plot(x, y, label=label, **style)

            # etc.
            title = f"block: {block}" if is_int(block) else "all blocks"
            ax.set_title(title)
            ax.legend()
            ax.set_xlim(xlim)
            ax.set_ylim([0, 0.7])

    fig.tight_layout()
    return fig


def make_cdf_figure(
    ses: Union[Session, List[Session]],
    score_fn: Callable = lambda pre, post: post - pre,
) -> "Figure":

    pair_to_style = {
        "A": dict(color="blue", ls="-"),
        "B": dict(color="gray", ls="--"),
        "C": dict(color="black", ls="-"),
        "D": dict(color="red", ls="-"),
        "E": dict(color="gray", ls="-"),
    }

    xlim = [-5, 5]
    ylim = [0, 1.05]
    X = np.linspace(xlim[0], xlim[1], 1000)

    fig = plt.figure(figsize=(4, 12))
    axes = [fig.add_subplot(4, 1, i + 1) for i in range(4)]
    for plot_num, block in enumerate([2, 3, 4, slice(2, None)]):
        ax = axes[plot_num]
        T, pairs = pool_histogram_scores(ses, block=block, score_fn=score_fn)
        for key in ["C", "D", "E", "A", "B"]:
            # for key in "ABCDE":
            p = pairs[key]
            # compute surprise and use for legend label
            high, low = p.high, p.low
            label = f"{low.name} - {high.name}"

            # get style
            style = pair_to_style[p.key]

            # plot curves
            arr = p.scores
            ecdf = ECDF(arr)
            x, y = ecdf.x, ecdf.y
            ax.plot(x, y, label=label, **style)

            # etc.
            title = f"block: {block}" if is_int(block) else "all blocks"
            ax.set_title(title)
            ax.legend()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

    fig.tight_layout()
    return fig

def dump_jumps_to_csv():

    sessions = get_sessions()
    if sessions[0].data.transitions["AC"].data[0] is None:
        load_spikes(sessions, block=slice(None))

    score_fn = lambda pre, post: post - pre
    # score_fn = lambda pre, post: post / pre
    # score_fn = lambda pre, post: (post + 1) / (pre + 1)

    # PLOTDIR = Path.home() / f"plots/jump_cdfs"
    # PLOTDIR.mkdir(exist_ok=True, parents=True)
    # for ses in sessions:
    #     fig = make_cdf_figure(ses, score_fn=score_fn)
    #     fname = PLOTDIR / f"{ses.mouse}.pdf"
    #     fig.savefig(fname)
    #
    # fig = make_cdf_figure(sessions, score_fn=score_fn)
    # fname = PLOTDIR / f"pooled.pdf"
    # fig.savefig(fname)
    block_col = []
    label_col = []
    data_col = []
    for blk in [2, 3, 4]:
        T, pairs = pool_histogram_scores(sessions, block=blk, score_fn=score_fn)
        for p in pairs.values():
            scores = p.scores
            n = len(scores)
            block_col.append(np.full(n, blk))
            label_col.append(np.full(n, p.key))
            data_col.append(p.scores)

    block_col = np.hstack(block_col)
    label_col = np.hstack(label_col)
    data_col = np.hstack(data_col)
    df = pd.DataFrame(dict(block=block_col, label=label_col, data=data_col))
    # df.to_csv(Path.home() / 'jumps.csv')
    return df
    # import statsmodels.api as sm
    # from statsmodels.formula.api import ols

    # moore = sm.datasets.get_rdataset("Moore", "carData", cache=True)  # load data
    # data = moore.data
    # model = ols("data ~ C(label)", data=df).fit()
    # table = sm.stats.anova_lm(model, typ=2)


def stattests():

    sessions = get_sessions()
    if sessions[0].data.transitions["AC"].data[0] is None:
        load_spikes(sessions, block=slice(None))

    fig, axes = plt.subplots(4, 1)
    for block_num, blk in enumerate([2, 3, 4, slice(None)]):
        print("Block: ", blk, "----------------")
        T, pairs = pool_histogram_scores(sessions, block=blk)
        events = list("CDEAB")
        mat = np.zeros([len(events), len(events)])
        for i in range(len(events)):
            for j in range(len(events)):
            # for j in range(i + 1, len(events)):
                ev1, ev2 = events[i], events[j]
                a, b = pairs[ev1], pairs[ev2]
                label1 = f"{a.low.name}-{a.high.name}"
                label2 = f"{b.low.name}-{b.high.name}"
                S1, S2 = a.scores, b.scores
                stat = kstest(S1, S2)
                # stat = f_oneway(S1, S2)
                is_sig = stat.pvalue < 0.05 / 5
                mat[i, j] = 1 - stat.pvalue / 10
                # if is_sig:
                #     mat[i, j] = 1
                print("\t", label1, label2, is_sig, stat)

        ax = axes[block_num]
        ax.imshow(mat, cmap='gray')
    plt.show()


logger = get_logger(logging.INFO)


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


if __name__ == "__main__":

    logger = get_logger(logging.INFO)

    sessions = get_sessions()
    if sessions[0].data.transitions["AC"].data[0] is None:
        load_spikes(sessions, block=slice(None))

    filter_rois = False
    jump_mode = True
    if filter_rois:
        for s in sessions:
            spikes = s.data.get("spikes")
            means = spikes.mean("time")
            cutoff = np.percentile(means, 50)
            s.keep_rois = argwhere(means > cutoff)

    T, pairs = get_transitions(), get_transition_pairs()
    for tr in itertransitions(sessions):
        pre = tr.data[0].mean("time")
        post = tr.data[1].mean("time")
        pre = pre.mean("trial")
        post = post.mean("trial")

        # filter rois
        if filter_rois:
            pre = pre.isel(roi=tr.session.keep_rois)
            post = post.isel(roi=tr.session.keep_rois)
        if jump_mode:
            tr.scores = post - pre
        else:
            tr.scores = post
        # tr.scores = post - pre
        # tr.scores = post / pre
        # tr.scores = (post + 1) / (pre + 1)
        T[tr.name].scores.append(tr.scores)

    for tr in T.values():
        tr.scores = pool_flat(tr.scores)

    PLOTDIR = Path.home() / f"plots/cdfs"
    PLOTDIR.mkdir(exist_ok=True, parents=True)

    T1, T2 = T["AC"], T["BC"]
    S1, S2 = T1.scores, T2.scores
    n = len(S1) // 2
    IDs = shuffled(np.arange(len(S1)))
    IDs1, IDs2 = IDs[:n], IDs[n:]
    # IDs1, IDs2 = slice(None), slice(None)
    fig = plt.figure(figsize=(4, 6))
    fig, axes = plt.subplots(2, 1, figsize=(4, 6))
    mypairs = [pairs["B"], pairs['C']]
    xlim = np.array([-5, 5]) if jump_mode else np.array([0, 5])

    for i, p in enumerate(mypairs):
        # ax = fig.add_subplot(2, 1, i + 1)
        ax = axes[i]
        T1, T2 = T[p.high_name], T[p.low_name]
        S1, S2 = T1.scores, T2.scores
        S1 = S1[IDs1]
        S2 = S2[IDs2]

        ecdf1 = ECDF(S1)
        ecdf2 = ECDF(S2)

        ax.plot(ecdf1.x, ecdf1.y, label=T1.name)
        ax.plot(ecdf2.x, ecdf2.y, label=T2.name)
        dx = np.abs(ecdf1.x - ecdf2.x)

        ax.legend()
        ax.set_xlim(xlim)

        ks_stat = kstest(S1, S2)
        pval = ks_stat.pvalue
        if i == 0:
            title = "$\Delta P = 0.0$"
            label = "ks-test: p = {:.3f}".format(ks_stat.pvalue)
        else:
            title = "$\Delta P = 0.8$".format(ks_stat.pvalue)
            label = "ks-test: p = {:.3e}".format(ks_stat.pvalue)
        ax.set_title(title)
        ax.text(0.2, 0.2, label)
        print(p.name, repr(ks_stat))

        stat = ttest(S1, S2)
        # print(p.name, stat)

        d = cohend(S1, S2)
        print(p.name, "cohen d", d)

    ax1, ax2 = axes
    fig.tight_layout()

    fig.savefig(PLOTDIR / "cdf.pdf")
    fig.savefig(PLOTDIR / "cdf.eps")
    plt.show()


