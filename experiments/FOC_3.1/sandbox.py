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
    Callable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import h5py
import pandas as pd
from matplotlib.animation import FFMpegWriter
import matplotlib.colors as mpc
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import xarray as xr

from ca_analysis import *
from ca_analysis.grouping import *
from ca_analysis.plot import *
from ca_analysis.traces import *
from ca_analysis.stats import *

from main import get_transitions, open_session, schema
from processing import import_session

"""
-------------------------------------------------------------------------------------
- utils
"""

def get_sessions(fs="ssd"):
    sessions = [
        open_session("611486-1", "2021-07-13", fs=fs),
        open_session("611521-1", "2021-07-14", fs=fs),
        open_session("619739-1", "2021-08-03", fs=fs),
        open_session("619745-2", "2021-10-19", fs=fs),
        open_session("619766-4", "2021-10-19", fs=fs),
        open_session("619816-2", "2021-11-02", fs=fs),
    ]
    return sessions


def get_splits(arr: xr.DataArray) -> List[xr.DataArray]:
    # Split array by event
    labels = [int(ev) for ev in arr.coords["event"]]
    inds = np.squeeze(np.argwhere(np.ediff1d(labels, to_begin=0) != 0))
    if inds.ndim == 0:
        # one split
        locs = [inds.item()]
    else:
        locs = [] if np.array_equal(inds, []) else list(inds)
    locs.insert(0, 0)
    locs.append(None)

    # split into chunks
    out = []
    for i in range(len(locs) - 1):
        chunk = arr.isel(time=slice(locs[i], locs[i + 1]))
        out.append(chunk)
    return out

"""
-------------------------------------------------------------------------------------
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
        high = s.transitions[sp[0]].plot_arr
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
        open_session("611486-1", "2021-07-13", fs="ssd"),
        open_session("611521-1", "2021-07-14", fs="ssd"),
        open_session("619739-1", "2021-08-03", fs="ssd"),
        open_session("619745-2", "2021-10-19", fs="ssd"),
        open_session("619766-4", "2021-10-19", fs="ssd"),
        open_session("619816-2", "2021-11-02", fs="ssd"),
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
        p.data = [] # each transition gets a list

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
        X, Y = gaussian_kde(d["scores"])
        label = d["name"]
        ax.plot(X, Y, color=colors[i], ls=linestyles[i], label=labels[i])
    ax.legend()
    ax.set_xlim([-5, 5])
    ax.set_xlabel("$\Delta r$")
    ax.set_ylabel("density")
    plt.show()

    fig.savefig("/home/scott/newplots/combined_histogram.pdf")







if __name__ == "__main__":

    sessions = get_sessions()

    #
    # sessions = [
    #     open_session("611486-1", "2021-07-13", fs="ssd"),
    #     open_session("611521-1", "2021-07-14", fs="ssd"),
    #     open_session("619739-1", "2021-08-03", fs="ssd"),
    #     open_session("619745-2", "2021-10-19", fs="ssd"),
    #     open_session("619766-4", "2021-10-19", fs="ssd"),
    #     open_session("619816-2", "2021-11-02", fs="ssd"),
    # ]
    s = open_session("619745-2", "2021-10-21", fs="ssd")
    import_session(s)

    # s = sessions[0]
    # for s in sessions:
    #     path1 = f"/home/scott/newplots/qualplots/{s.mouse}_heatmap.png"
    #     path2 = f"/home/scott/newplots/qualplots/{s.mouse}_histogram.png"
    #     plot_heatmap_and_histogram(s, slice(2/3, 1), heatmap_path=path1, histogram_path=path2)

    # get trial-averaged responses for each transition, across sessions.
    # put result in Transition object as Transition.lst.
    #
    # mode = 1
    # assert mode in range(4)
    # pool = get_transitions()
    # for p in pool.values():
    #     p.data = []  # each transition gets a list
    #
    # for s in sessions:
    #     # filter 1 here: i.e., remove cells with low firing rates by
    #     # settings s.cells[i].iscell = False.
    #
    #     # collect transitions for remaining cells.
    #     s.transitions = get_transitions(s)
    #
    #     # get trial-averaged responses for each transition, and add to master list.
    #     for tname, T in s.transitions.items():
    #         data = T.arr
    #         ntrials = data.sizes["trial"]
    #         if mode == 0:
    #             data = data
    #         elif mode == 1:
    #             data = data.isel(trial=slice(0, ntrials // 3))  # front third
    #         elif mode == 2:
    #             data = data.isel(trial=slice(ntrials // 3, 2 * ntrials // 3))  # middle third
    #         elif mode == 3:
    #             data = data.isel(trial=slice(2 * ntrials // 3, None))  # back third
    #
    #         pool[tname].data.append(data.mean("trial"))
    #
    # # concatenate master lists.
    # for p in pool.values():
    #     p.data = xr.concat(p.data, dim="roi")  # this is n_timepoints in transition by n_rois.
    #
    # specs = [("BC", "AC"), ("CD", "BD"), ("DE", "CE"), ("EA", "DA"), ("AB", "EB")]
    #
    # chunklen = 6
    # info = []
    #
    # for i, sp in enumerate(specs):
    #
    #     high = pool[sp[0]].data
    #     high.name = sp[0]
    #
    #     low = pool[sp[1]].data
    #     low.name = sp[1]
    #
    #     diff = low - high
    #     diff.name = f"{low.name} - {high.name}"
    #     diff.coords["event"] = high.coords["event"]
    #     # diff.attrs["P"] = high.attrs["P"] - low.attrs["P"]
    #
    #     # grab subregions of transitions for computing oddball score
    #     labels = [int(ev) for ev in high.coords["event"]]
    #     ind = np.squeeze(np.argwhere(np.ediff1d(labels, to_begin=0) != 0)).item()
    #     chunk_1 = diff.isel(time=slice(max(0, ind - chunklen), ind))  # before onset
    #     chunk_2 = diff.isel(time=slice(ind, ind + chunklen))  # after onset
    #
    #     # compute oddball score: total absolute deflection (subtractive, not ratio)
    #     c1 = chunk_1.mean("time")
    #     c2 = chunk_2.mean("time")
    #     scores = c2 - c1
    #     # scores = np.abs(scores)
    #
    #     # sort based on oddball score
    #     sorting_scores = chunk_2.mean("time")
    #     sort_order = np.flipud(np.squeeze(np.argsort(sorting_scores)))
    #
    #     d = {"name": diff.name, "scores": scores}
    #     info.append(d)
    #
    #
    # fig, ax = plt.subplots(figsize=(8, 8))
    # colors = ["black", "red", "teal", "blue", "gray"]
    # linestyles = ["-", "-", "-", "-", "--"]
    # labels = [
    #     "AC/BC (0.1/0.8)",
    #     "BD/CD (0.2/0.7)",
    #     "CE/DE (0.3/0.6)",
    #     "DA/EA (0.4/0.5)",
    #     "(EB/AB) 0.5/0.9",
    # ]
    # for i, d in enumerate(info):
    #     X, Y = gaussian_kde(d["scores"])
    #     label = d["name"]
    #     ax.plot(X, Y, color=colors[i], ls=linestyles[i], label=labels[i])
    # ax.legend()
    # ax.set_xlim([-5, 5])
    # ax.set_xlabel("$r$")
    # ax.set_ylabel("density")
    # if mode == 0:
    #     ax.set_title("all")
    #     fname = "all"
    # elif mode == 1:
    #     ax.set_title("front third")
    #     fname = "1"
    # elif mode == 2:
    #     ax.set_title("middle third")
    #     fname = "2"
    # elif mode == 3:
    #     ax.set_title("last third")
    #     fname = "3"
    #
    # plt.show()
    #
    # fig.savefig(f"/home/scott/newplots/qualplots/combined_histogram_{fname}.png")
    #
