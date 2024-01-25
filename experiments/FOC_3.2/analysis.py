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
from sklearn.linear_model import LinearRegression
import xarray as xr

xr.set_options(keep_attrs=True)

from ca_analysis import *
from ca_analysis.environment import *
from ca_analysis.grouping import *
from ca_analysis.plot import *
from ca_analysis.traces import *
from ca_analysis.stats import *



"""
-------------------------------------------------------------------------------------
- plots
"""


def make_meanmov(
        s: Session,
        outfile="analysis/meanmov.mp4",
        cmap="inferno",
        qlim=(2.5, 99.5),
        sigma=1,
        upsample=4,
        fps=30,
        dpi=220,
        frameon=True,
        facecolor=None,
        inches_per_plot=3,
        reverse_DCBA: bool = True,
):
    t0 = time.time()

    from matplotlib.figure import Figure
    from matplotlib.animation import FFMpegWriter

    schema = s.events.schema
    day = s.attrs["day"]
    fps = s.attrs["capture"]["fps"]
    outfile = Path(s.fs.getsyspath(outfile)).with_suffix(".mp4")

    if day == 1:
        SPECS = {"ABCD": [FPad("A"), "A", "B", "C", "D", BPad("D")]}

    else:
        if reverse_DCBA:
            SPECS = {
                "ABCD": [FPad("A"), "A", "B", "C", "D", BPad("D")],
                "DCBA": [FPad("D_r"), "A_r", "B_r", "C_r", "D_r", BPad("A_r")],
                "ABBD": [FPad("A_g"), "A_g", "B_g", "C_g", "D_g", BPad("D_g")],
                "ABCD_novT": [FPad("A_novT"), "A_novT", "B_novT", "C_novT", "D_novT", BPad("D_novT")],
            }
        else:
            SPECS = {
                "ABCD": [FPad("A"), "A", "B", "C", "D", BPad("D")],
                "DCBA": [FPad("D_r"), "D_r", "C_r", "B_r", "A_r", BPad("A_r")],
                "ABBD": [FPad("A_g"), "A_g", "B_g", "C_g", "D_g", BPad("D_g")],
                "ABCD_novT": [FPad("A_novT"), "A_novT", "B_novT", "C_novT", "D_novT", BPad("D_novT")],
            }

            # Change duration of novel elements.
            for ev_name in ("A", "B", "C", "D"):
                base = schema[ev_name]
                novel = schema[ev_name + "_novT"]
                novel.duration = base.duration

    # Group data, and average over trials.
    f = h5py.File(s.fs.getsyspath(PATHS["mov"]), "r")
    mov = f["data"]
    grouper = Grouper(s.events)
    items = {}  # seq_name -> SimpleNamespace
    print(f't1: {time.time() - t0}')
    for name, spec in SPECS.items():
        # Group movies by element type, get trial average, and concatenate them.
        g = SimpleNamespace(name=name)
        lst = grouper.group(mov, spec)
        for i, obj in enumerate(lst):
            lst[i] = obj.mean("trial")
        g.arr = xr.concat(lst, dim="dim_1")

        # Get time-series of labels.
        labels = []
        for elt in lst:
            labels.extend([elt.name] * len(elt))
        g.labels = simplify_labels(labels)

        items[name] = g

    f.close()
    print(f't2: {time.time() - t0}')

    # Optionally upsample.
    if upsample and upsample != 1:
        for name, g in items.items():
            n_frames_out = len(g.arr) * upsample
            arr = scipy.signal.resample(g.arr, n_frames_out, axis=0)
            g.arr = xr.DataArray(arr, name=name, dims=("time", "ypix", "xpix"))
            g.labels = resample_labels(g.labels, factor=upsample)
    print(f't2: {time.time() - t0}')

    n_plots = len(items)
    width = inches_per_plot * n_plots
    height = inches_per_plot
    fig = Figure(figsize=(width, height))
    fig.tight_layout(pad=0)

    count = 0
    for name, g in items.items():
        count += 1
        g.ax = ax = fig.add_subplot(1, n_plots, count)
        remove_ticks(g.ax)
        g.ax.set_title(g.name)
        g.im = g.ax.imshow(g.arr[0])
        g.smap = get_smap(data=g.arr, qlim=qlim, cmap=cmap)
        if sigma:
            g.arr = gaussian_filter(g.arr, sigma)

        fontdict = {'size': 16, 'color': 'white'}
        label_loc = [0.05, 0.95]
        g.lbl = ax.text(label_loc[0], label_loc[1], ' ',
                        fontdict=fontdict,
                        transform=ax.transAxes,
                        horizontalalignment='left',
                        verticalalignment='top',
                        usetex=False)

    print(f't3: {time.time() - t0}')
    n_frames = max([g.arr.shape[0] for g in items.values()])
    writer = FFMpegWriter(fps=fps)
    with writer.saving(fig, str(outfile), dpi):
        for i in range(n_frames):
            for g in items.values():
                if i >= g.arr.shape[0]:
                    continue
                cdata = g.smap(g.arr[i])
                cdata[:, :, -1] = 1
                g.im.set_data(cdata)
                g.lbl.set_text(g.labels[i])
                writer.grab_frame()
    print(f't4: {time.time() - t0}')


def make_roi_meanmov(
        s: Session,
        outfile="analysis/roi_meanmov.mp4",
        cmap="inferno",
        qlim=(2.5, 99.5),
        sigma=1,
        upsample=4,
        fps=30,
        dpi=220,
        frameon=True,
        facecolor=None,
        inches_per_plot=3,
        reverse_DCBA: bool = True,
):
    outfile = Path(s.fs.getsyspath(outfile)).with_suffix(".mp4")

    fps = s.attrs["capture"]["fps"]
    cells = s.cells
    iscell = s.segmentation["iscell"]
    spikes = s.segmentation["spikes"]
    spikes = spikes[:, iscell]
    frame_shape = s.attrs["capture"]["frame_shape"]

    # Group data, and average over trials.
    SPECS = get_grouper_specs(s, reverse_DCBA=reverse_DCBA, truncate_novT=True)
    grouper = Grouper(s.events)
    items = {}  # seq_name -> SimpleNamespace
    for name, spec in SPECS.items():
        # Group movies by element type, get trial average, and concatenate them.
        g = SimpleNamespace(name=name)
        lst = grouper.group(spikes, spec)
        for i, obj in enumerate(lst):
            lst[i] = obj.mean("trial")
        g.spikes = xr.concat(lst, dim="time")

        # Get time-series of labels.
        labels = []
        for elt in lst:
            labels.extend([elt.name] * len(elt))
        g.labels = simplify_labels(labels)
        items[name] = g

        # Optionally upsample and smooth spike data.
    if upsample and upsample != 1:
        for g in items.values():
            n_frames_out = len(g.spikes) * upsample
            spikes_out = scipy.signal.resample(g.spikes, n_frames_out, axis=0)
            g.spikes = xr.DataArray(spikes_out, dims=("time", "roi"))
            g.labels = resample_labels(g.labels, factor=upsample)

    if sigma:
        from scipy.ndimage import gaussian_filter1d
        g.spikes = gaussian_filter1d(g.spikes, sigma, axis=0)

        # Create images.
    for g in items.values():
        g.arr = np.zeros([len(g.spikes), frame_shape[0], frame_shape[1]])
        for i, c in enumerate(cells):
            mask = c.mask
            for t in range(len(g.spikes)):
                g.arr[t, mask.y, mask.x] = g.spikes[t, i]

    # Initialize figure and axes.
    n_plots = len(items)
    width = inches_per_plot * n_plots
    height = inches_per_plot
    fig = Figure(figsize=(width, height))
    fig.tight_layout(pad=0)
    count = 0
    for name, g in items.items():
        count += 1
        g.ax = ax = fig.add_subplot(1, n_plots, count)
        remove_ticks(g.ax)
        g.ax.set_title(g.name)
        g.im = g.ax.imshow(g.arr[0])
        g.smap = get_smap(data=g.spikes, qlim=qlim, cmap=cmap)

        fontdict = {'size': 16, 'color': 'white'}
        label_loc = [0.05, 0.95]
        g.lbl = ax.text(label_loc[0], label_loc[1], ' ',
                        fontdict=fontdict,
                        transform=ax.transAxes,
                        horizontalalignment='left',
                        verticalalignment='top',
                        usetex=False)

    # Render.
    n_frames = max([g.arr.shape[0] for g in items.values()])
    writer = FFMpegWriter(fps=fps)
    with writer.saving(fig, str(outfile), dpi):
        for i in range(n_frames):
            for g in items.values():
                if i >= g.arr.shape[0]:
                    continue
                cdata = g.smap(g.arr[i])
                cdata[:, :, -1] = 1
                g.im.set_data(cdata)
                g.lbl.set_text(g.labels[i])
                writer.grab_frame()


def make_heatmaps(
        s: Session,
        outfile: PathLike = "analysis/heatmaps.pdf",
        cmap: str = "inferno",
        qlim: Tuple[Number, Number] = (2.5, 97.5),
        sigma: Optional[Number] = None,
        upsample: Optional[int] = None,
        inches_per_plot: Number = 3,
        reverse_DCBA: bool = True,
        rank_by: Optional[Union[str, Sequence[str], Callable]] = "mean",
        descending: bool = True,
) -> None:
    outfile = Path(s.fs.getsyspath(outfile))

    iscell = s.segmentation["iscell"]
    spikes = s.segmentation["spikes"]
    spikes = spikes[:, iscell]

    # Group data, and average over trials.
    SPECS = get_grouper_specs(s, reverse_DCBA=reverse_DCBA, truncate_novT=True)
    grouper = Grouper(s.events)
    items = {}  # seq_name -> SimpleNamespace

    for name, spec in SPECS.items():
        # Group movies by element type, get trial average, and concatenate them.
        g = SimpleNamespace(name=name)
        lst = grouper.group_traces(spikes, spec)
        for i, obj in enumerate(lst):
            lst[i] = obj.mean("trial")
        g.spikes = xr.concat(lst, dim="time")

        # Ensure time is in zeroth axis.
        g.spikes = g.spikes.transpose("time", "roi")

        # Get time-series of labels.
        labels = []
        for elt in lst:
            labels.extend([elt.name] * len(elt))
        g.labels = simplify_labels(labels)
        g.spikes.coords['label'] = xr.DataArray(g.labels, dims=("time",))
        items[name] = g

    # Optionally upsample spike data.
    if upsample and upsample != 1:
        for g in items.values():
            n_frames_out = len(g.spikes) * upsample
            data = scipy.signal.resample(g.spikes, n_frames_out, axis=0)
            g.spikes = xr.DataArray(data, dims=("time", "roi"))
            g.labels = resample_labels(g.labels, factor=upsample)
            g.spikes.coords['label'] = xr.DataArray(g.labels, dims=("time",))

    # Optionally smooth spike data.
    if sigma:
        from scipy.ndimage import gaussian_filter1d
        data = gaussian_filter1d(g.spikes, sigma, axis=0)
        g.spikes = xr.DataArray(data, dims=("time", "roi"), coords=g.spikes.coords)

    # Rank traces.
    if rank_by:
        g = list(items.values())[0]
        roi_order = argranked(g.spikes, rank_by, descending=descending)
        for g in items.values():
            g.spikes = g.spikes.isel(roi=roi_order)

    # Put time in first axis to prepare for plotting.
    for g in items.values():
        g.spikes = g.spikes.transpose("roi", "time")

    # Initialize figure and axes.
    nrows, ncols = len(items), 2
    width, height = 6, inches_per_plot * nrows
    fig = Figure(figsize=(width, height))

    fig.tight_layout(pad=0)
    grid = fig.add_gridspec(nrows, ncols)

    # Plot ABCD first.
    g = ABCD = items.pop("ABCD")
    labels = g.labels
    g.ax = fig.add_subplot(grid[0, :])
    smap = get_smap(data=g.spikes, qlim=qlim, cmap=cmap)
    # cdata = smap(g.spikes)
    g.im = g.ax.imshow(
        g.spikes,
        cmap=smap.cmap,
        norm=smap.norm,
        aspect="auto",
        interpolation="hanning",

    )
    annotate_heatmap_axes(g.ax, g.name, labels)
    axes = fig.get_axes()
    g.cbar = fig.colorbar(g.im, ax=axes[-1])
    g.cbar.ax.tick_params(labelsize=8)

    for i, g in enumerate(items.values()):

        # Draw regular plot.
        g.ax1 = fig.add_subplot(grid[i + 1, 0])
        smap = get_smap(data=g.spikes, qlim=qlim, cmap=cmap)
        cdata = smap(g.spikes)
        g.ax1.imshow(cdata, aspect="auto", interpolation="hanning")
        annotate_heatmap_axes(g.ax1, g.name, labels)

        # Draw difference plot.
        g.ax2 = fig.add_subplot(grid[i + 1, 1])
        data = g.spikes - ABCD.spikes
        vmin, vmax = np.percentile(data, [2.5, 97.5])
        i
        if vmin < 0 and vmax <= 0:
            vmax = -vmin
        elif vmin < 0 and vmax >= 0:
            vmin = -vmax
        else:
            # vmin > 0, vmax > 0
            vmin = -vmax

        # g.im = g.ax2.imshow(cdata, aspect="auto", interpolation="hanning")
        g.norm = get_norm(vlim=[vmin, vmax])
        g.cmap = get_cmap("coolwarm")
        g.im = g.ax2.imshow(data, aspect="auto", cmap=g.cmap, norm=g.norm)
        # g.im.set_data(data)

        annotate_heatmap_axes(g.ax2, g.name + " - ABCD", labels)
        g.ax2.set_yticks([])

        axes = fig.get_axes()
        g.cbar = fig.colorbar(g.im, ax=axes[-1])
        g.cbar.ax.tick_params(labelsize=8)

    if nrows == 1:
        pass
    else:
        fig.subplots_adjust(hspace=0.3, wspace=0.05, top=0.95, bottom=0.05)

    fig.savefig(outfile)


def annotate_heatmap_axes(
        ax: "Axes",
        title: str,
        labels: Sequence,
) -> None:
    """

    """
    # Add title
    ax.set_title(title)

    # Add onset indicators.
    xticks = []
    for i, elt in enumerate(labels[1:]):
        if labels[i] != labels[i - 1]:
            xticks.append(i)
    xticklabels = [r'$\Delta$'] * len(xticks)
    xticklabels[-1] = "-"
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    for x in xticks:
        ax.axvline(x, color="white", ls="--")


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


def trace_plot():
    logging.getLogger("ca_analysis").setLevel(logging.INFO)

    s1 = open_session("611486-1", "2021-06-24", fs="ssd")
    s2 = open_session("611521-1", "2021-06-24", fs="ssd")
    schema = SCHEMA

    s = s1
    frames_df = s.events.tables["frames"]
    events_df = s.events.tables["events"]
    event_ids = events_df["event_id"].values

    # find matching transitions
    transitions = get_transitions()
    for T in transitions.values():
        T.fit(s)

    # sort/filter
    for T in transitions.values():
        data = T.data.copy()
        data = data.isel(trial=slice(-data.sizes["trial"] // 2, None))
        scores = data.mean("trial").std("time")
        sort_order = np.flipud(np.argsort(scores.data))
        n_rois = len(sort_order)
        data = T.data.isel(roi=sort_order[:n_rois // 2])
        T.data = data

    # bootstrap 2.5, 50, and 97.5 percentiles
    for T in transitions.values():
        n_trials = T.data.sizes["trial"]
        lows, mids, highs = [], [], []
        n_perms = 100
        for _ in range(n_perms):
            rand_inds = np.random.randint(0, n_trials, n_trials)
            rand = T.arr.isel(trial=rand_inds)
            rand = rand.mean("roi")
            a, b, c = np.percentile(rand, (2.5, 50, 97.5), axis=0)
            lows.append(a)
            mids.append(b)
            highs.append(c)

        T.ptiles = {}
        for name, lst in zip(("low", "mid", "high"), (lows, mids, highs)):
            arr = np.array(lst)  # average over trials
            trace = xr.DataArray(
                data.mean(axis=0),
                dims=("time",),
                coords={"event": T.arr.coords["event"]},
            )
            T.ptiles[name] = trace

    fig, axes = plt.subplots(5, 1, figsize=(8, 8))
    plot_specs = [
        {"axis": 0, "title": "P=0.9", "transitions": ["AB", "AC"]},
        {"axis": 1, "title": "P=0.8", "transitions": ["BC", "BD"]},
        {"axis": 2, "title": "P=0.7", "transitions": ["CD", "CE"]},
        {"axis": 3, "title": "P=0.6", "transitions": ["DE", "DA"]},
        {"axis": 4, "title": "P=0.5", "transitions": ["EA", "EB"]},
    ]
    colors = ["red", "blue"]
    alphas = [0.4, 0.4]

    for dct in plot_specs:
        ax = axes[dct["axis"]]
        for i, T_name in enumerate(dct["transitions"]):
            T = transitions[T_name]
            low, mid, high = T.ptiles["low"], T.ptiles["mid"], T.ptiles["high"]
            X = np.arange(len(mid))
            ax.fill_between(X, low, high, color=colors[i], alpha=alphas[i])
            ax.plot(mid, color=colors[i], alpha=1)
        labels = [int(ev) for ev in mid.coords["event"]]

        onset = np.argwhere(np.ediff1d(labels, to_begin=0) != 0)[0][0] + 1
        xticklabels = [""] * len(mid)
        xticklabels[onset] = r"$\Delta$"
        ax.set_xticklabels(xticklabels)
        ax.set_title(dct["title"])

    ymax = np.max([ax.get_ylim()[1] for ax in axes])
    for ax in axes:
        ax.set_ylim([0, ymax])


def plot_heatmap_and_histogram(s: Session):
    schema = s.events.schema

    # construct transitions
    # transitions = get_transitions(s)

    # construct trial-averaged arrays
    for T in s.transitions.values():
        data = T.arr
        ntrials = data.sizes["trial"]
        # data = data.isel(trial=slice(0, 10))  # front third
        # data = data.isel(trial=slice(0, ntrials // 3))  # front third
        # data = data.isel(trial=slice(ntrials // 3, 2* ntrials // 3))  # middle third
        data = data.isel(trial=slice(2 * ntrials // 3, None))  # back third
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
    fig.savefig(Path.home() / "heatmaps.pdf")

    plt.rcParams.update({"font.size": 8})
    fig = plt.figure(figsize=(6, 4))
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
    ax.set_xlim([-7.5, 7.5])
    ax.set_xlabel('$r$')
    ax.set_ylabel('density')
    plt.show()
    fig.savefig(Path.home() / "coeffs.png")

    import scipy.stats
    print("Rank Sum Tests (nov. excess ratios)\n------------------------")
    chance_scores = rows[-1]["scores"]
    for i in range(4):
        test_scores = rows[i]["scores"]
        pvalue = scipy.stats.ranksums(test_scores, chance_scores).pvalue
        name = rows[i]["diff"].name
        print(f"{name}: pvalue = {pvalue}")

    for i, row in enumerate(rows):
        scores = row["scores"]
        sc = scores.mean()
        print(f"{row['diff'].name} : {sc}")

    """
    annotate_heatmap_axes(g.ax, g.name, labels)
    axes = fig.get_axes()
    g.cbar = fig.colorbar(g.im, ax=axes[-1])
    g.cbar.ax.tick_params(labelsize=8)

    for i, g in enumerate(items.values()):

        # Draw regular plot.
        g.ax1 = fig.add_subplot(gs[i+1, 0])
        smap = get_smap(data=g.spikes, qlim=qlim, cmap=cmap)
        cdata = smap(g.spikes)
        g.ax1.imshow(cdata, aspect="auto", interpolation="hanning")
        annotate_heatmap_axes(g.ax1, g.name, labels)

        # Draw difference plot.
        g.ax2 = fig.add_subplot(gs[i+1, 1])
        data = g.spikes - ABCD.spikes
        vmin, vmax = np.percentile(data, [2.5, 97.5])
        i
        if vmin < 0 and vmax <= 0:
            vmax = -vmin
        elif vmin < 0 and vmax >= 0:
            vmin = -vmax
        else:
            #vmin > 0, vmax > 0
            vmin = -vmax

        #g.im = g.ax2.imshow(cdata, aspect="auto", interpolation="hanning")
        g.norm = get_norm(vlim=[vmin, vmax])
        g.cmap = get_cmap("coolwarm")
        g.im = g.ax2.imshow(data, aspect="auto", cmap=g.cmap, norm=g.norm)
        #g.im.set_data(data)

        annotate_heatmap_axes(g.ax2, g.name + " - ABCD", labels)
        g.ax2.set_yticks([])

        axes = fig.get_axes()
        g.cbar = fig.colorbar(g.im, ax=axes[-1])
        g.cbar.ax.tick_params(labelsize=8)


    if nrows == 1:
        pass
    else:
        fig.subplots_adjust(hspace=0.3, wspace=0.05, top=0.95, bottom=0.05)

    fig, axes = plt.subplots(1, 5, figsize=(8, 8))
    plot_specs = [
        {"axis": 0, "title": "P=0.9", "transitions": ["AB", "AC"]},
        {"axis": 1, "title": "P=0.8", "transitions": ["BC", "BD"]},
        {"axis": 2, "title": "P=0.7", "transitions": ["CD", "CE"]},
        {"axis": 3, "title": "P=0.6", "transitions": ["DE", "DA"]},
        {"axis": 4, "title": "P=0.5", "transitions": ["EA", "EB"]},
    ]
    colors = ["red", "blue"]
    alphas = [0.4, 0.4]

    for dct in plot_specs:
        ax = axes[dct["axis"]]
        for i, T_name in enumerate(dct["transitions"]):
            T = transitions[T_name]

            X = np.arange(len(mid))
            ax.fill_between(X, low, high, color=colors[i], alpha=alphas[i])
            ax.plot(mid, color=colors[i], alpha=1)
        labels = [int(ev) for ev in mid.coords["event"]]

        onset = np.argwhere(np.ediff1d(labels, to_begin=0) != 0)[0][0] + 1
        xticklabels = [""] * len(mid)
        xticklabels[onset] = r"$\Delta$"
        ax.set_xticklabels(xticklabels)
        ax.set_title(dct["title"])

    ymax = np.max([ax.get_ylim()[1] for ax in axes])
    for ax in axes:
        ax.set_ylim([0, ymax])

    """


def pooling_sandbox():
    sessions = [
        open_session("611486-1", "2021-07-13", fs="ssd"),
        open_session("611521-1", "2021-07-14", fs="ssd"),
        open_session("619739-1", "2021-08-03", fs="ssd"),
        open_session("619745-2", "2021-10-19", fs="ssd"),
        open_session("619766-4", "2021-10-19", fs="ssd"),
        open_session("619816-2", "2021-11-02", fs="ssd"),
    ]

    # get trial-averaged responses for each transition, across sessions.
    # put result in Transition object as Transition.lst.
    pool = get_transitions()
    for p in pool.values():
        p.data = []

    for s in sessions:
        # filter 1 here: i.e., remove cells with low firing rates by
        # settings s.cells[i].iscell = False.

        # collect transitions for remaining cells.
        s.transitions = get_transitions(s)

        # get trial-averaged response trajectories for each transition, and add to master list.
        for tname, t in s.transitions.items():
            pool[tname].data.append(t.arr.mean("trial"))

    # concatenate master lists.
    for p in pool.values():
        p.data = xr.concat(p.data, dim="roi")  # this is n_timepoints x n_rois.

    specs = [("BC", "AC"), ("CD", "BD"), ("DE", "CE"), ("EA", "DA"), ("AB", "EB")]
    # specs = [
    #     {"high_name": "BC", "low_name": "AC", "P"}
    # ]
    chunklen = 6
    info = []
    for i, sp in enumerate(specs):
        d = {"name": f"{sp[0]} v {sp[1]}"}

        high = pool[sp[0]].data
        a1, b1 = get_splits(high)
        d["HP"] = b1.mean("time")

        low = pool[sp[1]].data
        a2, b2 = get_splits(low)
        d["LP"] = b2.mean("time")

        d["diff"] = d["LP"] - d["HP"]
        info.append(d)

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["black", "red", "teal", "blue", "gray"]
    linestyles = ["-", "-", "-", "-", "--"]
    for i, d in enumerate(info):
        X, Y = gaussian_kde(d["diff"])
        label = d["name"]
        ax.plot(X, Y, color=colors[i], ls=linestyles[i], label=label)
    ax.legend()
    ax.set_xlim([-5, 5])
    plt.show()
    # for s in sessions:
    #     s.T = get_transitions(s)
    #
    # AC, BC = Transition(["A", "C"], P=0.1), Transition(["B", "C"], P=0.8)
    # AC_arrays, BC_arrays = [], []
    #
    # for s in sessions:
    #     AC.parse(s), BC.parse(s)
    #     AC_arrays.append(AC.arr)
    #     BC_arrays.append(BC.arr)
    #
    # # Restrict to particular trials.
    # for lst in (AC_arrays, BC_arrays):
    #     for i in range(len(lst)):
    #         arr = lst[i]
    #         n_trials = arr.sizes["trial"]
    #         lst[i] = arr.isel(trial=slice(2 * n_trials // 3, None))
    #
    # # Get trial-averaged responses
    # for lst in (AC_arrays, BC_arrays):
    #     for i in range(len(lst)):
    #         lst[i] = lst[i].mean("trial")
    #
    # AC.clear()
    # AC._arr = xr.concat(AC_arrays, dim="roi")
    # AC._splits = get_splits(AC.arr)
    #
    # BC.clear()
    # BC._arr = xr.concat(BC_arrays, dim="roi")
    # BC._splits = get_splits(BC.arr)
    #
    # C_hp = BC.splits[1]
    # C_lp = AC.splits[1]
    #
    # fig, ax = plt.subplots()
    #
    # vals = C_hp.mean("time")
    # X, Y = gaussian_kde(vals, np.linspace(0, 20, 1000))
    # ax.plot(X, Y, color="black", label="B(C)")
    #
    # vals = C_lp.mean("time")
    # X, Y = gaussian_kde(vals, np.linspace(0, 20, 1000))
    # ax.plot(X, Y, color="red", label="A(C)")
    #
    # ax.legend()
    # ax.set_xlim([0, 10])
    # plt.show()
    #


if __name__ == "__main__":
    import yaml

    info_path = Path(__file__).parent / "info.yaml"
    d = yaml.load(open(info_path, "r"), Loader=yaml.Loader)

    #
    # sessions = [
    #     open_session("611486-1", "2021-07-13", fs="ssd"),
    #     open_session("611521-1", "2021-07-14", fs="ssd"),
    #     open_session("619739-1", "2021-08-03", fs="ssd"),
    #     open_session("619745-2", "2021-10-19", fs="ssd"),
    #     open_session("619766-4", "2021-10-19", fs="ssd"),
    #     open_session("619816-2", "2021-11-02", fs="ssd"),
    # ]
    #
    # s = sessions[0]
    # frames = s.events.tables["frames"]
    # spikes = s.cells.spikes
    # length = min(len(frames), spikes.sizes["time"])
    # frames = frames.iloc[:length]
    # spikes = spikes.isel(time=slice(0, length))
    #
    # event_ids = frames["event_id"].values
    # arrays = {}
    # for ev_id in range(1, 7):
    #     arr = np.zeros(length, dtype=int)
    #     arr[event_ids == ev_id] = 1
    #     arrays[ev_id] = arr
    # X = pd.DataFrame(arrays)
    #
    # mod = LinearRegression()
    # n_rois = spikes.sizes["roi"]
    # coefs = np.zeros([n_rois, 6])
    # scores = np.zeros(n_rois)
    # for i in range(n_rois):
    #     y = spikes[:, i]
    #     mod.fit(X, y)
    #     coefs[i] = mod.coef_
    #     scores[i] = mod.score(X, y)
    #
    #
    #
