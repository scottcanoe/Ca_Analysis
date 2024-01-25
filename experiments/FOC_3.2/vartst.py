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
from statsmodels.distributions.empirical_distribution import ECDF
import toolz
import xarray as xr

from ca_analysis import *
from ca_analysis.stats import *

from main import *
from processing import *


def plot_PDFs(P):

    pair_to_style = {
        "A": dict(color="orange", ls="-", label="A"),
        "B": dict(color="gray", ls="--", label="B"),
        "C": dict(color="black", ls="-", label="C"),
        "D": dict(color="blue", ls="-", label="D"),
        "E": dict(color="red", ls="-", label="E"),
    }
    xlim = [-3, 3]
    X = np.linspace(xlim[0], xlim[1], 1000)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)

    for key in ["C", "D", "E", "A", "B"]:
        p = P[key]

        # plot PDFs
        arr = p.scores
        x, y = gaussian_kde(arr, X)
        style = pair_to_style[p.name]
        ax.plot(x, y, **style)

    ax.legend()
    ax.set_xlim(xlim)
    # ax.set_ylim([0, 0.95])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax.set_ylabel('density')
    ax.set_xlabel('$\Delta$')
    fig.tight_layout()
    PLOTDIR = Path.home() / "plots"
    # fig.savefig(PLOTDIR / "normal.pdf")
    # fig.savefig(PLOTDIR / "pdfs.eps")
    plt.show()
    return fig


@ensure_iterable_first_argument
def make_transition_scores(
    ses: Iterable[Session],
    score_fn: Callable = lambda pre, post: post,
    blocks: Optional[IndexLike] = [2, 3, 4],
    pre_sel: Optional[IndexLike] = None,
    post_sel: Optional[IndexLike] = None,
    name: Optional = None,
) -> Mapping[str, Transition]:

    block_filter = GetBlocks(blocks)
    pre_funcs = [block_filter, ISel(pre_sel), Mean('time'), Mean('trial')]
    post_funcs = [block_filter, ISel(post_sel), Mean('time'), Mean('trial')]
    ok_names = handle_transition_names(name)

    # make scores for each transition
    for s in ses:
        for key in ok_names:
            tr = s.data.transitions[key]
            pre = toolz.pipe(tr.pre, *pre_funcs)
            post = toolz.pipe(tr.post, *post_funcs)
            tr.scores = score_fn(pre, post)
            tr.n = tr.pre.sizes["trial"]

    # pool them
    T = get_transitions()
    for key in T.keys():
        if key in ok_names:
            lst = []
            n = 0
            for s in ses:
                lst.append(s.data.transitions[key].scores)
                n += s.data.transitions[key].n
            T[key].scores = xr.concat(lst, "roi")
            T[key].n = n
        else:
            del T[key]
    return T


@ensure_iterable_first_argument
def make_transition_pair_scores(
    ses: Iterable[Session],
    score_fn: Callable = lambda high, low: low - high,
    name: Optional = None,
    T = None,
) -> Mapping[str, TransitionPair]:

    ok_names = handle_transition_pair_names(name)

    # make scores for each transition pair
    for s in ses:
        for key in ok_names:
            p = s.data.transition_pairs[key]
            p.scores = score_fn(p.high.scores, p.low.scores)

    # pool them
    P = get_transition_pairs()
    for key in P.keys():
        p = P[key]
        if key in ok_names:
            lst = []
            for s in ses:
                lst.append(s.data.transition_pairs[key].scores)
            p.scores = xr.concat(lst, "roi")
            if T:
                p.high.n = T[p.high.name].n
                p.low.n = T[p.low.name].n
        else:
            del P[key]
    return P



"""
old shuffle

        # # find biggest dataset
        # pre_max, post_max, n_max = None, None, 0
        # for key, tr in s.data.transitions.items():
        #     if tr.n > n_max:
        #         n_max = tr.n
        #         pre_max = tr.pre
        #         post_max = tr.post

        # replace existing datasets with sub-sampled version of biggest
        for key, tr in s.data.transitions.items():
            tr.pre_old = tr.pre
            tr.post_old = tr.post
            tr.pre = pre_max
            tr.post = post_max
            if tr.n_trials < n_max:
                inds = np.random.choice(n_max, tr.n_trials, replace=False)
                tr.pre = tr.pre.isel(trial=inds)
                tr.post = tr.post.isel(trial=inds)
    .
    .
    .
    
    # after plotting
        if randomize:
        for key, tr in s.data.transitions.items():
            tr.pre = tr.pre_old
            tr.post = tr.post_old
"""


def print_sizes(P):
    for key in "ABCDE":
        p = P[key]
        n = p.low.n
        print(f"{p.name}: {n}")


@ensure_iterable_first_argument
def rotate_events(ses):
    for s in ses:
        s.data._ensure_prepared()
        shift = np.random.randint(1, 10)
        df = s.data.attrs["event_table"]
        s.data.attrs["event_table_old"] = df.copy()
        event_ids = np.roll(df["event_id"].values, shift)
        df["event_id"] = event_ids
        s.data.attrs["event_table"] = df


@ensure_iterable_first_argument
def undo_rotate_events(ses):

    for s in ses:
        s.data.attrs["event_table"] = s.data.attrs["event_table_old"]


post_score_fn = lambda pre, post: post
jump_score_fn = lambda pre, post: post - pre

score_fn = jump_score_fn

def rotation_rand():
    sessions = get_sessions()
    ses = sessions

    for s in sessions:
        s.data.clear()

    rotate_events(ses)
    T = make_transition_scores(ses, score_fn=jump_score_fn)
    P = make_transition_pair_scores(ses, T=T)
    plot_PDFs(P)
    undo_rotate_events(ses)


def shuffle_rand():
    sessions = get_sessions()
    ses = sessions

    for s in sessions:
        s.data.clear()

    # - this is for shuffling events
    T = make_transition_scores(ses, score_fn=jump_score_fn)
    scores = []
    for tr in T.values():
        scores.append(tr.scores)
    scores = xr.concat(scores, "roi")
    for tr in T.values():
        inds = np.random.choice(len(scores), size=tr.n)
        tr.scores = scores.isel(roi=inds)

    P = make_transition_pair_scores(ses, T=T)
    plot_PDFs(P)


if __name__ == "__main__":

    sessions = get_sessions()
    ses = sessions
    s = ses[0]
    randomize_events = True

    rotation_rand()
    shuffle_rand()



