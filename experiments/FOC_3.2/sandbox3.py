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
            tr.n = pre.sizes["trial"]

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
) -> Mapping[str, TransitionPair]:

    ok_names = handle_transition_pair_names(name)

    # make scores for each transition pair
    for s in ses:
        for key in ok_names:
            p = s.data.transition_pairs[key]
            p.scores = score_fn(p.high.scores, p.low.scores)
            p.n = p.high.n + p.low.n

    # pool them
    P = get_transition_pairs()
    for key in P.keys():
        if key in ok_names:
            lst = []
            n = 0
            for s in ses:
                lst.append(s.data.transition_pairs[key].scores)
                n += s.data.transition_pairs[key].n
            P[key].scores = xr.concat(lst, "roi")
            P[key].n = n
        else:
            del P[key]
    return P

def plot_blockwise():

    logger.setLevel(logging.INFO)
    sessions = get_sessions("3.2")
    # if sessions[0].data.transitions["AC"].data[0] is None:
    #     load_spikes(sessions, block=slice(None))
    ses = sessions
    s = ses[0]
    ses = ses[0:1]
    T = make_transition_scores(ses)
    P = make_transition_pair_scores(ses)
    sizes = {tr.name: len(tr.scores) for tr in T.values()}

    score_fn = lambda pre, post: post - pre
    score_fn = lambda pre, post: post

    pair_to_style = {
        "A": dict(color="orange", ls="-", label="$\Delta P=0.2$"),
        "B": dict(color="gray", ls="--", label="$\Delta P=0.0$"),
        "C": dict(color="black", ls="-", label="$\Delta P=0.8$"),
        "D": dict(color="blue", ls="-", label="$\Delta P=0.6$"),
        "E": dict(color="red", ls="-", label="$\Delta P=0.4$"),
    }
    xlim = [-3, 3]
    X = np.linspace(xlim[0], xlim[1], 1000)

    fig = plt.figure(figsize=(5, 10))
    axes = [fig.add_subplot(4, 1, i + 1) for i in range(4)]
    for plot_num, block in enumerate([2, 3, 4, [2, 3, 4]]):
        ax = axes[plot_num]
        T = make_transition_scores(ses, blocks=block, score_fn=score_fn)
        P = make_transition_pair_scores(ses)

        for key, p in P.items():

            # plot curves
            arr = p.scores
            x, y = gaussian_kde(arr, X)
            style = pair_to_style[p.name]
            ax.plot(x, y, **style)

            # etc.
            title = f"block: {block - 1}" if is_int(block) else "all blocks"
            ax.set_title(title)
            ax.legend()
            ax.set_xlim(xlim)
            # ax.set_ylim([0, 0.95])
            ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
            ax.set_ylabel('density')
            ax.set_xlabel('$\Delta$')

    fig.tight_layout()
    PLOTDIR = Path.home() / "plots/cosyne"
    # fig.savefig(PLOTDIR / "pdfs.pdf")
    # fig.savefig(PLOTDIR / "pdfs.eps")
    plt.show()


SIZES = {
    'AB': 461,
    'AC': 52,
    'BC': 575,
    'BD': 162,
    'CD': 425,
    'CE': 201,
    'DE': 334,
    'DA': 252,
    'EA': 258,
    'EB': 276,
}


if __name__ == "__main__":

    logger.setLevel(logging.INFO)
    sessions = get_sessions("3.2")
    # if sessions[0].data.transitions["AC"].data[0] is None:
    #     load_spikes(sessions, block=slice(None))
    ses = sessions
    s = ses[0]
    ses = ses[0:1]
    T = make_transition_scores(ses)
    P = make_transition_pair_scores(ses)

    sizes = {tr.name: tr.pre.sizes["trial"] for tr in s.data.transitions.values()}

    pair_to_style = {
        "A": dict(color="orange", ls="-", label="$A: \Delta P=0.2$"),
        "B": dict(color="gray", ls="--", label="B: $\Delta P=0.0$"),
        "C": dict(color="black", ls="-", label="C: $\Delta P=0.8$"),
        "D": dict(color="blue", ls="-", label="$D: \Delta P=0.6$"),
        "E": dict(color="red", ls="-", label="$E: \Delta P=0.4$"),
    }
    xlim = [-3, 3]
    X = np.linspace(xlim[0], xlim[1], 1000)

    fig = plt.figure(figsize=(5, 10))
    axes = [fig.add_subplot(4, 1, i + 1) for i in range(4)]
    for plot_num, block in enumerate([2, 3, 4, [2, 3, 4]]):
        ax = axes[plot_num]
        T = make_transition_scores(ses, blocks=block, score_fn=score_fn)
        P = make_transition_pair_scores(ses)

        for key, p in P.items():

            # plot curves
            arr = p.scores
            x, y = gaussian_kde(arr, X)
            style = pair_to_style[p.name]
            ax.plot(x, y, **style)

            # etc.
            title = f"block: {block - 1}" if is_int(block) else "all blocks"
            ax.set_title(title)
            ax.legend()
            ax.set_xlim(xlim)
            # ax.set_ylim([0, 0.95])
            ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
            ax.set_ylabel('density')
            ax.set_xlabel('$\Delta$')

    fig.tight_layout()
    PLOTDIR = Path.home() / "plots/cosyne"
    # fig.savefig(PLOTDIR / "pdfs.pdf")
    # fig.savefig(PLOTDIR / "pdfs.eps")
    plt.show()


