import copy
import datetime
import json

import multiprocessing
# multiprocessing.set_start_method('forkserver')  #?

import os
from pathlib import Path
import shutil

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
import fs as pyfs
import fs.errors

import h5py
import napari
from jinja2 import Template
import ndindex as nd
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.animation import FFMpegWriter
import matplotlib.colors as mpc
from matplotlib.figure import Figure
from ca_analysis.plot import get_smap
import matplotlib.pyplot as plt
import numpy as np

from statsmodels.distributions.empirical_distribution import ECDF
import xarray as xr
import yaml

from ca_analysis import *
from ca_analysis.indexing import *
from ca_analysis.io import *

from ca_analysis.processing.motion_correction import run_motion_correction
from ca_analysis.stats import *
from ca_analysis.persistence import *
from ca_analysis.plot import *
from ca_analysis.resampling import *

from main import *
from processing import *


def write_movie(
    outfile: PathLike,
    mov: ArrayLike,
    labels: Optional[ArrayLike] = None,
    fps: Number = 30,
    dpi: int = 190,
    figsize=(4, 4),
    frameon: bool = False,
    facecolor: str = 'black',
    cmap: str = "inferno",
    qlim: Sequence = (0.01, 99.99),
) -> None:

    # Initialize figure.
    fig = Figure(
        figsize=figsize,
        frameon=frameon,
        facecolor=facecolor,
    )
    ax = fig.add_subplot(1, 1, 1, xmargin=0, ymargin=0)
    ax.set_aspect("equal")
    ax.axis('off')
    fig.tight_layout(pad=0)
    im = ax.imshow(np.zeros_like(mov[0]))

    # Optionally initialize label
    if labels is not None:
        fontdict = {'size': 16, 'color': 'white'}
        label_loc = [0.05, 0.95]
        label_obj = ax.text(
            label_loc[0],
            label_loc[1],
            ' ',
            fontdict=fontdict,
            transform=ax.transAxes,
            horizontalalignment='left',
            verticalalignment='top',
            usetex=False,
        )

    # set up colormap
    smap = get_smap(data=mov, qlim=qlim, cmap=cmap)

    # write movie
    writer = FFMpegWriter(fps=fps)
    with writer.saving(fig, str(outfile), dpi):
        for i in range(mov.shape[0]):
            fm = smap(mov[i])
            fm[:, :, -1] = 1
            im.set_data(fm)
            if labels is not None:
                label_obj.set_text(labels[i])
            writer.grab_frame()




# s = open_session('61102-1', '2023-02-03', '1')
s = open_session('61107-1', '2023-01-27', '1', fs=0)
schema = s.events.schema


# spks = s.segmentation['spikes']
# plt.plot(spks.mean('roi'))
# plt.show()
# s.play_movie()


tlim = (2000, 2500)
outfile = Path.home() / 'progress_report/mov.mp4'
tempfile = Path.home() / 'progress_report/temp.h5'
motion_correct = False
use_corrected = True

if motion_correct:
    mov = s.mov.data.isel(time=slice(tlim[0], tlim[1])).compute()
    run_motion_correction(
        mov.data,
        out=tempfile,
        single_thread=False,
        smooth=False,
    )
    with h5py.File(tempfile, 'r') as f:
        data = f['data'][:]
    mov = xr.DataArray(data, dims=('time', 'y', 'x'))
elif use_corrected:
    with h5py.File(tempfile, 'r') as f:
        data = f['data'][:]
    mov = xr.DataArray(data, dims=('time', 'y', 'x'))
else:
    mov = s.mov.data.isel(time=slice(tlim[0], tlim[1])).compute()

events = s.events['frames']['event'].values[tlim[0]: tlim[1]]
events = np.array([schema.get_event(ev) for ev in events], dtype=object)
labels = events.astype(str)

mov.data = gaussian_filter(mov, (0.75, 0.75, 0.75))
mov.coords['event'] = xr.DataArray(events, dims=('time',))

write_movie(
    outfile,
    mov,
    labels,
    cmap='gray',
    qlim=(0.1, 99.9),
)

# ---------------
# Write frames
