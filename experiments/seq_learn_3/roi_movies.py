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

from ca_analysis.plot import *
from ca_analysis.plot.images import *
from ca_analysis.resampling import *

from main import *
from processing import *


# s = open_session('61102-1', '2023-02-03', '1')
s = open_session('61107-1', '2023-01-27', '1', fs=0)
schema = s.events.schema

tlim = (1950, 2300)
outfile = Path.home() / 'progress_report/rois.mp4'
tempfile = Path.home() / 'progress_report/temp.h5'

mov = s.mov.data.isel(time=slice(tlim[0], tlim[1])).compute()
bg = mov.mean('time')
bg = bg / bg.max()

spikes = s.spikes.data.isel(time=slice(tlim[0], tlim[1])).compute()
events = s.events['frames']['event'].values[tlim[0]: tlim[1]]
events = np.array([schema.get_event(ev) for ev in events], dtype=object)
# spikes.coords['event'] = xr.DataArray(events, dims=('time',))
labels = events.astype(str)

roi_ids = spikes.coords['roi'].data
stat = s.segmentation['stat']
rois = []
for i, rid in enumerate(roi_ids):
    dct = stat[rid]
    obj = SimpleNamespace(roi=rid)
    obj.ypix = dct['ypix']
    obj.xpix = dct['xpix']
    obj.color = np.random.random(4)
    spks = spikes.sel(roi=rid)
    del spks['roi']
    spks = resample1d(spks, 'time', factor=2)
    obj.spikes = spks
    obj.alpha = obj.spikes / obj.spikes.max()
    rois.append(obj)


# Initialize figure.
fig = Figure(
    figsize=(4, 4),
    frameon=False,
    facecolor='black',
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


# write movie
n_timepoints = rois[0].spikes.sizes['time']
template = np.zeros([mov.sizes['y'], mov.sizes['x'], 4])
# template[:, :, 0] = bg
# template[:, :, 1] = bg
# template[:, :, 2] = bg
template[:, :, 3] = 1
template = gray_to_rgba(bg, alpha=1, dtype=np.float64)
writer = FFMpegWriter(fps=15)
with writer.saving(fig, str(outfile), 190):
    for i in range(mov.shape[0]):
        mat = template.copy()
        for obj in rois:
            color = obj.color
            alpha = np.clip(obj.alpha[i].item(), 0, 1)
            # alpha = np.clip(obj.spikes[i], 0, 1)
            blend_pixels(mat, obj.ypix, obj.xpix, obj.color, alpha=alpha, out=mat)
            # mat[obj.ypix, obj.xpix] = color
        im.set_data(mat)
        if labels is not None:
            label_obj.set_text(labels[i])
        writer.grab_frame()
