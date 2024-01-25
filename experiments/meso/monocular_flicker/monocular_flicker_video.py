import intake
from intake.source.discovery import drivers, load_plugins_from_module
import weakref
from time import perf_counter as clock
from typing import Mapping

import dask.array as da
import h5py
import matlab.engine
import matplotlib
# matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.animation import FFMpegWriter
import napari
import numpy as np
import pandas as pd
from scipy.io import loadmat

from experiments.meso.main import *
# from processing import *
from ca_analysis.plot import get_smap


def get_trials(s: Session) -> xr.DataArray:
    if s.fs.exists('trials.h5'):
        with h5py.File(s.fs.getsyspath('trials.h5'), 'r') as f:
            left = xr.DataArray(f['left'][:], dims=('trial', 'time', 'y', 'x'))
            right = f['right'][:]
            return xr.DataArray(data, dims=('trial', 'time', 'y', 'x'))
    trials = s.ach_mov.split(1)
    with h5py.File(s.fs.getsyspath('trials.h5'), 'w') as f:
        f.create_dataset('data', data=trials.data)
    return trials


s = open_session('M110', '2023-04-21', '2', fs=0)
outfile = 'video/monocular_flicker.mp4'

fps = 10
qlim = (2.5, 97.5)
cmap = "inferno"
dpi: Number = 190
width: Number = 4
frameon: bool = True
facecolor: "ColorLike" = "white"
cmap = "inferno"
lpad = 10
rpad = 10
kernel = (1.5, 2, 2)

left = s.ach_mov.split(1, lpad=lpad, rpad=rpad, concat=True)
print('got left')
right = s.ach_mov.split(2, lpad=lpad, rpad=rpad, concat=True)
print('got right')

left = left.mean('trial')
right = right.mean('trial')


left.data = gaussian_filter(left, kernel)
right.data = gaussian_filter(right, kernel)

items = [dict(data=left, name='left'), dict(data=right, name='right')]

# Figure out fps.
if fps is None:
    fps = s.attrs["samplerate"]

# Initialize figure.
ypix, xpix = items[0]['data'][0].shape
aspect = ypix / xpix
figsize = (2 * width, width * aspect)
fig = Figure(
    figsize=figsize,
    frameon=frameon,
    facecolor=facecolor,
)

# Setup normalization + colormapping pipelinex, axes, and labels
fontdict = {'size': 16, 'color': 'white'}
label_loc = [0.05, 0.95]
for i, dct in enumerate(items):
    data = dct['data']
    dct['smap'] = get_smap(data=data, qlim=qlim, cmap=cmap)
    dct['ax'] = ax = fig.add_subplot(1, 2, i + 1, xmargin=0, ymargin=0)
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_title(dct['name'] + " flicker")
    dct['im'] = ax.imshow(np.zeros_like(data[0]))
    dct['label'] = ax.text(
        label_loc[0],
        label_loc[1],
        ' ',
        fontdict=fontdict,
        transform=ax.transAxes,
        horizontalalignment='left',
        verticalalignment='top',
        usetex=False,
    )


fig.tight_layout(pad=1)


# ---------------
# Write frames
n_frames = items[0]['data'].shape[0]
writer = FFMpegWriter(fps=fps)
with writer.saving(fig, str(outfile), dpi):
    for i in range(n_frames):
        for dct in items:
            frame = dct['data'].isel(time=i)
            cdata = dct['smap'](frame)
            cdata[:, :, -1] = 1
            dct['im'].set_data(cdata)
            string = frame.coords['event'].item().name
            if string != "":
                dct['label'].set_text("ON")
            else:
                dct['label'].set_text("")
        writer.grab_frame()

