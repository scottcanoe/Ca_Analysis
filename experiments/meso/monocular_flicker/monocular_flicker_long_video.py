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
outfile = 'video/M110_monocular_flicker_long.mp4'

fps = 10
qlim = (2.5, 97.5)
cmap = "inferno"
dpi: Number = 190
width: Number = 4
frameon: bool = True
facecolor: "ColorLike" = "white"
cmap = "hot"
lpad = 10
rpad = 10
kernel = (1, 2, 2)

with h5py.File(s.fs.getsyspath('mov.h5'), 'r') as f:
    mov = f['ach'][:1000]
    mov = xr.DataArray(mov, dims=('time', 'y', 'x'))

mov.data = gaussian_filter(mov.data, kernel)
mov.coords['event'] = xr.DataArray(
    s.events['frames']['event'].values[:1000],
    dims=('time',)
)

# baseline = mov.isel(time=slice(0, 50)).mean('time')
# for i in range(mov.sizes['time']):
#     mov.data[i] = mov.data[i] - baseline.data
#
# scale = 0.04393532
# for i in range(mov.sizes['time']):
#     mov.data[i] = mov.data[i] / scale

# add cortex mask
print('adding cortex mask')
bitmask = np.zeros(mov.shape[1:], dtype=bool)
with h5py.File(s.fs.getsyspath('rois.h5'), 'r') as f:
    x = f['LH']['x'][:]
    y = f['LH']['y'][:]
    bitmask[y, x] = True
    x = f['RH']['x'][:]
    y = f['RH']['y'][:]
    bitmask[y, x] = True
YX = np.argwhere(~bitmask)
Y, X = YX[:, 0], YX[:, 1]
for i in range(mov.sizes['time']):
    mov.data[i, Y, X] = np.nan

# normalize
mov = mov[50:]

# mmean = np.nanmean(mov)
# for i in range(mov.sizes['time']):
#     mov.data[i] = mov.data[i] - mmean
#
# mmin, mmax = np.nanmin(mov), np.nanmax(mov)
# scale = mmax - mmin
# for i in range(mov.sizes['time']):
#     mov.data[i] = mov.data[i] / scale

items = [dict(data=mov, name='mov')]

# Initialize figure.
ypix, xpix = items[0]['data'][0].shape
aspect = ypix / xpix
figsize = (width, width * aspect)
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
    vlim = np.nanpercentile(mov, [20, 99])
    # vlim = 0.05, 0.05
    dct['smap'] = get_smap(data=data, vlim=vlim, cmap=cmap)
    dct['ax'] = ax = fig.add_subplot(1, 1, i + 1, xmargin=0, ymargin=0)
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_title("monocular flicker")
    dct['im'] = ax.imshow(np.zeros_like(data[0]))
    dct['left label'] = ax.text(
        label_loc[0],
        label_loc[1],
        ' ',
        fontdict=fontdict,
        transform=ax.transAxes,
        horizontalalignment='left',
        verticalalignment='top',
        usetex=False,
    )
    dct['right label'] = ax.text(
        0.95,
        0.95,
        ' ',
        fontdict=fontdict,
        transform=ax.transAxes,
        horizontalalignment='right',
        verticalalignment='top',
        usetex=False,
    )

fig.tight_layout(pad=1)


# ---------------
# Write frames
fps = s.attrs["samplerate"] if fps is None else fps
writer = FFMpegWriter(fps=fps)
with writer.saving(fig, str(outfile), dpi):
    for i in range(mov.sizes['time']):
        for dct in items:
            frame = dct['data'].isel(time=i)
            cdata = dct['smap'](frame)
            cdata[:, :, -1] = 1
            dct['im'].set_data(cdata)
            ev = mov.coords['event'][i].item()
            if ev == 0:
                dct['left label'].set_text("")
                dct['right label'].set_text("")
            elif ev == 1:
                dct['left label'].set_text("LEFT")
            else:
                dct['right label'].set_text("RIGHT")

        writer.grab_frame()

