import intake
from intake.source.discovery import drivers, load_plugins_from_module
import weakref
from time import perf_counter as clock
from typing import Mapping, Tuple

import dask.array as da
import h5py
import matlab.engine
import matplotlib
# matplotlib.use('TkAgg')

from matplotlib.animation import FFMpegWriter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

import napari
import numpy as np
import pandas as pd
from scipy.io import loadmat

from experiments.meso.main import *
# from processing import *
from ca_analysis.plot import get_smap


custom_cmap = LinearSegmentedColormap.from_list(
    'blackredblue',
    [(0, 0, 1), (0, 0, 0), (1, 0, 0)],
    N=256,
)


def save_trials(s: Session) -> None:
    left = s.ach_mov.split(1, lpad=lpad, rpad=rpad, concat=True)
    print('got left')
    right = s.ach_mov.split(2, lpad=lpad, rpad=rpad, concat=True)
    print('got right')

    with h5py.File(s.fs.getsyspath('trials.h5'), 'w') as f:
        group = f.create_group('ach')
        group.create_dataset('left', data=left.data)
        group.create_dataset('right', data=right.data)


def load_trials(s: Session) -> Tuple[xr.DataArray]:
    with h5py.File(s.fs.getsyspath('trials.h5'), 'r') as f:
        group = f['ach']
        left = xr.DataArray(group['left'][:], dims=('trial', 'time', 'y', 'x'))
        right = xr.DataArray(group['right'][:], dims=('trial', 'time', 'y', 'x'))
    evs = [s.events.schema.get(event=0)] * left.sizes['time']
    evs[10:20] = [s.events.schema.get(event=1)] * 10
    left.coords['event'] = xr.DataArray(evs, dims=('time',))
    evs = [s.events.schema.get(event=0)] * left.sizes['time']
    evs[10:20] = [s.events.schema.get(event=2)] * 10
    right.coords['event'] = xr.DataArray(evs, dims=('time',))

    return left, right


s = open_session('M115', '2023-04-21', '2', fs=0)
outfile = 'video/M115_monocular_flicker.mp4'

fps = 10
qlim = (1, 99)
cmap = "gnuplot"
dpi: Number = 190
width: Number = 4
frameon: bool = True
facecolor: "ColorLike" = "white"
cmap = "hot"
lpad = 10
rpad = 10
kernel = (0.5, 2, 2)

# save_trials(s)
left, right = load_trials(s)


# subtract baseline from images
print('subtracting baseline (left)')

# you have go to get the scale from the LV or RV data. See mean_traces_sandbox
scale = 0.04393532
mov = left
baselines = []
for i in range(mov.sizes['trial']):
    front = mov.isel(trial=i, time=slice(0, 10))
    img = front.mean('time')
    baselines.append(img)
for i in range(mov.sizes['trial']):
    for j in range(mov.sizes['time']):
        mov.data[i, j] = (mov.data[i, j] - baselines[i]) / scale

# subtract baseline from images
print('subtracting baseline (right)')
mov = right
baselines = []
for i in range(mov.sizes['trial']):
    front = mov.isel(trial=i, time=slice(0, 10))
    img = front.mean('time')
    baselines.append(img)
for i in range(mov.sizes['trial']):
    for j in range(mov.sizes['time']):
        mov.data[i, j] = (mov.data[i, j] - baselines[i]) / scale

left = left.mean('trial')
right = right.mean('trial')

# add cortex mask
print('adding cortex mask')
bitmask = np.zeros([mov.sizes['y'], mov.sizes['x']], dtype=bool)
with h5py.File(s.fs.getsyspath('rois.h5'), 'r') as f:
    x = f['LH']['x'][:]
    y = f['LH']['y'][:]
    bitmask[y, x] = True
    x = f['RH']['x'][:]
    y = f['RH']['y'][:]
    bitmask[y, x] = True
YX = np.argwhere(~bitmask)
Y, X = YX[:, 0], YX[:, 1]

mov = left
for i in range(mov.sizes['time']):
    mov.data[i, Y, X] = np.nan
mov = right
for i in range(mov.sizes['time']):
    mov.data[i, Y, X] = np.nan

# with h5py.File(s.fs.getsyspath('trials.h5'), 'w') as f:
#     group = f.create_group('ach')
#     group.create_dataset('left', data=left.data)
#     group.create_dataset('right', data=right.data)

left.data = gaussian_filter(left, kernel)
right.data = gaussian_filter(right, kernel)

items = [dict(data=left, name='left'), dict(data=right, name='right')]

# Figure out fps.
if fps is None:
    fps = s.attrs["samplerate"]

# Initialize figure.
ypix, xpix = items[0]['data'][0].shape
aspect = ypix / xpix
figsize = (2.1 * width, width * aspect)
fig = Figure(
    figsize=figsize,
    frameon=frameon,
    facecolor=facecolor,
)
# Setup normalization + colormapping pipelinex, axes, and labels
print('setting up normalization')
fontdict = {'size': 16, 'color': 'white'}
label_locs = ([0.05, 0.95], [0.95, 0.95])
for i, dct in enumerate(items):
    data = dct['data']
    # vlim = [0, 0.0025]
    vlim = -0.1, 0.1
    dct['smap'] = get_smap(data=data, vlim=vlim, cmap=cmap)
    # dct['smap'] = get_smap(data=data, qlim=qlim, cmap=cmap)
    dct['ax'] = ax = fig.add_subplot(1, 2, i + 1, xmargin=0, ymargin=0)
    dct['ax'] = ax
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_title(dct['name'] + " flicker")
    dct['im'] = ax.imshow(np.zeros_like(data[0]))
    ha = 'left' if i == 0 else 'right'
    dct['label'] = ax.text(
        label_locs[i][0],
        label_locs[i][1],
        ' ',
        fontdict=fontdict,
        transform=ax.transAxes,
        horizontalalignment=ha,
        verticalalignment='top',
        usetex=False,
    )


ax = items[1]['ax']
smap = items[1]['smap']
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.add_axes(cax)
cbar = fig.colorbar(smap, cax=cax)
cax.yaxis.tick_right()
cax.yaxis.set_tick_params(labelright=True)
cax.yaxis.set_ticks([-0.05, 0, 0.05])
cax.yaxis.set_ticklabels([5, 0, 5])
cax.set_ylabel(r'dF/F $\Delta_{BL}$ (%)', rotation=270, va='bottom')
fig.tight_layout(h_pad=0.05)


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

