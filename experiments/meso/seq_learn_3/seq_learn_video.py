import intake
from intake.source.discovery import drivers, load_plugins_from_module
import weakref
from time import perf_counter as clock
from types import SimpleNamespace
from typing import Mapping

import dask.array as da
import h5py
import matlab.engine
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.animation import FFMpegWriter
import napari
import numpy as np
import pandas as pd
from scipy.io import loadmat

from main import *
from processing import *
from ca_analysis.plot import get_smap



s = open_session('meso-1', '2023-02-10', '1', fs=0)

outfile = s.fs.getsyspath('trial_averages.mp4')
lpad = 5
rpad = None
fps = 10
qlim = (0.25, 97.5)
cmap = "inferno"
dpi: Number = 190
width: Number = 4
frameon: bool = True
facecolor: "ColorLike" = "white"
cmap = "inferno"

# mask = np.load(s.fs.getsyspath('mask.npy'))
# mov = s.ca_mov.split('ABCD', concat=True)
# mov = s.ca_mov.data[:100].compute()
# data = add_mask(mov.data, mask)
# arr = xr.DataArray(data, dims=mov.dims, coords=mov.coords)

ABCD_ca = SimpleNamespace(name='ABCD')
ABCD_ca.data = s.ca_mov.split('ABCD', lpad=lpad, rpad=rpad, concat=True).mean('trial')

ABCD_ach = SimpleNamespace(name='ABCD')
ABCD_ach.data = s.ach_mov.split('ABCD', lpad=lpad, rpad=rpad, concat=True).mean('trial')
print('finished collecting ABCD data')

ABBD_ca = SimpleNamespace(name='ABBD')
ABBD_ca.data = s.ca_mov.split('ABBD', lpad=lpad, rpad=rpad, concat=True).mean('trial')

ABBD_ach = SimpleNamespace(name='ABBD')
ABBD_ach.data = s.ach_mov.split('ABBD', lpad=lpad, rpad=rpad, concat=True).mean('trial')
print('finished collecting ABBD data')

ACBD_ca = SimpleNamespace(name='ACBD')
ACBD_ca.data = s.ca_mov.split('ACBD', lpad=lpad, rpad=rpad, concat=True).mean('trial')

ACBD_ach = SimpleNamespace(name='ACBD')
ACBD_ach.data = s.ach_mov.split('ACBD', lpad=lpad, rpad=rpad, concat=True).mean('trial')
print('finished collecting ACBD data')

objects = [
    ABCD_ca,
    ABCD_ach,
    ABBD_ca,
    ABBD_ach,
    ACBD_ca,
    ACBD_ach,
]

# mask data
mask = np.load(s.fs.getsyspath('mask.npy'))
for obj in objects:
    data = obj.data
    dims = data.dims
    coords = data.coords
    obj.data = xr.DataArray(add_mask(data.data, mask), dims=dims, coords=coords)

print('finished collecting data')

# Initialize figure.
ypix, xpix = ABCD_ca.data.shape[1:]
aspect = ypix / xpix
figsize = (2 * width, 3 * width * aspect)
fig = Figure(
    figsize=figsize,
    frameon=frameon,
    facecolor=facecolor,
)
fig.tight_layout(pad=1)

# Initialize colormaps
ca_data = xr.concat([ABCD_ca.data, ABBD_ca.data, ACBD_ca.data], 'time')
ca_smap = get_smap(data=ca_data, qlim=qlim, cmap=cmap)
ABCD_ca.smap = ca_smap
ABBD_ca.smap = ca_smap
ACBD_ca.smap = ca_smap

ach_data = xr.concat([ABCD_ach.data, ABBD_ach.data, ACBD_ach.data], 'time')
ach_smap = get_smap(data=ach_data, qlim=qlim, cmap=cmap)
ABCD_ach.smap = ach_smap
ABBD_ach.smap = ach_smap
ACBD_ach.smap = ach_smap

# Initialize axes, labels, and images.

def init_plot(obj, plot_num):
    obj.ax = fig.add_subplot(3, 2, plot_num)
    obj.ax.set_xticks([])
    obj.ax.set_xticklabels([])
    obj.ax.set_yticks([])
    obj.ax.set_yticklabels([])
    obj.ax.set_title(obj.name)
    obj.im = obj.ax.imshow(np.zeros_like(obj.data[0]))
    obj.label = obj.ax.text(
        0.05,
        0.95,
        ' ',
        fontdict={'size': 16, 'color': 'white'},
        transform=obj.ax.transAxes,
        horizontalalignment='left',
        verticalalignment='top',
        usetex=False,
    )


for i, obj in enumerate(objects):
    init_plot(obj, i + 1)

# ------------------------------------------------------------------------------
# Write frames

n_frames = ABCD_ca.data.shape[0]
writer = FFMpegWriter(fps=fps)
with writer.saving(fig, str(outfile), dpi):
    for i in range(n_frames):
        for obj in objects:
            frame = obj.data.isel(time=i)
            cdata = obj.smap(frame)
            obj.im.set_data(cdata)
            obj.label.set_text(frame.coords['event'].item())
        writer.grab_frame()

