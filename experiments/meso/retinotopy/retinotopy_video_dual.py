import intake
from intake.source.discovery import drivers, load_plugins_from_module
import weakref
from time import perf_counter as clock
from typing import Mapping
from types import SimpleNamespace

import dask.array as da
import h5py
import matlab.engine
import matplotlib
# matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

import napari
import numpy as np
import pandas as pd
from scipy.io import loadmat

from experiments.meso.main import *
# from experiments.meso.processing import *
from ca_analysis.plot import get_smap
from experiments.meso.roi_processing import ROI


plt.rcParams['font.size'] = 11


def mask_cortex(s: Session, arr: xr.DataArray) -> None:
    """
    Mask out non-cortical areas from an xarray DataArray in-place.

    Parameters
    ----------
    s
    obj

    Returns
    -------

    """
    bitmask = np.zeros((arr.sizes['y'], arr.sizes['x']), dtype=bool)
    with h5py.File(s.fs.getsyspath('rois.h5'), 'r') as f:
        lx = f['LH']['x'][:]
        ly = f['LH']['y'][:]
        bitmask[ly, lx] = True
        rx = f['RH']['x'][:]
        ry = f['RH']['y'][:]
        bitmask[ry, rx] = True
    YX = np.argwhere(~bitmask)
    Y, X = YX[:, 0], YX[:, 1]

    if arr.dims == ('y', 'x'):
        arr.data[Y, X] = np.nan
    elif arr.dims == ('time', 'y', 'x'):
        for t in range(arr.sizes['time']):
            arr.data[t, Y, X] = np.nan
    elif arr.dims == ('trial', 'time', 'y', 'x'):
        for i in range(arr.sizes['trial']):
            for t in range(arr.sizes['time']):
                arr.data[i, t, Y, X] = np.nan
    else:
        msg = ("Dimensions of input array must be one of ('y', 'x'), "
               "('time', 'y', 'x'), or ('trial', 'time', 'y', 'x').")
        raise ValueError(msg)



s = open_session('M153', '2023-03-30', '1', fs=0)
outfile = f'video/{s.mouse}_retinotopy.mp4'
cmap = "hot"
qlim = (0.5, 99.75)
kernel = (1, 1, 1)
fps = 30
dpi = 220
width = 8


# Load movie, apply smoothing, and mask cortex.
with h5py.File(s.fs.getsyspath('mean_mov.h5'), 'r') as f:
    ca = xr.DataArray(f['ca'][:], dims=('time', 'y', 'x'))
    ach = xr.DataArray(f['ach'][:], dims=('time', 'y', 'x'))

ca = gaussian_filter(ca, sigma=kernel)
ach = gaussian_filter(ach, sigma=kernel)
mask_cortex(s, ca)
mask_cortex(s, ach)

# rescale movies by ipsilateral monocular V1 range
LVM = ROI(s, 'LVM', 'ca')
sp = LVM.split('right').mean('trial')
mmin, mmax = np.min(sp).item(), np.max(sp).item()
scale = mmax - mmin
ca = ca / scale

LVM = ROI(s, 'LVM', 'ach')
sp = LVM.split('right').mean('trial')
mmin, mmax = np.min(sp).item(), np.max(sp).item()
scale = mmax - mmin
ach = ach / scale
items = [SimpleNamespace(name='Ca', mov=ca), SimpleNamespace(name='ACh', mov=ach)]

# Setup figure and axes.
nframes, ypix, xpix = ca.shape
aspect = ypix / xpix
figsize = (2 * width, width * aspect)
fig = Figure(
    figsize=figsize,
    frameon=True,
    facecolor=None,
)
items[0].ax = fig.add_subplot(1, 2, 1)
items[1].ax = fig.add_subplot(1, 2, 2)

for obj in items:

    # setup axes and scalar mappable
    ax = obj.ax
    mov = obj.mov
    ax.set_title(obj.name)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    obj.im = ax.imshow(np.zeros_like(mov[0]))
    vlim = np.nanmin(mov), np.nanmax(mov)
    vlim = np.nanpercentile(mov, [2.5, 97.5])
    vlim = [-0.25, 2.5]
    obj.smap = get_smap(data=mov, cmap=cmap, vlim=vlim)

    # setup animation that indicates where the sweeping bar is
    hbar_x = 20
    hbar_width = 150
    hbar_y = 20

    # - horizontal line, and left and right vertical ends
    ax.plot([hbar_x, hbar_x + hbar_width], [hbar_y, hbar_y], color='white')
    ax.plot([hbar_x, hbar_x], [hbar_y + 5, hbar_y - 5], color='white')
    ax.plot([hbar_x + hbar_width, hbar_x + hbar_width], [hbar_y + 5, hbar_y - 5], color='white')

    # - 100 degree marker
    deg_110_x = hbar_x + hbar_width * 20 / 120
    ax.plot([deg_110_x, deg_110_x], [hbar_y + 5, hbar_y - 5], color='red')

    # - moving dot
    obj.points = ax.scatter([hbar_x + 5], [hbar_y], color='white')
    x_positions = np.linspace(hbar_x, hbar_x + hbar_width, nframes)
    scat_pos = np.zeros([nframes, 2])
    scat_pos[:, 1] = hbar_y
    scat_pos[:, 0] = x_positions

    # - text labels
    ax.text(hbar_x, hbar_y - 10, r'$-120 \degree$', fontsize=12, ha='center', color='white')
    ax.text(hbar_x + hbar_width, hbar_y - 10, r'$0 \degree$', fontsize=12, ha='center', color='white')

    # add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.add_axes(cax)
    cbar = plt.colorbar(obj.smap, cax=cax, shrink=0.7)
    cax.yaxis.tick_right()
    cax.yaxis.set_tick_params(labelright=True)
    cax.set_ylabel('dF/F, %Ipsi V1M', rotation=270, va='bottom')

fig.tight_layout(pad=1)
# Save to file.
writer = FFMpegWriter(fps=fps)
with writer.saving(fig, str(outfile), dpi):
    for i in range(nframes):
        for obj in items:
            cdata = obj.smap(obj.mov[i])
            cdata[:, :, -1] = 1
            obj.im.set_data(cdata)
            obj.points.set_offsets(scat_pos[i])
            writer.grab_frame()



