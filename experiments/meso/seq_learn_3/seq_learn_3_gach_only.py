import intake
from intake.source.discovery import drivers, load_plugins_from_module
import weakref
from time import perf_counter as clock
from typing import Mapping

import dask.array as da
import h5py
import matlab.engine
# import matplotlib
# matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.animation import FFMpegWriter
import napari
import numpy as np
import pandas as pd
from scipy.io import loadmat

from main import *
from processing import *
from ca_analysis.plot import get_smap
from experiments.meso.roi_processing import *


def load_ach_splits(s) -> Mapping[str, xr.DataArray]:

    schema = s.events.schema
    with h5py.File(s.fs.getsyspath('splits.h5'), 'r') as f:
        ach = f['ach']
        splits = {}
        for key in ach.keys():
            group = ach[key]
            dset = group['data']
            arr = xr.DataArray(dset[:], dims=('trial', 'time', 'y', 'x'))
            events = group['events'][:]
            events = [schema.get_event(ev) for ev in events]
            coord = xr.DataArray(events, dims=('time',))
            arr.coords['event'] = coord
            splits[key] = arr
    return splits


s = open_session('M172', '2023-05-19', '1', fs=0)
path = s.fs.getsyspath('mov.h5')
f = h5py.File(path, 'r')
dset = f['ca']
arr = da.from_array(dset, chunks=(1, -1, -1))
napari.view_image(arr)


# create_masks(s)
# process_masks(s)

# s = open_session('M174', '2023-05-19', '1', fs=0)
# s.LH = ROI(s, 'LH', 'ach')
# s.RH = ROI(s, 'RH', 'ach')
#
# traces = {}
# for roi_name in ('LH', 'RH'):
#     accessor = getattr(s, roi_name)
#     seq_names = ('ABCD', 'ABBD', 'ACBD')
#     arrays = {}
#     for seq in seq_names:
#         arr = accessor.split(seq, concat=True).mean('trial')
#         arrays[seq] = arr
#     traces[roi_name] = arrays
#
#
# fig, axes = plt.subplots(2, 1)
#
# ax, roi_name = axes[0], 'LH'
# arrays = traces[roi_name]
# for key, val in arrays.items():
#     val = val - np.mean(val)
#     ax.plot(val, label=key)
# ax.legend(loc='upper right')
# annotate_onsets(ax, val.coords['event'], last='-')
# ax.set_title(roi_name)
#
# ax, roi_name = axes[1], 'RH'
# arrays = traces[roi_name]
# for key, val in arrays.items():
#     val = val - np.mean(val)
#     ax.plot(val, label=key)
# ax.legend(loc='upper right')
# annotate_onsets(ax, val.coords['event'], last='-')
# ax.set_title(roi_name)
#
# fig.tight_layout(pad=2)
# plt.show()
#
# savedir = Path.home() / 'plots/meso/seq_learn_3'
# path = savedir / 'rois.png'
# fig.savefig(path)
