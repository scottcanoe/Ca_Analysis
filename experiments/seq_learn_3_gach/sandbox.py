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
import matplotlib.colors as mpc
import matplotlib.pyplot as plt
import numpy as np

from statsmodels.distributions.empirical_distribution import ECDF
import xarray as xr
import yaml

from ca_analysis import *
from ca_analysis.indexing import *
from ca_analysis.io import *

from ca_analysis.stats import *
from ca_analysis.persistence import *
from ca_analysis.plot import *
from ca_analysis.resampling import *

from main import *
from processing import *

import matplotlib as mpl


def concat(
    arrays: Sequence[xr.DataArray],
    dim: str,
    array_coord: Optional[str] = None,
    **kw,
) -> xr.DataArray:
    out = xr.concat(arrays, dim, **kw)
    if array_coord:
        chunks = []
        for i, arr in enumerate(arrays):
            chunks.append(np.full(arr.sizes[dim], i, dtype=int))
        chunks = np.hstack(chunks)
        out.coords[array_coord] = xr.DataArray(chunks, dims=(dim,))
    return out


def get_seq_data_one(
    s: Session,
    seq: Sequence,
    rank_by: Optional[Union[str, ArrayLike]] = None,
    resample_factor: int = 2,
    gray_size: int = 6,
) -> xr.DataArray:
    # collect data
    seq = s.data.split(seq)
    seq[0] = seq[0].isel(time=slice(-gray_size, None))
    seq[-1] = seq[-1].isel(time=slice(0, gray_size))
    for i, arr in enumerate(seq):
        seq[i] = arr.mean("trial")

    # combine arrays
    arr = concat(seq, "time", array_coord="chunk").transpose("roi", "time")

    # rank rois
    if rank_by is None:
        pass
    elif is_str(rank_by):
        stat = getattr(seq[1], rank_by)("time")
        inds = np.flipud(np.argsort(stat))
        arr = arr.isel(roi=inds)
    elif isinstance(rank_by, xr.DataArray):
        arr = arr.isel(roi=rank_by)
    else:
        raise NotImplementedError

    # resample
    if resample_factor != 1:
        arr = resample1d(arr, "time", factor=resample_factor, preserve_ints=True)

    return arr


def get_seq_data(
    ses: Union[Session, Sequence[Session]],
    seq: Sequence,
    rank_by: Optional[Union[str, ArrayLike]] = None,
    resample_factor: int = 2,
    gray_size: int = 6,
) -> xr.DataArray:
    if isinstance(ses, Session):
        return get_seq_data_one(
            ses,
            seq,
            rank_by=rank_by,
            resample_factor=resample_factor,
            gray_size=gray_size,
        )
    arrays = []
    for s in ses:
        arr = get_seq_data_one(
            s,
            seq,
            rank_by=rank_by,
            resample_factor=resample_factor,
            gray_size=gray_size,
        )
        arrays.append(arr)
    out = xr.concat(arrays, "roi")
    return out


def do_import(mouse, date, run, day):

    path = f'sessions/{mouse}/{date}/{run}'
    remote = get_fs(-1)
    local = get_fs(0)
    pyfs.copy.copy_dir_if(remote, path, local, path, "newer", preserve_time=True)
    s = open_session(mouse, date, run, fs=0)
    import_session(s, 'seq_learn_3', day)
    pyfs.copy.copy_dir_if(local, path, remote, path, "newer", preserve_time=True)
    s.fs.rmtree("thorlabs", missing_ok=True)
    s.fs.remove("mov.h5")




def make_mean(s: Session):
    schema = s.events.schema
    f = h5py.File(s.fs.getsyspath('mov.h5'), 'r')
    dset = f['data']
    mov = da.from_array(dset, chunks=(20, -1, -1))
    G = np.zeros(mov.shape[0])
    for i in range(len(G)):
        print(f'{i}/{len(G)}: {100*i/len(G)}%')
        frame = mov[i].compute()
        G[i] = frame.mean()
    np.save(s.fs.getsyspath('G.npy'), G)
    f.close()


to_import = [
    ('58470-1', '2022-12-19', '1'),
    ('58470-2', '2022-12-19', '1'),
    ('58470-3', '2022-12-19', '1'),
    ('63668-1', '2022-12-16', '1'),
    ('63668-2', '2022-12-16', '1'),
    ('63668-3', '2022-12-16', '1'),
    ('63668-4', '2022-12-19', '1'),
    ('63668-5', '2022-12-16', '1'),
]
for tup in to_import:
    do_import(tup[0], tup[1], tup[2], 5)

# sessions = [
#     open_session("63668-1", "2022-12-09", "1"),
#     open_session("63668-2", "2022-12-09", "1"),
    # open_session("63668-3", "2022-12-09", "1"),
    # open_session("63668-5", "2022-12-09", "1"),
# ]

# s = sessions[0]
# for s in sessions:
#     print(f'{s}')
#     with h5py.File(s.fs.getsyspath('mov.h5'), 'r+') as f:
#         dset = f['data']
#         n_frames = dset.shape[0]
#         G = np.zeros(n_frames)
#         for i in range(dset.shape[0]):
#             if i % 10 == 0:
#                 print(f'{i} / {n_frames}    {100*i/n_frames} pct')
#             frame = dset[i]
#             G[i] = np.mean(frame)
#         f.create_dataset('G', data=G)

# s = open_session("61097-`", "2022-11-`4", "1", fs='ca-nas')
# f = h5py.File(s.fs.getsyspath('mov.h5'), 'r')
# mov = da.from_array(f['data'], chunks=(10, -1, -1))
# viewer = napari.view_image(mov)
#
# seq_name = 'ACBD'
# s = open_session("55708-3", "2022-10-14", "1", fs='ca-nas')
#
# fix, ax = plt.subplots()
#
#
# for seq_name in ('ABCD', 'ABBD', 'ACBD'):
#
#     # make_mean(s)
#
#     schema = s.events.schema
#     frame_df = s.events['frame']
#     ev_df = s.events['event']
#     fps = s.events.get_fps()
#
#     G = np.load(s.fs.getsyspath("G.npy"))
#     G = xr.DataArray(G, dims=('time',), coords=dict(time=frame_df['time']))
#
#     events = schema.get_sequence(seq_name)
#     events = list(events)
#     events = [events[-1]] + events
#
#     ev_lengths = [round(ev.duration * fps) for ev in events]
#     ev_ints = [int(ev) for ev in events]
#     matches = find_sequences(ev_ints, ev_df.event)
#     starts = matches["start"].values
#     stops = matches["stop"].values
#
#     # build a grid of slice objects with shape (n_trials, n_events)
#     # where each entry slices frames, and add block coordinate.
#     block_ids = np.zeros(len(matches), dtype=int)
#     slices = np.zeros([len(matches), len(events)], dtype=object)
#     for i in range(len(matches)):
#         sub_df = ev_df.iloc[slice(starts[i], stops[i])]
#         slices[i] = [slice(row[1].start, row[1].start + ev_lengths[j])
#                      for j, row in enumerate(sub_df.iterrows())]
#     coords = {
#         # "block": xr.DataArray(block_ids, dims=("trial",)),
#         "event": xr.DataArray(events, dims=("event",)),
#     }
#     slices = xr.DataArray(slices, dims=("trial", "event"), coords=coords)
#     n_trials = slices.sizes["trial"]
#     n_events = slices.sizes["event"]
#
#     target = G
#
#     splits = []
#     for i in range(n_events):
#         ev_slices = slices.isel(event=i)
#         if ev_slices.sizes["trial"]:
#             n_frames = ev_slices[0].item().stop - ev_slices[0].item().start
#         else:
#             n_frames = 0
#         mat = np.zeros((n_trials, n_frames), dtype=target.dtype)
#         for j, slc in enumerate(ev_slices):
#             mat[j] = target.isel(time=slc.item())
#
#         dims = ("trial", "time")
#         coords = {
#             "event": ev_slices.event,
#         }
#         arr = xr.DataArray(mat, dims=dims, coords=coords)
#         splits.append(arr)
#
#
#     for i, arr in enumerate(splits):
#         ev = arr.coords['event'].item()
#         lbls = [ev.name for _ in range(arr.sizes['time'])]
#         lbls = xr.DataArray(lbls, dims=('time',))
#         arr.coords['label'] = lbls
#
#
#     seq = xr.concat(splits, 'time')
#     pre, A, B, C, D, post = splits
#     meanseq = seq.mean('trial')
#     arr = resample1d(meanseq, 'time', factor=4)
#     smoothed = gaussian_filter(arr, 0.25)
#     arr = xr.DataArray(smoothed, dims=arr.dims, coords=arr.coords)
#
#     # fig, ax = plt.subplots()
#     ax.plot(arr, label=seq_name)
#
# labels = seq.coords['label'].data
# annotate_onsets(ax, labels, color='gray')
# ax.set_xlim([10, 68])
# # ax.set_title(seq_name)
# ax.legend()
# plt.show()

