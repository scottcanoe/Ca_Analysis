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
import fs.errors

import h5py
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
from ca_analysis.io import *

from ca_analysis.stats import *
from ca_analysis.persistence import *
from ca_analysis.plot import *

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


def do_import(mouse, date, name, schema, day, fs=0):
    s = open_session(mouse, date, name, fs=fs, require=True)
    pull_session(s)
    import_session(s, schema, day)
    s.fs.rmtree("thorlabs", missing_ok=True)
    push_session(s)
    s.fs.remove("mov.h5")


# s = open_session("36654-1", "2022-07-30", fs=0)
# evs = s.events["event"]
# seq = evs.event.values
# abcd = find_sequences([1, 2, 3, 4], seq)
#


def __change_schemas__():

    src_paths = {
        'seq_learn_2': get_fs(0).getsyspath("event_schemas/seq_learn_2.yaml"),
        'seq_learn_3': get_fs(0).getsyspath("event_schemas/seq_learn_3.yaml"),
    }

    sessions = get_sessions()

    for s in sessions:
        try:
            attrs = s.attrs
            schema = s.events.schema
        except fs.errors.ResourceNotFound:
            print(f"{s} not processed\n")
            continue

        frame_df_0 = s.events.tables["frame"]
        event_df_0 = s.events.tables["event"]
        seq_df_0 = s.events.tables["sequence"]

        seq_to_events = [
            np.array([1, 2, 3, 4, 5]),
            np.array([6, 7, 8, 9, 10]),
            np.array([11, 12, 13, 14, 15]),
        ]

        # Modify events table
        new_ids = np.zeros(len(event_df_0), dtype=int)
        for i, row in seq_df_0.iterrows():
            new_ids[row.start:row.stop] = seq_to_events[row.sequence]
        event_df = event_df_0.copy()
        event_df["event"] = new_ids

        # Modify frames table
        new_ids = np.zeros(len(frame_df_0), dtype=int)
        for i, row in event_df_0.iterrows():
            new_ids[row.start:row.stop] = row.event

        frame_df = frame_df_0.copy()
        frame_df["event"] = new_ids

        ses_fs = s.get_fs(0)
        schema_src = src_paths[s.attrs['schema']]
        shutil.copy(schema_src, ses_fs.getsyspath("events/schema.yaml"))
        frame_df.to_csv(ses_fs.getsyspath("events/frames.csv"))
        event_df.to_csv(ses_fs.getsyspath("events/events.csv"))

        # print(f"{s}:  schema={schema.name}  , "
        #       f"schema={s.attrs['schema']}  ,  day={attrs['day']}")
        # print('----------------------------------------------------------------------')
        # print(' - Sequences')
        # for seq in schema.sequences:
        #     num = np.sum(seq_df['sequence'] == seq.id)
        #     print(f'   - {seq.name}: {num}')
        # print(' - Events')
        # for id in range(16):
        #     num = np.sum(event_df['event'] == id)
        #     print(f'   - {id}: {num}')
        #
        # print('\n')



to_import = [
    # # ("36654-1", "2022-08-01", "1", 1),
    # # ("36654-2", "2022-08-01", "1", 1),
    # # ("46274-1", "2022-08-01", "1", 1),
    # ("46288-1", "2022-08-01", "1", 1),
    # ("36654-1", "2022-08-01", "1", 1),
    # ("36654-2", "2022-08-01", "1", 1),
    # ("36662-1", "2022-09-22", "1", 5),
    # ("36662-2", "2022-09-22", "2", 5),
    # ("36662-1", "2022-09-23", "1", 1),
    # ("36662-2", "2022-09-23", "1", 1),
    # ("36662-1", "2022-09-28", "1", 5),
    # ("36662-2", "2022-09-28", "1", 5),
]

for args in to_import:
    do_import(args[0], args[1], args[2], 'seq_learn_3', day=args[3])

"""
Modify event info to reflect changes in schema.
"""
from ca_analysis.processing.utils import finalize_alignment
#
# sessions = [
#     open_session("36662-1", "2022-09-28", "1"),
#     open_session("36662-2", "2022-09-28", "2"),
# ]
# s = sessions[0]
# finalize_alignment(s, tolerance=3000)


# s = open_session('36654-1', '2022-08-01', '1')
# splits = s.data.split([1, 2, 3, 4])
# a, b, c, d = splits

#
# sessions = get_sessions(day=5)
# for s in sessions:
#     frame_df_0 = s.events.tables["frame"]
#     event_df = s.events.tables["event"]
#
#     # Modify frames table
#     new_ids = np.zeros(len(frame_df_0), dtype=int)
#     for i, row in event_df.iterrows():
#         new_ids[row.start:row.stop] = row.event
#
#     frame_df = frame_df_0.copy()
#     frame_df["event"] = new_ids
#
#     ses_fs = s.get_fs(0)
#     frame_df.to_csv(ses_fs.getsyspath("events/frames.csv"))
#
