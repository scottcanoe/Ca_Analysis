from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any, List, Optional, Sequence

import fs as pyfs

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ca_analysis import *
from ca_analysis.persistence import PersistentMapping

# datadir = Path.home() / "ephys/experiments/seq_learn_3/data/68230_Run3/h5"
#
#
# plexon_dir = Path.home() / "ephys/experiments/seq_learn_3/data/68230_Run3/plexon"
# h5_dir = Path.home() / "ephys/experiments/seq_learn_3/data/68230_Run3/h5"
# ephys_root = Path.home() / "ephys_data/sessions"
# cage = '68230'
# day_to_date = {
#     0: '2022-11-18',
#     1: '2022-11-21',
#     2: '2022-11-22',
#     3: '2022-11-23',
#     4: '2022-11-24',
#     5: '2022-11-25',
# }


def open_session(mouse: str, date: str, run: str = '1') -> Session:
    pass


def read_h5_data(h5_path):

    data = SimpleNamespace()
    with h5py.File(h5_path, "r") as f:
        data.filename = f.attrs["filename"]
        data.asciiString = f.attrs["asciiString"]
        if not isinstance(data.asciiString, str):
            data.asciiString = ""
        data.adFreq = f.attrs["adFreq"].item()
        data.nEvents = int(f.attrs["nEvents"].item())
        data.nSamples = int(f.attrs["nSamples"].item())

        data.adData = adData = f["adData"][:]
        data.adTimestamps = f["adTimestamps"][:].squeeze()
        data.adChannels = f["adChannels"][:].squeeze().astype(int)
        data.eventValues = f["eventValues"][:].squeeze().astype(int)
        data.eventTimestamps = f["eventTimestamps"][:].squeeze()

        if data.nSamples != len(adData):
            print('mismatch')
        if data.nEvents != len(data.eventValues):
            print('mismatch')

        data.V = xr.DataArray(
            adData,
            dims=("time", "channel"),
            coords={"time": data.adTimestamps, "channel": data.adChannels},
        )
        data.V.coords["t"] = xr.DataArray(np.arange(data.V.sizes["time"]), dims="time")

        data.events = xr.DataArray(
            data.eventValues,
            dims=("time",),
            coords={"time": data.eventTimestamps},
        )
        data.events.coords["t"] = xr.DataArray(
            np.arange(data.events.sizes["time"]), dims="time",
        )

    arr = data.V.isel(channel=0)
    del arr.coords['t']
    del arr.coords['channel']
    time = arr.coords['time'].data
    events = np.zeros(arr.sizes['time'], dtype=int)
    inds = np.searchsorted(time, data.eventTimestamps)
    for i, ix in enumerate(inds):
        events[ix:] = data.eventValues[i]
    data.events = events
    data.arr = arr

    return data


def import_ephy_session(
    plx_file: PathLike,
    mouse,
    date,
    run='1',
    **attrs,
):
    """
    Read plexon file, either in the session directory or elsewhere, and
    write it as an h5 file in the session directory. Also write attributes.


    Parameters
    ----------
    plx_file
    mouse
    date
    run

    Returns
    -------

    """
    pass


h5Dir = Path('/home/scott/ephys/experiments/seq_learn_3/data/58471_Run2/h5')
path = h5Dir / "9_day5_seqlearn3.h5"
data = read_h5_data(path)

sessions_day0 = [
    ('68231-1', '2022-12-09', '1'),
    ('68231-1', '2022-12-09', '1'),
    ('68231-1', '2022-12-09', '1'),
    ('68231-1', '2022-12-09', '1'),
]

sessions_day5 = [
    ('68231-1', '2022-12-16', '1'),
    ('68231-1', '2022-12-16', '1'),
    ('68231-1', '2022-12-16', '1'),
    ('68231-1', '2022-12-16', '1'),
]

tup = sessions_day0[0]
mouse, date, run = tup



#
# starts = []
# stops = []
# a = events[:-1]
# b = events[1:]
# edges = argwhere(a != b) + 1
# edges = np.r_[[0], edges]
# edges = edges[edges <= arr.sizes['time']]
# if edges[-1] != arr.sizes['time']:
#     edges = np.r_[edges, [arr.sizes['time']]]
# starts, stops = edges[:-1], edges[1:]
# vals = events[starts]
#
# df = pd.DataFrame(dict(event=vals, start=starts, stop=stops))
# s = EphysSession(mouse_num, day)
# df = pd.DataFrame(dict(event=vals, start=starts, stop=stops))
# df.to_csv(s.sesdir / "events/events.csv")
