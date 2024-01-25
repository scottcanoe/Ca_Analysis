import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Union

import fs as pyfs
import h5py
# matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
from fs.multifs import MultiFS

from ca_analysis import *


_filesystem = MultiFS()
_filesystem.add_fs(
    'ca-nas',
    pyfs.open_fs("/media/scott/ca-nas/ephys_data"),
)
_filesystem.add_fs(
    'ssd',
    pyfs.open_fs("/home/scott/ephys_data"),
)


def get_fs(fs: Optional[Union[int, str]]):
    if fs is None:
        return _filesystem
    return _filesystem[fs]



EXPERIMENT = 'seq_learn_3'
schema = EventSchema(get_fs(0).getsyspath(f"event_schemas/{EXPERIMENT}.yaml"))


def read_h5_data(h5_path: PathLike, events: bool = True) -> SimpleNamespace:
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

        data.events = xr.DataArray(
            data.eventValues,
            dims=("time",),
            coords={"time": data.eventTimestamps},
        )
        data.events.coords["t"] = xr.DataArray(
            np.arange(data.events.sizes["time"]), dims="time",
        )
    if events:
        data.events = np.zeros(data.V.sizes['time'], dtype=int)
        inds = np.searchsorted(data.adTimestamps, data.eventTimestamps)
        for i, ix in enumerate(inds):
            data.events[ix:] = data.eventValues[i]

    return data


def import_session(
    mouse: str,
    date: str,
    run: str,
    day: int,
    plx_dir: PathLike,
    h5_dir: PathLike,
) -> None:

    # initialize session object and directory
    stem = f"sessions/{mouse}/{date}/{run}"
    fs_root = get_fs(0)
    ses_fs = fs_root.mkdir(stem, exist_ok=True, parents=True)
    s = Session(fs=ses_fs, mouse=mouse, date=date, run=run)

    # copy over plexon and h5 files
    cage, mouse_num = mouse.split('-')
    plx_dir, h5_dir = Path(plx_dir), Path(h5_dir)
    plx_file = plx_dir / f"{mouse_num}_day{day}_seqlearn3.plx"
    shutil.copyfile(plx_file, s.fs.getsyspath('data.plx'))
    h5_file = h5_dir / f"{mouse_num}_day{day}_seqlearn3.h5"
    shutil.copyfile(h5_file, s.fs.getsyspath('data.h5'))

    # handle events
    s.fs.mkdir('events', exist_ok=True)

    # - write 'schema.yaml'
    schema_file = get_fs(0).getsyspath(f'event_schemas/{schema.name}.yaml')
    shutil.copyfile(schema_file, s.fs.getsyspath('events/schema.yaml'))

    data = read_h5_data(s.fs.getsyspath('data.h5'))

    # - write 'frames.csv'
    frame_df = pd.DataFrame({
        'time': data.V.coords['time'],
        'event': data.events,
    })
    frame_df.to_csv(s.fs.getsyspath('events/frames.csv'))

    # - write 'events.csv'
    a = data.events[:-1]
    b = data.events[1:]
    edges = argwhere(a != b) + 1
    edges = np.r_[[0], edges]
    edges = edges[edges <= data.V.sizes['time']]
    if edges[-1] != data.V.sizes['time']:
        edges = np.r_[edges, [data.V.sizes['time']]]
    starts, stops = edges[:-1], edges[1:]
    vals = data.events[starts]
    event_df = pd.DataFrame(dict(event=vals, start=starts, stop=stops))
    event_df.to_csv(s.fs.getsyspath('events/events.csv'))

    # write attrs
    s.attrs['mouse'] = mouse
    s.attrs['date'] = date
    s.attrs['run'] = run
    s.attrs['day'] = day
    s.attrs['schema'] = schema.name
    s.attrs['samplerate'] = 1000
    s.attrs.save()


def open_session(
    mouse: str,
    date: str,
    run: str,
    fs: Optional[Union[int, str]] = None,
) -> None:
    stem = f"sessions/{mouse}/{date}/{run}"
    ses_fs = get_fs(fs).opendir(stem)
    s = Session(fs=ses_fs, mouse=mouse, date=date, run=run)
    s.LFP = LFPData(s)
    return s


"""
--------------------------------------------------------------------------------
"""


def get_sessions(
    filename: PathLike = "sessions.ods",
    fs: Optional[Union[int, str]] = None,
    **filters,
) -> SessionGroup:

    path = Path(filename)
    if not path.is_absolute():
        path = Path(__file__).parent / path

    df = pd.read_excel(path).dropna()
    df['date'] = df['date'].astype(str)
    df['run'] = df['run'].astype(int).astype(str)
    if 'day' in df.columns:
        df['day'] = df['day'].astype(int)

    # filter enabled
    df['enabled'] = df['enabled'].astype(bool)
    df = df[df['enabled']]

    # filter others
    for key, val in filters.items():
        df = df[df[key] == val]

    group = SessionGroup()
    for i in range(len(df)):
        row = df.iloc[i]
        group.append(open_session(row.mouse, row.date, row.run, fs=fs))

    return group
