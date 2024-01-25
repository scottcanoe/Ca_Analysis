import datetime
import logging
import os
from pathlib import Path
from typing import (
    List, Mapping, Optional,
    Union,
)

import dask.array as da
from fs.errors import *
import h5py
from ca_analysis import *


logger = logging.getLogger(f"ca_analysis.experiments.meso")
logging.basicConfig(level=logging.INFO)


def open_session(
        mouse: str,
        date: Union[str, datetime.date],
        run: Union[int, str] = "1",
        fs: Optional[Union[int, str]] = None,
        create: bool = False,
) -> Session:

    date = as_date(date)
    run = str(run)
    stem = os.path.join("sessions", mouse, str(date), run)

    parent_fs = get_fs(fs)
    try:
        fs = parent_fs.opendir(stem)
    except ResourceNotFound:
        if not create:
            raise
        fs = parent_fs.makedirs(stem)

    s = Session(fs, mouse=mouse, date=date, run=run)
    s.event_class = EventModel
    s.segmentation_class = None

    # thorlabs
    s.reg.add("thorlabs", "thorlabs")
    s.reg.add("thor_md", "thorlabs/Experiment.xml")
    s.reg.add("thor_raw", "thorlabs/Image.raw")
    s.reg.add("thor_sync", "thorlabs/Episode.h5")

    # events
    s.reg.add("events", "events")
    s.reg.add("schema", "events/schema.yaml")
    s.reg.add("frames_table", "events/frames.csv")
    s.reg.add("events_table", "events/events.csv")

    # scratch
    s.reg.add("scratch", "scratch")

    # segmentation
    s.reg.add("segmentation", "segmentation")

    # analysis files
    s.reg.add("analysis", "analysis")

    # etc
    s.reg.add("attrs", "attrs.yaml")


    s.ca_mov = MovieData(s, 'mov.h5', 'ca')
    s.ach_mov = MovieData(s, 'mov.h5', 'ach')
    s.ach_mov_raw = MovieData(s, 'raw/mov.h5', 'gfp/norm')
    return s


def get_sessions(
    filename: PathLike = "sessions.ods",
    fs: Optional[Union[int, str]] = None,
    **filters,
) -> SessionGroup:

    import pandas as pd

    path = Path(filename)
    if not path.is_absolute():
        path = Path(__file__).parent / path

    df = pd.read_excel(path).dropna()
    df['date'] = df['date'].astype(str)
    df['run'] = df['run'].astype(int).astype(str)
    if 'day' in df.columns:
        df['day'] = df['day'].astype(int)

    df['enabled'] = df['enabled'].astype(bool)
    df = df[df['enabled']]
    for key, val in filters.items():
        df = df[df[key] == val]
    group = SessionGroup()
    for i in range(len(df)):
        row = df.iloc[i]
        group.append(open_session(row.mouse, row.date, row.run, fs=fs))

    return group



def get_channels(s: Session) -> List[str]:
    with h5py.File(s.fs.getsyspath('mov.h5'), 'r') as f:
        return list(f.keys())


def make_splits(s: Session) -> None:
    """
    - check attrs for schema
    - check for num channels
    retinotopy:
      ach/right/data
      ca/right/data (optional)

    monoc/gach_test_4:
      ach/left/lpad
      ach/left/data
      ach/left/rpad
    """

    schema_name = s.attrs['schema']
    assert schema_name in {'retinotopy', 'gach_test_4'}
    with h5py.File(s.fs.getsyspath('mov.h5'), 'r') as f:
        channels = list(f.keys())

    if schema_name == 'retinotopy':
        right = s.ach_mov.split('right')
        with h5py.File(s.fs.getsyspath('splits.h5'), 'w') as f:
            f.create_dataset('/ach/right/data', data=right)
        if 'ca' in channels:
            right = s.ca_mov.split('right')
            with h5py.File(s.fs.getsyspath('splits.h5'), 'a') as f:
                f.create_dataset('/ca/right/data', data=right)

    else:
        left = s.ach_mov.split('left', lpad=5, rpad=5)
        right = s.ach_mov.split('right', lpad=5, rpad=5)
        with h5py.File(s.fs.getsyspath('splits.h5'), 'w') as f:
            f.create_dataset('/ach/left/lpad', data=left[0])
            f.create_dataset('/ach/left/data', data=left[1])
            f.create_dataset('/ach/left/rpad', data=left[2])
            f.create_dataset('/ach/right/lpad', data=right[0])
            f.create_dataset('/ach/right/data', data=right[1])
            f.create_dataset('/ach/right/rpad', data=right[2])

        if 'ca' in channels:
            left = s.ca_mov.split('left', lpad=5, rpad=5)
            right = s.ca_mov.split('right', lpad=5, rpad=5)
            with h5py.File(s.fs.getsyspath('splits.h5'), 'a') as f:
                f.create_dataset('/ca/left/lpad', data=left[0])
                f.create_dataset('/ca/left/data', data=left[1])
                f.create_dataset('/ca/left/rpad', data=left[2])
                f.create_dataset('/ca/right/lpad', data=right[0])
                f.create_dataset('/ca/right/data', data=right[1])
                f.create_dataset('/ca/right/rpad', data=right[2])


def load_splits(
    s: Session,
    channel: str,
    event: str,
    lpad: bool = True,
    rpad: bool = True,
    concat: bool = False,
) -> Union[List, xr.DataArray]:

    import h5py

    assert channel in {'ca', 'ach'}
    splits = []
    data_path = f'/{channel}/{event}/data'
    lpad_path = f'/{channel}/{event}/lpad'
    rpad_path = f'/{channel}/{event}/rpad'
    dims = ('trial', 'time', 'y', 'x')
    with h5py.File(s.fs.getsyspath('splits.h5'), 'r') as f:
        splits = [xr.DataArray(f[data_path][:], dims=dims)]
        if lpad and lpad_path in f:
            arr = xr.DataArray(f[lpad_path][:], dims=dims)
            splits.insert(0, arr)
        if rpad and rpad_path in f:
            arr = xr.DataArray(f[rpad_path][:], dims=dims)
            splits.append(arr)
    if len(splits) == 1:
        return splits[0]
    if concat:
        return xr.concat(splits, 'time')
    return splits


"""
--------------------------------------------------------------------------------
Segmentation

"""
