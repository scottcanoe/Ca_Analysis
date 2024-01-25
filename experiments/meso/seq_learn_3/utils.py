from typing import Mapping, Optional

import h5py
import numpy as np
import xarray as xr
from ca_analysis import *


def save_sequence_splits(s: Session) -> Mapping:

    lpad = 8
    day = s.attrs['day']
    sequences = ['ABCD', 'ABBD', 'ACBD'] if day == 5 else ['ABCD']
    with h5py.File(s.fs.getsyspath('mov.h5'), 'r') as f:
        channels = list(f.keys())

    splits = {'ach': {}, 'ca': {}}
    for seq in sequences:
        splits['ach'][seq] = s.ach_mov.split(seq, lpad=lpad, concat=True)

    if 'ca' in channels:
        for seq in sequences:
            splits['ca'][seq] = s.ca_mov.split(seq, lpad=lpad, concat=True)

    with h5py.File(s.fs.getsyspath('splits.h5'), 'w') as f:
        for ch in channels:
            dct = splits[ch]
            for seq, arr in dct.items():
                group = f.create_group(f'{ch}/{seq}')
                group.create_dataset('data', data=arr.data)
                group.create_dataset('events', data=arr.coords['event'].astype(int))
                group.attrs['lpad'] = str(lpad)

    return splits


def load_sequence_splits(
    s: Session,
    sequence: str,
    channel: str,
    lpad: Optional[int] = None,
) -> xr.DataArray:

    with h5py.File(s.fs.getsyspath('splits.h5'), 'r') as f:
        group = f[f'{channel}/{sequence}']
        padsize = int(group.attrs['lpad'])
        if lpad is None:
            start = padsize
        else:
            start = padsize - lpad
        arr = xr.DataArray(group['data'][:], dims=('trial', 'time', 'y', 'x'))
        events = [s.events.schema.get_event(ev) for ev in group['events'][:]]
        coord = xr.DataArray(events, dims=('time',))
        arr.coords['event'] = coord
        out = arr.isel(time=slice(start, None))

    return out
