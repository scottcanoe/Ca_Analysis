
import h5py
import matlab.engine
import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import pandas as pd

from experiments.meso.main import *
from experiments.meso.roi_processing import *



def get_seqs(ev_df: pd.DataFrame, start_id: int) -> set:
    events = ev_df.event.values
    df = ev_df[ev_df.event == start_id]
    starts = df.index
    seqs = set()
    for st in starts:
        subseq = tuple(events[st:st + 5])
        seqs.add(subseq)
    return seqs


def fix_sequence(events: np.ndarray, target: np.ndarray) -> np.ndarray:
    events = np.array(events, dtype=int)
    target = np.array(target, dtype=int)
    starts = argwhere(events == target[0])
    for st in starts:
        events[st:st + 5] = target
    return events


def save(s, new_df):
    new_df.to_csv(s.fs.getsyspath('events/events.csv'))


def check(df):

    seqs = get_seqs(df, 1)
    print(f'ABCD: {seqs}')
    seqs = get_seqs(df, 6)
    print(f'ABBD: {seqs}')
    seqs = get_seqs(df, 11)
    print(f'ACBD: {seqs}')

    start = df.start.values[1:]
    stop = df.stop.values[:-1]
    deltas = set(stop - start)
    print(f'deltas: {deltas}')


def fix(s) -> pd.DataFrame:
    ev_df = s.events['events']
    events = ev_df.event.values
    events = fix_sequence(events, [1, 2, 3, 4, 5])
    events = fix_sequence(events, [6, 7, 8, 9, 10])
    events = fix_sequence(events, [11, 12, 13, 14, 15])
    new_df = pd.DataFrame(
        dict(event=events, start=ev_df.start, stop=ev_df.stop)
    )
    return new_df


s = open_session('M172', '2023-05-15', '1', fs=0)

# - Check if well-formed: check(s.events['events'])
check(s.events['events'])

# - If not well-formed, fix it.
# new_df = fix(s)

# Check fixed version.
# check(new_df)

# If OK, save.
# new_df.to_csv(s.fs.getsyspath('events/events.csv'))
