import matplotlib
# matplotlib.use('TkAgg')

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

# datadir = Path.home() / "ephys/experiments/seq_learn_3/data/58471_Run2/h5"

h5_dir = Path.home() / "ephys/experiments/seq_learn_3/data/68230_Run3/h5"
ephys_root = Path.home() / "ephys_data/sessions"
cage = '68230'

day_to_date = {
    0: '2022-11-18',
    1: '2022-11-21',
    2: '2022-11-22',
    3: '2022-11-23',
    4: '2022-11-24',
    5: '2022-11-25',
}

# cage = '58471'
# day_to_date = {
#     0: '2022-09-02',
#     1: '2022-09-05',
#     2: '2022-09-06',
#     3: '2022-09-07',
#     4: '2022-09-08',
#     5: '2022-09-09',
# }


class EphysSession:

    def __init__(self, mouse, date):
        if is_int(mouse):
            mouse = f"{cage}-{mouse}"
        self.mouse = mouse
        if is_int(date):
            date = date = day_to_date[date]
        self.date = date
        self.sesdir = ephys_root / f"{self.mouse}/{self.date}/1"
        self.fs = pyfs.open_fs(str(self.sesdir))
        self._attrs = None
        self._V = None
        self._events = None
        self.data = DataHandler(self)

    @property
    def attrs(self) -> PersistentMapping:
        if self._attrs is None:
            try:
                path = self.fs.getsyspath("attrs.yaml")
            except pyfs.errors.ResourceNotFound:
                path = self.fs.getsyspath("") + "/attrs.yaml"
            self._attrs = PersistentMapping(path, load=True)
        return self._attrs

    @property
    def events(self) -> EventModel:
        if self._events is None:
            path = self.sesdir / "events"
            self._events = EventModel(path)
        return self._events

    @property
    def V(self):
        if self._V is None:
            h5_path = self.fs.getsyspath("data.h5")
            with h5py.File(h5_path, "r") as f:
                adData = f["adData"][:]
                adTimestamps = f["adTimestamps"][:].squeeze()
                V = xr.DataArray(
                    adData,
                    dims=("time", "channel"),
                    coords={"time": adTimestamps}
                )
            arr = V.isel(channel=0)
            self._V = arr
        return self._V


class DataHandler:
    """
    Accessor class assigned to each session instance during`open_session`.
    Specialized to handle slicing and grouping of transitions.
    """

    def __init__(
        self,
        session: Session,
    ):

        self._session = session
        self.attrs = {}
        self._prepared = False

    @property
    def session(self) -> Session:
        return self._session

    @property
    def n_cells(self) -> int:
        return sum(self.get("iscell")).item()

    @property
    def n_rois(self) -> int:
        return len(self.get("iscell"))

    def clear(self):
        self._prepared = False
        self.attrs.clear()

    def get(self, key: str) -> Any:

        self._ensure_prepared()
        if key == "fps":
            return 1000
        if key in self.attrs:
            return self.attrs[key]

        # lazy loading
        if key == "V":
            val = self._session.V
        elif key == "schema":
            val = self._session.events.schema
        else:
            raise KeyError(key)
        self.attrs[key] = val
        return val

    def find_sequences(
        self,
        events: Sequence,
    ) -> pd.DataFrame:
        """
        Find where sequences occur.

        Parameters
        ----------
        events
        blocks
        kw

        Returns
        -------
        A DataArray of Chunk objects with dimensions (trial, n_events)

        """

        self._ensure_prepared()
        schema = self._session.events.schema
        ev_df = self._session.events['event']

        if is_str(events):
            events = schema.get(sequence=events)
        else:
            events = EventSequence(events=events, schema=schema)

        fps = self.get("fps")
        ev_lengths = [round(ev.duration * fps) for ev in events]

        # find locations of matching sequences, and add block and event info.
        matches = find_sequences(events.astype(int), ev_df.event)
        starts = matches["start"].values
        stops = matches["stop"].values

        # build a grid of slice objects with shape (n_trials, n_events)
        # where each entry slices frames, and add block coordinate.
        slices = np.zeros([len(matches), len(events)], dtype=object)
        for i in range(len(matches)):
            sub_df = ev_df.iloc[slice(starts[i], stops[i])]
            slices[i] = [slice(int(row[1].start), int(row[1].start + ev_lengths[j]))
                         for j, row in enumerate(sub_df.iterrows())]

        # put in a data array to hold extra metadata, like block id.
        coords = {
            "event": xr.DataArray(events, dims=("event",)),
        }
        arr = xr.DataArray(slices, dims=("trial", "event"), coords=coords)
        return arr

    def split(
        self,
        events: Sequence,
    ) -> List[xr.DataArray]:
        """
        Given a sequence of elements, return a list of arrays (one for each
        item) having dimensions ('trial', 'time').

        Parameters
        ----------
        events

        Returns
        -------

        """

        schema = self._session.events.schema

        return_one = False

        if is_str(events):
            if events in schema.tables["sequence"]["name"].values:
                events = schema.get(sequence=events).astype(str)
            else:
                events = [events]
                return_one = True

        # Get matrix of slices, where each column contains slices for
        # all trials for an event.
        slices = self.find_sequences(events)
        target = self._session.V
        n_trials = slices.sizes["trial"]
        n_events = slices.sizes["event"]

        # Use the slices to extract the data from a target array for each event.
        splits = []
        for i in range(n_events):
            ev_slices = slices.isel(event=i)
            if ev_slices.sizes["trial"]:
                n_frames = int(ev_slices[0].item().stop - ev_slices[0].item().start)
            else:
                n_frames = 0
            mat = np.zeros((n_trials, n_frames), dtype=target.dtype)
            for j, slc in enumerate(ev_slices):
                mat[j] = target.isel(time=slc.item())

            dims = ("trial", "time")
            coords = {"event": ev_slices.event}
            arr = xr.DataArray(mat, dims=dims, coords=coords)
            splits.append(arr)

        return splits[0] if return_one else splits

    def _ensure_prepared(self):

        if self._prepared:
            return

        self._prepared = True


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

# for mouse_num in range(1, 6):
#     for day in range(6):
#         data = read_h5_data(mouse_num, day)
#         arr, events = data.arr, data.events
#
#         starts = []
#         stops = []
#         a = events[:-1]
#         b = events[1:]
#         edges = argwhere(a != b) + 1
#         edges = np.r_[[0], edges]
#         edges = edges[edges <= arr.sizes['time']]
#         if edges[-1] != arr.sizes['time']:
#             edges = np.r_[edges, [arr.sizes['time']]]
#         starts, stops = edges[:-1], edges[1:]
#         vals = events[starts]
#
#         df = pd.DataFrame(dict(event=vals, start=starts, stop=stops))
#         s = EphysSession(mouse_num, day)
#         df = pd.DataFrame(dict(event=vals, start=starts, stop=stops))
#         df.to_csv(s.sesdir / "events/events.csv")


# for mouse_num in range(1, 6):
#     for day in range(6):
#         mouse = f"68230-{mouse_num}"
#         date = day_to_date[day]
#         sesdir = ephys_root / f"{mouse}/{date}/1"
#         sesdir.mkdir(exist_ok=True, parents=True)
#         src = plexon_dir / f"{mouse_num}_day{day}_seqlearn3.plx"
#         dst = sesdir / 'data.plx'
#         shutil.copyfile(src, dst)
#         src = h5_dir / f"{mouse_num}_day{day}_seqlearn3.h5"
#         dst = sesdir / 'data.h5'
#         shutil.copyfile(src, dst)



#
# schema_file = ephys_root.parent / "event_schemas/seq_learn_3.yaml"
# for mouse_num in range(1, 6):
#     for day in range(6):
#         s = EphysSession(mouse_num, day)
#         evdir = s.sesdir / "events"
#         arr = s.data
        # arr, events = data.arr, data.events
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
        # for i in range(1, 15):
        #     ind = edges[i]
        #     print(f"({events[ind-1]})  {events[ind]}  --> {events[ind + 1]}")
        # starts, stops = edges[:-1], edges[1:]
        # vals = events[starts]
        # df = pd.DataFrame(dict(event=vals, start=starts, stop=stops))


from scipy.signal import periodogram, welch
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import pearsonr
from time import perf_counter as clock


def autocov(arr, maxlag=None):
    arr = np.asarray(arr)
    n = len(arr)
    if maxlag is None:
        maxlag = len(arr) // 2
    winsize = len(arr) - maxlag
    main_chunk = arr[:winsize]
    lags = np.arange(maxlag)
    coeffs = np.zeros(len(lags))
    for i, lag in enumerate(lags):
        r = pearsonr(main_chunk, arr[i:i + winsize])
        coeffs[i] = r.statistic
    return lags, coeffs


mouse = 4
maxlag = 1250
portion = 1

for mouse in range(1, 6):

    t_start = clock()
    s = EphysSession(mouse, 0)
    arr = s.V.data
    arr = arr[:arr.shape[0] // portion]
    lags, coeffs_day0 = autocov(arr, maxlag)

    s = EphysSession(mouse, 5)
    arr = s.V.data
    arr = arr[:arr.shape[0] // portion]
    lags, coeffs_day5 = autocov(arr, maxlag)

    fig, ax = plt.subplots(figsize=(6, 4))
    for x in [0.25, 0.5, 0.75, 1.0]:
        ax.axvline(x, color='gray', ls='--', alpha=0.5)
    X = lags / 1000
    ax.plot(X, coeffs_day0, color='blue', label='day 0')
    ax.plot(X, coeffs_day5, color='red', label='day 5')
    ax.legend()
    ax.set_xlabel('lag (sec)')
    ax.set_ylabel('pearson r')
    ax.set_title(f'mouse {s.mouse}')
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlim([0, 1.25])
    fig.tight_layout()
    plt.show()
    datadir = Path.home() / 'plots/seq_learn_3/ephys/autocorr'
    datadir.mkdir(exist_ok=True, parents=True)
    path = datadir / f"mouse_{mouse}.pdf"
    fig.savefig(path)
    t_stop = clock()
    print(f"took {t_stop - t_start} secs")


# fig, ax = plt.subplots(figsize=(10, 5))
# ax.plot(f, Pxx)
# ax.set_xlim([0, 15])
# ax.semilogy()
# plt.show()
# dirpath = Path.home() / 'plots/seq_learn_3/ephys'
# dirpath.mkdir(exist_ok=True, parents=True)
#
# for mouse in range(1, 6):
#
#     fig, axes = plt.subplots(3, 1, figsize=(8, 8))
#
#     s_day0 = EphysSession(mouse, 0)
#     s_day5 = EphysSession(mouse, 5)
#
#     for ax_num, seq_name in enumerate(['ABCD', 'ABBD', 'ACBD']):
#         ax = axes[ax_num]
#         X = np.arange(1000)
#
#         lst = s_day0.data.split(seq_name)
#         for i, obj in enumerate(lst):
#             obj = obj.mean('trial')
#             lst[i] = obj
#         lst = lst[:-1]
#         arr = xr.concat(lst, 'time')
#         arr = arr - arr[0]
#         ax.plot(X, arr, label='day 0')
#
#         lst = s_day5.data.split(seq_name)
#         for i, obj in enumerate(lst):
#             obj = obj.mean('trial')
#             lst[i] = obj
#         lst = lst[:-1]
#         arr = xr.concat(lst, 'time')
#         arr = arr - arr[0]
#         ax.plot(X, arr, label='day 5')
#
#         ax.legend()
#         ax.set_xticks([0, 250, 500, 750])
#         ax.set_xticklabels(['0', '0.25', '0.5', '0,75'])
#         ax.set_title(seq_name)
#
#     fig.tight_layout()
#     plt.show()
#
#     fname = f"mouse_{mouse}"
#     path = dirpath / f"{fname}.png"
#     fig.savefig(path)

#         arr, events = data.arr, data.events
#
#         starts = []
#         stops = []
#         a = events[:-1]
#         b = events[1:]
#         edges = argwhere(a != b) + 1
#         edges = np.r_[[0], edges]
#         edges = edges[edges <= arr.sizes['time']]
#         if edges[-1] != arr.sizes['time']:
#             edges = np.r_[edges, [arr.sizes['time']]]
#         starts, stops = edges[:-1], edges[1:]
#         vals = events[starts]
#
#         df = pd.DataFrame(dict(event=vals, start=starts, stop=stops))
#         s = EphysSession(mouse_num, day)
#         df = pd.DataFrame(dict(event=vals, start=starts, stop=stops))
#         df.to_csv(s.sesdir / "events/events.csv")

