
import datetime
import functools
import logging
import os
from typing import (
    Any,
    List,
    Optional,
    Sequence,
    Union,
)
from fs.errors import *
import numpy as np
import pandas as pd

from ca_analysis import *
from ca_analysis.environment import get_fs
from ca_analysis.indexing import *
from ca_analysis.io.suite2p import Suite2PStore


__experiment__ = "seq_learn_2"
schema = get_event_schema(__experiment__)
logger = logging.getLogger("ca_analysis")
logging.basicConfig(level=logging.INFO)


def open_session(
    mouse: str,
    date: Union[str, datetime.date],
    run: Union[int, str] = "1",
    fs: Optional[Union[int, str]] = None,
    cell_mode: bool = True,
    require: bool = False,
) -> Session:

    date = as_date(date)
    run = str(run)
    stem = os.path.join("sessions", mouse, str(date), run)

    if fs is None:
        mfs = get_fs()
        try:
            ses_fs = mfs.opendir(stem)
        except ResourceNotFound:
            if not require:
                raise
            mfs.write_fs.makedirs(stem)
            ses_fs = mfs.opendir(stem)
    else:
        fs_branch = get_fs(fs)
        try:
            ses_fs = fs_branch.opendir(stem)
        except ResourceNotFound:
            if not require:
                raise
            ses_fs = fs_branch.makedirs(stem)

    s = Session(mouse=mouse, date=date, run=run, fs=ses_fs)
    s.data = DataHandler(s)
    s.segmentation_class = Suite2PStore

    # thorlabs
    s.reg["thorlabs"] = "thorlabs"
    s.reg.add("thor_md", "thorlabs/Experiment.xml"),
    s.reg.add("thor_raw", "thorlabs/Image.raw"),
    s.reg.add("thor_sync", "thorlabs/Episode.h5"),

    # events
    s.reg.add("events", "events")
    s.reg.add("schema", "events/schema.yaml")
    s.reg.add("frames_table", "events/frames.csv")
    s.reg.add("events_table", "events/events.csv")
    s.reg.add("sequences_table", "events/sequences.csv")
    s.reg.add("blocks_table", "events/blocks.csv")

    # motion correction
    s.reg.add("mc", "motion_correction"),
    s.reg.add("mc_shifts", "motion_correction/shifts.npy"),
    s.reg.add("mc_template", "motion_correction/template.npy"),
    s.reg.add("mc_summary", "motion_correction/summary.pdf"),

    # scratch
    s.reg.add("scratch", "scratch"),
    s.reg.add("mov_unprocessed", "scratch/mov_unprocessed.h5?name=data"),

    # segmentation
    s.reg.add("segmentation", "suite2p"),

    # analysis files
    s.reg.add("analysis", "analysis"),

    # etc
    s.reg.add("attrs", "attrs.yaml"),
    s.reg.add("mov", "mov.h5?name=data"),
    s.reg.add("sample", "sample.mp4"),

    # data handler
    s.data = DataHandler(s, cell_mode=cell_mode)

    return s


"""
--------------------------------------------------------------------------------
"""


class DataHandler:
    """
    Accessor class assigned to each session instance during`open_session`.
    Specialized to handle slicing and grouping of transitions.
    """

    def __init__(
        self,
        session: Session,
        cell_mode: bool = True,
        target: str = "spikes",
    ):

        self._session = session
        self.cell_mode = cell_mode
        self.target = target
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

    def get(self, key: str, **kw) -> Any:

        self._ensure_prepared()
        if key in self.attrs:
            return self.attrs[key]

        # lazy loading
        if key == "iscell":
            val = self._session.segmentation["iscell"]
        elif key in ("F", "Fneu", "spikes"):
            val = self.get_fluorescence(key, **kw)
        elif key == "schema":
            val = self._session.events.schema
        else:
            raise KeyError(key)
        self.attrs[key] = val
        return val

    def get_fluorescence(self, name: str) -> xr.DataArray:

        self._ensure_prepared()

        # Get the needed data, optionally filtering by cell id.
        # Also make sure the number of imaging frames matches the
        # number of frames captured by thorsync.
        arr = self._session.segmentation[name]

        if self.cell_mode:
            inds = argwhere(self.get("iscell"))
            arr = arr.isel(roi=inds)
        n_frames = self.get("n_frames")
        arr = arr.isel(time=slice(0, n_frames))

        # Add time coordinates (such as event_id)
        coords = dict(arr.coords)
        time_coords = self.get("time_coords")
        coords["time"] = time_coords["time"]
        coords["event"] = time_coords["event"]

        # Define useful attributes.
        attrs = dict(arr.attrs)
        attrs["name"] = name
        attrs["session"] = self._session

        out = xr.DataArray(
            arr.data,
            dims=arr.dims,
            coords=coords,
            attrs=attrs,
            name=name,
        )
        return out

    def find_sequences(
        self,
        events: Sequence,
        blocks: Optional[IndexLike] = None,
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

        if is_str(events):
            events = schema.get(sequence=events)
        else:
            events = EventSequence(events=events, schema=schema)

        fps = self.get("fps")
        ev_lengths = [round(ev.duration * fps) for ev in events]
        ev_df = self.get("event_table")

        # Optionally apply block filter to event table.
        if blocks is not None:
            block_ids = ev_df.block
            if is_int(blocks):
                rows = argwhere(block_ids == blocks)
            elif isinstance(blocks, slice):
                valid_blocks = np.arange(block_ids.max() + 1)[blocks]
                rows = np.in1d(block_ids, valid_blocks)
            else:
                rows = np.in1d(block_ids, np.array(blocks, dtype=int))
            ev_df = ev_df.iloc[rows]

        # find locations of matching sequences, and add block and event info.
        matches = find_sequences(events.astype(int), ev_df.event)
        starts = matches["start"].values
        stops = matches["stop"].values

        # build a grid of slice objects with shape (n_trials, n_events)
        # where each entry slices frames, and add block coordinate.
        block_ids = np.zeros(len(matches), dtype=int)
        slices = np.zeros([len(matches), len(events)], dtype=object)
        for i in range(len(matches)):
            sub_df = ev_df.iloc[slice(starts[i], stops[i])]
            slices[i] = [slice(row[1].start, row[1].start + ev_lengths[j])
                         for j, row in enumerate(sub_df.iterrows())]
            # block_ids[i] = sub_df.iloc[0].block

        # put in a data array to hold extra metadata, like block id.
        coords = {
            # "block": xr.DataArray(block_ids, dims=("trial",)),
            "event": xr.DataArray(events, dims=("event",)),
        }
        arr = xr.DataArray(slices, dims=("trial", "event"), coords=coords)
        return arr

    def split(
        self,
        events: Sequence,
        target: Optional[str] = None,
        blocks: Optional[IndexLike] = None,
    ) -> List[xr.DataArray]:
        """
        Given a sequence of elements, return a list of arrays (one for each
        item) having dimensions ('trial', 'time', 'roi').

        Parameters
        ----------
        target
        events
        block

        Returns
        -------

        """

        return_one = False
        if is_str(events):
            if events in schema.tables["sequence"]["name"].values:
                events = schema.get(sequence=events).astype(str)
            else:
                events = [events]
                return_one = True

        # Get matrix of slices, where each column contains slices for
        # all trials for an event.
        slices = self.find_sequences(events, blocks=blocks)
        target = target or self.target
        target = self.get(target)
        n_trials = slices.sizes["trial"]
        n_events = slices.sizes["event"]
        n_rois = target.sizes["roi"]

        # Use the slices to extract the data from a target array for each event.
        if target.dims != ("time", "roi"):
            target = target.transpose("time", "roi")

        splits = []
        for i in range(n_events):
            ev_slices = slices.isel(event=i)
            if ev_slices.sizes["trial"]:
                n_frames = ev_slices[0].item().stop - ev_slices[0].item().start
            else:
                n_frames = 0
            mat = np.zeros((n_trials, n_frames, n_rois), dtype=target.dtype)
            for j, slc in enumerate(ev_slices):
                mat[j] = target.isel(time=slc.item())

            dims = ("trial", "time", "roi")
            coords = {
                # "block": ev_slices.coords["block"],
                "roi": target.coords["roi"],
                "event": ev_slices.event,
            }
            arr = xr.DataArray(mat, dims=dims, coords=coords)
            splits.append(arr)

        return splits[0] if return_one else splits

    def _ensure_prepared(self):

        if self._prepared:
            return

        frame_df = self._session.events.tables["frame"]
        event_df = self._session.events.tables["event"]
        self.attrs["n_frames"] = len(frame_df)
        self.attrs["frame_table"] = frame_df
        self.attrs["event_table"] = event_df

        try:
            fps = self._session.attrs["capture"]["frame"]["rate"]
        except KeyError:
            fps = self._session.events.get_fps()
        self.attrs["fps"] = fps

        coords = {}
        coords['time'] = xr.DataArray(frame_df['time'].values, dims=('time',))
        coords['event'] = xr.DataArray(frame_df['event'].values, dims=('time',))
        if "block" in frame_df.columns:
            coords['block'] = xr.DataArray(frame_df['event'].values, dims=('time',))
        self.attrs["time_coords"] = xr.Dataset(coords)

        self._prepared = True



"""
--------------------------------------------------------------------------------
"""

_sessions_day_0 = [
    # seq_learn_3
    # open_session("36662-1", "2022-09-22", "1", fs="ssd"),
    open_session("36662-2", "2022-09-22", "2", fs="ssd"),
]

_sessions_day_1 = [
    # seq_learn_2
    open_session("19053-1", "2022-05-26", "1", fs="ssd"),
    open_session("19054-1", "2022-05-26", "1", fs="ssd"),
    open_session("36654-1", "2022-07-26", "1", fs="ssd"),
    open_session("36654-2", "2022-07-26", "1", fs="ssd"),
    open_session("46274-1", "2022-07-26", "1", fs="ssd"),
    open_session("46288-1", "2022-07-25", "1", fs="ssd"),
    open_session("51537-1", "2022-07-25", "1", fs="ssd"),
    # seq_learn_3
    open_session("36654-1", "2022-08-01", "1", fs="ssd"),
    open_session("36654-2", "2022-08-01", "1", fs="ssd"),
    open_session("46274-1", "2022-08-01", "1", fs="ssd"),
    open_session("46288-1", "2022-08-01", "1", fs="ssd"),
]

_sessions_day_5 = [
    # seq_learn_2
    open_session("19053-1", "2022-05-30", "1", fs="ssd"),
    open_session("19054-1", "2022-05-30", "1", fs="ssd"),
    open_session("36654-1", "2022-07-30", "1", fs="ssd"),
    open_session("36654-2", "2022-07-30", "1", fs="ssd"),
    open_session("46274-1", "2022-07-30", "1", fs="ssd"),
    open_session("46288-1", "2022-07-29", "1", fs="ssd"),
    # open_session("51537-1", "2022-07-29", "1", fs="ssd"), # no good
    # seq_learn_3
    open_session("36654-1", "2022-08-18", "1", fs="ssd"),
    open_session("36654-2", "2022-08-18", "1", fs="ssd"),
    open_session("46274-1", "2022-08-18", "1", fs="ssd"),
    open_session("46288-1", "2022-08-18", "1", fs="ssd"),
]


def get_sessions(day: Optional[int] = None) -> List[Session]:
    if day is None:
        return _sessions_day_0 + _sessions_day_1 + _sessions_day_5
    if day == 0:
        return _sessions_day_0
    if day == 1:
        return _sessions_day_1
    if day == 5:
        return _sessions_day_5
    raise KeyError(day)

