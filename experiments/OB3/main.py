import datetime
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

__experiment__ = "OB3"
schema = get_event_schema(__experiment__)
logger = logging.getLogger(f"ca_analysis.experiments.{__experiment__}")
logging.basicConfig(level=logging.INFO)


def open_session(
    mouse: str,
    date: Union[str, datetime.date],
    run: Union[int, str] = "1",
    fs: Optional[Union[int, str]] = 0,
    cell_mode: bool = True,
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
    s.data = DataHandler(s)
    s.event_class = EventModel
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
        return self.cell_indices().size

    @property
    def n_rois(self) -> int:
        return len(self.get("iscell"))

    def cell_indices(self):
        iscell = self.get("iscell")
        return argwhere(iscell)

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
        df = self.get("event_table")

        # find locations of matching sequences, and add block and event info.
        matches = find_sequences(events.astype(int), df.event)
        starts = matches["start"].values
        stops = matches["stop"].values

        # build a grid of slice objects with shape (n_trials, n_events)
        # where each entry slices frames.
        slices = np.zeros([len(matches), len(events)], dtype=object)
        for i in range(len(matches)):
            sub_df = df.iloc[slice(starts[i], stops[i])]
            slices[i] = [slice(row[1].start, row[1].start + ev_lengths[j])
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
        target: Optional[str] = None,
        # blocks: Optional[IndexLike] = None,
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
        elif isinstance(events, Event):
            return_one = True
            events = [str(events)]
        elif is_int(events):
            return_one = True
            events = [str(schema.get(event=events))]

        # Get matrix of slices, where each column contains slices for
        # all trials for an event.
        slices = self.find_sequences(events)
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
                "roi": target.coords["roi"],
                "event": ev_slices.event,
            }
            arr = xr.DataArray(mat, dims=dims, coords=coords)
            splits.append(arr)

        return splits[0] if return_one else splits

    def _ensure_prepared(self):

        if self._prepared:
            return

        spikes = self._session.segmentation["spikes"]
        n_frames_data = spikes.sizes["time"]
        frame_df = self._session.events.tables["frame"]
        n_frames_sync = len(frame_df)
        event_df = self._session.events.tables["event"]
        n_frames = n_frames_data
        if n_frames_data != n_frames_sync:
            raise ValueError('finalize alignment')
        self.attrs["n_frames"] = n_frames
        self.attrs["frame_table"] = frame_df
        self.attrs["event_table"] = event_df

        try:
            fps = self._session.attrs["capture"]["frame"]["rate"]
        except KeyError:
            fps = self._session.events.get_fps()
        self.attrs["fps"] = fps

        vars = {}
        for name in frame_df.columns:
            arr = xr.DataArray(frame_df[name].values, dims=("time",), name=name)
            arr = arr.isel(time=slice(0, n_frames))
            vars[name] = arr
        del vars["event_value"]
        del vars["event_strobe"]
        self.attrs["time_coords"] = xr.Dataset(vars)

        self._prepared = True


"""
--------------------------------------------------------------------------------
"""

_sessions = [
    open_session("19054-1", "2022-06-28", "1", fs="ssd"),
]


def get_sessions() -> List[Session]:
    return _sessions
