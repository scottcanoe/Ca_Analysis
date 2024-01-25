import datetime
import logging
import os
from typing import (
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from ca_analysis import *
from ca_analysis.environment import get_fs
from ca_analysis.io.suite2p import Suite2PStore

__all__ = [
    "get_transitions",
    "open_session",
    "schema",
    "Transition",
]


schema_name = "FOC_3.1"
schema_path = get_fs(0).getsyspath(f"event_schemas/{schema_name}.yaml")
schema = EventSchema(schema_path)

logger = logging.getLogger(f"ca_analysis.pipelines.{schema_name}")

"""
---------------------------------------------------------------------------------------------------------
"""


def open_session(
        mouse: str,
        date: Union[str, datetime.date],
        exp: Union[int, str] = "1",
        fs: Optional[Union[int, str]] = 0,
        create: bool = False,
) -> Session:

    date = as_date(date)
    exp = str(exp)
    path = os.path.join("sessions", mouse, str(date), exp)

    parent_fs = get_fs(fs)
    if create:
        sfs = parent_fs.makedir(path)
    else:
        sfs = parent_fs.opendir(path)

    s = Session(sfs, mouse=mouse, date=date, exp=exp)
    s.segmentation_class = Suite2PStore
    s.event_class = EventModel

    # thorlabs
    s.reg.add("thorlabs", "thorlabs", is_dir=True)
    s.reg.add("thor_md", "thorlabs/Experiment.xml"),
    s.reg.add("thor_raw", "thorlabs/Image.raw"),
    s.reg.add("thor_sync", "thorlabs/Episode.h5"),

    # events
    s.reg.add("events", "events", is_dir=True)
    s.reg.add("schema", "events/schema.yaml")
    s.reg.add("frames_table", "events/frames.csv")
    s.reg.add("events_table", "events/events.csv")

    # motion correction
    s.reg.add("mc", "motion_correction", is_dir=True),
    s.reg.add("mc_shifts", "motion_correction/shifts.npy"),
    s.reg.add("mc_template", "motion_correction/template.npy"),
    s.reg.add("mc_summary", "motion_correction/summary.pdf"),

    # scratch
    s.reg.add("scratch", "scratch", is_dir=True),
    s.reg.add("mov_unprocessed", "scratch/mov_unprocessed.h5"),

    # segmentation
    s.reg.add("segmentation", "segmentation", is_dir=True),
    s.reg.add("suite2p", "suite2p", is_dir=True),

    # analysis files
    s.reg.add("analysis", "analysis", is_dir=True),

    # etc
    s.reg.add("root", "", is_dir=True)
    s.reg.add("attrs", "attrs.yaml"),
    s.reg.add("mov", "mov.h5"),
    s.reg.add("sample", "sample.mp4"),

    return s


"""
---------------------------------------------------------------------------------------------------------
"""



"""
-------------------------------------------------------------------------------------
- transitions
"""


class Transition:
    """
    Parses event info to find transitions from one element to another for a given session.

    """
    _events: Tuple["EventSchema.Event"]
    _P: Optional[float]  # probability of transition
    _kind: Optional[str]  # ?
    _session: Optional[Session]
    _splits: Optional[xr.DataArray]
    _arr: Optional[xr.DataArray]

    events = property(fget=lambda self: self._events)
    P = property(fget=lambda self: self._P)
    kind = property(fget=lambda self: self._kind)
    session = property(fget=lambda self: self._session)

    def __init__(
            self,
            events: Sequence[Union[int, str, "EventSchema.Event"]],
            P: Optional[float] = None,
            kind: Optional[str] = None,
    ):

        # coerce events to event objects.
        self._events = tuple([schema.get(event=ev) for ev in events])
        assert len(self._events) >= 2

        self._P = P
        self._kind = kind
        self._locs = None
        self._splits = None
        self._arr = None
        self.indata = None

    @property
    def arr(self) -> xr.DataArray:
        if self._arr is None:
            self._arr = xr.concat(self.splits, dim="time")
        return self._arr

    @property
    def splits(self) -> List[xr.DataArray]:
        if self._splits is None:
            self._parse()
        return self._splits

    def clear(self):
        self._session = None
        self._locs = None
        self._splits = None
        self._arr = None

    def bind(self, s: Session) -> None:
        self.clear()
        self._session = s

    def _parse(self) -> None:
        """
        Parses events and sets self._locs and self._splits. Clears all existing data (except for
        session) prior to running.

        """
        s = self._session
        self.clear()
        self._session = s

        # Find where transitions occur.
        event_df = s.events.tables["event"]
        event_ids = event_df["event_id"].values
        to_match = np.array([int(ev) for ev in self._events])
        inds = np.argwhere(event_ids == to_match[0])
        if inds.size:
            inds = inds.squeeze()
            self._locs = [ix for ix in inds if np.array_equal(event_ids[ix:ix + len(to_match)], to_match)]
            self._locs = np.array(self._locs, dtype=np.intp)
        else:
            self._locs = np.array([], dtype=np.intp)
            self._arr = xr.DataArray()
            self._splits = [xr.DataArray([[[]]], dims=("trial", "time", "roi")) for _ in range(len(self._events))]
            return

        # Extract chunks

        frames = self._session.events.tables["frame"]
        if self.indata is None:
            mat = np.load(s.fs.getsyspath("trace.npy"))
            mat = mat[:len(frames)]
        else:
            mat = self.indata

        start_col = event_df["start"].values

        self._splits = []
        for i, ev in enumerate(self._events):
            try:
                fps = s.attrs["capture"]["fps"]
            except KeyError:
                fps = s.attrs["capture"]["frame"]["rate"]
            starts = start_col[self._locs + i]
            stops = starts + round(fps * ev.duration)
            data = np.stack([mat[x:y] for x, y in zip(starts, stops)])
            ev_coords = np.array([ev] * data.shape[1], dtype=object)
            split = xr.DataArray(
                data,
                dims=("trial", "time"),
                coords={"event": xr.DataArray(ev_coords, dims=("time",))},
                name=ev,
            )
            self._splits.append(split)


def get_transitions(s: Optional[Session] = None) -> dict:
    d = {
        "AB": Transition(["A", "B"], P=0.9, kind="high"),
        "AC": Transition(["A", "C"], P=0.1, kind="low"),
        "BC": Transition(["B", "C"], P=0.8, kind="high"),
        "BD": Transition(["B", "D"], P=0.2, kind="low"),
        "CD": Transition(["C", "D"], P=0.7, kind="high"),
        "CE": Transition(["C", "E"], P=0.3, kind="low"),
        "DE": Transition(["D", "E"], P=0.6, kind="high"),
        "DA": Transition(["D", "A"], P=0.4, kind="low"),
        "EA": Transition(["E", "A"], P=0.5, kind="high"),
        "EB": Transition(["E", "B"], P=0.5, kind="low"),
    }

    if s is not None:
        for ts in d.values():
            ts.bind(s)
    return d
