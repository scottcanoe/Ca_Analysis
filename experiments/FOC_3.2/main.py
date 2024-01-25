
import datetime
import functools
import logging
import os
from typing import (
    Any,
    Callable, Iterable, Iterator, List,
    Mapping, Optional,
    Sequence,
    Tuple,
    Union,
)
from fs.errors import *
import numpy as np
import pandas as pd

from ca_analysis import *
from ca_analysis.environment import get_fs
from ca_analysis.indexing import *
from ca_analysis.io.suite2p import Suite2PStore


__experiment__: str = "FOC_3.2"

schema_path = get_fs(0).getsyspath(f"event_schemas/{__experiment__}.yaml")
schema = EventSchema(get_fs(0).getsyspath(f"event_schemas/{__experiment__}.yaml"))
del schema_path

logger = logging.getLogger(f"ca_analysis.experiments.{__experiment__}")


def open_session(
    mouse: str,
    date: Union[str, datetime.date],
    exp: Union[int, str] = "1",
    fs: Optional[Union[int, str]] = 0,
    require: bool = False,
    cell_mode: bool = True,
) -> Session:
    date = as_date(date)
    exp = str(exp)
    stem = os.path.join("sessions", mouse, str(date), exp)

    parent_fs = get_fs(fs)
    try:
        fs = parent_fs.opendir(stem)
    except ResourceNotFound:
        if not require:
            raise
        fs = parent_fs.makedir(stem)

    s = Session(fs, mouse=mouse, date=date, exp=exp)
    s.data = DataHandler(s)
    s.event_class = EventModel
    s.segmentation_class = Suite2PStore

    # thorlabs
    s.reg.add("thorlabs", "thorlabs")
    s.reg.add("thor_md", "thorlabs/Experiment.xml"),
    s.reg.add("thor_raw", "thorlabs/Image.raw"),
    s.reg.add("thor_sync", "thorlabs/Episode.h5"),

    # events
    s.reg.add("events", "events")
    s.reg.add("schema", "events/schema.yaml")
    s.reg.add("frames_table", "events/frames.csv")
    s.reg.add("events_table", "events/events.csv")

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
        pair_mode: str = "post",
        target: str = "spikes",
    ):

        self._session = session
        self.cell_mode = cell_mode
        self.pair_mode = pair_mode
        self.target = target
        self.attrs = {}
        self._transitions = None
        self._transition_pairs = None

        self._prepared = False

    @property
    def session(self) -> Session:
        return self._session


    def cell_indices(self) -> ArrayLike:
        return argwhere(self.get("iscell"))

    @property
    def n_cells(self) -> int:
        return self.cell_indices().size

    @property
    def n_rois(self) -> int:
        return len(self.get("iscell"))

    @property
    def n_frames(self) -> int:
        return self.get("n_frames")

    @property
    def transitions(self) -> Mapping[str, "Transition"]:
        self._ensure_prepared()
        return self._transitions

    @property
    def transition_pairs(self) -> Mapping[str, "TransitionPair"]:
        self._ensure_prepared()
        return self._transition_pairs

    def clear(self):
        self._prepared = False
        if self._transitions:
            for tr in self._transitions.values():
                tr.clear()
        if self._transition_pairs:
            for p in self._transition_pairs.values():
                p.clear()

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
        cell_mode = self.cell_mode
        if cell_mode:
            inds = argwhere(self.get("iscell"))
            arr = arr.isel(roi=inds)
        n_frames = self.get("n_frames")
        arr = arr.isel(time=slice(0, n_frames))

        # Add time coordinates (such as event_id)
        coords = dict(arr.coords)
        time_coords = self.get("time_coords")
        coords["time"] = time_coords["time"]
        coords["event_id"] = time_coords["event_id"]

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
        schema = self.get("schema")
        events = EventSequence(schema, events=events).events
        fps = self.get("fps")
        ev_lengths = [round(ev.duration * fps) for ev in events]
        df = self.get("event_table")

        # Optionally apply block filter to event table.
        if blocks is not None:
            block_ids = df.block
            if is_int(blocks):
                rows = argwhere(block_ids == blocks)
            elif isinstance(blocks, slice):
                valid_blocks = np.arange(block_ids.max() + 1)[blocks]
                rows = np.in1d(block_ids, valid_blocks)
            else:
                rows = np.in1d(block_ids, np.array(blocks, dtype=int))
            df = df.iloc[rows]

        # find locations of matching sequences, and add block and event info.
        matches = find_sequences(events.astype(int), df.event_id)
        starts = matches["start"].values
        stops = matches["stop"].values
        block_ids = np.zeros(len(matches), dtype=int)

        # build a grid of slice objects with shape (n_trials, n_events)
        # where each entry slices frames.
        slices = np.zeros([len(matches), len(events)], dtype=object)
        for i in range(len(matches)):
            sub_df = df.iloc[slice(starts[i], stops[i])]
            slices[i] = [slice(row[1].start, row[1].start + ev_lengths[j])
                         for j, row in enumerate(sub_df.iterrows())]
            block_ids[i] = sub_df.iloc[0].block

        # put in a data array to hold extra metadata, like block id.
        coords = {
            "block": xr.DataArray(block_ids, dims=("trial",)),
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

        if is_str(events):
            return_one, events = True, [events]
        else:
            return_one = False

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
                "block": ev_slices.coords["block"],
                "roi": target.coords["roi"],
                "event": ev_slices.event,
            }
            arr = xr.DataArray(mat, dims=dims, coords=coords)
            splits.append(arr)

        return splits[0] if return_one else splits

    def _ensure_prepared(self):

        if self._prepared:
            return

        F = self._session.segmentation["F"]
        n_frames_data = F.sizes["time"]
        frames = self._session.events.tables["frame"]
        n_frames_sync = len(frames)
        events = self._session.events.tables["event"]
        if n_frames_data == n_frames_sync:
            trim = None
            n_frames = n_frames_data
        elif n_frames_data < n_frames_sync:
            trim = "sync"
            n_frames = n_frames_data
        else:
            trim = "data"
            n_frames = n_frames_sync
        frames = frames.iloc[slice(0, n_frames)]
        events = events[events["stop"] < n_frames]
        self.attrs["trim"] = trim
        self.attrs["n_frames"] = n_frames
        self.attrs["frame_table"] = frames
        self.attrs["event_table"] = events

        try:
            fps = self._session.attrs["capture"]["frame"]["rate"]
        except KeyError:
            fps = self._session.events.get_fps()
        self.attrs["fps"] = fps

        self._transitions = get_transitions(self)
        self._transition_pairs = pair_transitions(self._transitions, self.pair_mode)

        vars = {}
        for name in frames.columns:
            arr = xr.DataArray(frames[name].values, dims=("time",), name=name)
            arr = arr.isel(time=slice(0, n_frames))
            vars[name] = arr
        del vars["event_value"]
        del vars["event_strobe"]
        self.attrs["time_coords"] = xr.Dataset(vars)

        self._prepared = True


class Transition:
    """
    Parses event info to find transitions from one element to another for a
    given session.

    """

    def __init__(
        self,
        name: str,
        pre: str,
        post: str,
        P: Optional[float] = None,
        kind: Optional[str] = None,
        parent: Optional[DataHandler] = None,
    ):
        self.name = name
        self.pre_name = pre
        self.post_name = post
        self.P = P
        self.kind = kind
        self.parent = parent
        self._pre = self._post = None
        self.scores = None
        self.temp = None

    @property
    def pre(self) -> xr.DataArray:
        # lazy loading
        if self._pre is None and self.parent is not None:
            self._load()
        return self._pre

    @pre.setter
    def pre(self, obj: xr.DataArray) -> None:
        self._pre = obj

    @property
    def post(self) -> xr.DataArray:
        # lazy loading
        if self._post is None and self.parent is not None:
            self._load()
        return self._post

    @post.setter
    def post(self, obj: xr.DataArray) -> None:
        self._post = obj

    @property
    def data(self) -> Tuple:
        return [self.pre, self.post]

    @property
    def events(self) -> Tuple[str, str]:
        return self.pre_name, self.post_name

    def clear(self, *args):
        self.scores = None
        self.temp = None
        for attr in args:
            setattr(self, attr, None)

    def _load(self):
        self._pre, self._post = self.parent.split(self.events, blocks=[2, 3, 4])
    def __getitem__(self, key):
        return [self.pre, self.post][key]

    def __repr__(self):
        return f"Transition: {self.name}"


class TransitionPair:
    name: str
    high: Transition
    low: Transition
    mode: str
    parent: Optional[Union[Session, List[Session]]]
    scores: Optional[ArrayLike]
    label: Optional[str] = None

    def __init__(
        self,
        name: str,
        high: Transition,
        low: Transition,
        mode: str,
        parent: Optional[DataHandler] = None,
    ):

        self.name = name
        self.high = high
        self.low = low
        self.mode = mode
        self.parent = parent
        self.scores = None
        self.temp = None

    def clear(self):
        self.high.clear()
        self.low.clear()
        self.scores = None
        self.temp = None

    def __repr__(self):
        return f"TransitionPair: high={self.high}, low={self.low}"


def get_transitions(
    parent: Optional[DataHandler] = None,
) -> Mapping[str, Transition]:

    transition_args = {
        "AB": dict(name="AB", pre="A", post="B", P=0.9, kind="high"),
        "AC": dict(name="AC", pre="A", post="C", P=0.1, kind="low"),
        "BC": dict(name="BC", pre="B", post="C", P=0.8, kind="high"),
        "BD": dict(name="BD", pre="B", post="D", P=0.2, kind="low"),
        "CD": dict(name="CD", pre="C", post="D", P=0.7, kind="high"),
        "CE": dict(name="CE", pre="C", post="E", P=0.3, kind="low"),
        "DE": dict(name="DE", pre="D", post="E", P=0.6, kind="high"),
        "DA": dict(name="DA", pre="D", post="A", P=0.4, kind="low"),
        "EA": dict(name="EA", pre="E", post="A", P=0.5, kind="high"),
        "EB": dict(name="EB", pre="E", post="B", P=0.5, kind="low"),
    }
    out = {}
    for key, val in transition_args.items():
        out[val["name"]] = Transition(
                        val["name"],
                        val["pre"],
                        val["post"],
                        P=val["P"],
                        kind=val["kind"],
                        parent=parent,
                        )
    return out


def get_transition_pairs(
    mode: str = "post",
    parent: Optional[DataHandler] = None,
) -> Mapping[str, TransitionPair]:

    T = get_transitions(parent)
    return pair_transitions(T, mode)


def pair_transitions(
    T: Mapping[str, Transition],
    mode: str = "post",
) -> Mapping[str, TransitionPair]:

    if mode == "pre":
        args = {
            "A": dict(name="A", high="AB", low="AC"),
            "B": dict(name="B", high="BC", low="BD"),
            "C": dict(name="C", high="CD", low="CE"),
            "D": dict(name="D", high="DE", low="DA"),
            "E": dict(name="E", high="EA", low="EB"),
        }
    elif mode == "post":
        args = {
            "A": dict(name="A", high="EA", low="DA"),
            "B": dict(name="B", high="AB", low="EB"),
            "C": dict(name="C", high="BC", low="AC"),
            "D": dict(name="D", high="CD", low="BD"),
            "E": dict(name="E", high="DE", low="CE"),
        }
    else:
        raise ValueError(f'invalid mode: {mode}')

    dct = {}
    for key, val in args.items():
        p = TransitionPair(
            val["name"],
            T[val["high"]],
            T[val["low"]],
            mode=mode,
        )
        dct[val["name"]] = p
    return dct


"""
--------------------------------------------------------------------------------
"""

def ensure_iterable_first_argument(fn):
    @functools.wraps(fn)
    def new_fn(obj, *args, **kw):
        if not hasattr(obj, "__iter__"):
            obj = [obj]
        return fn(obj, *args, **kw)
    return new_fn


def handle_transition_names(name: Optional[Iterable[str]]) -> List[str]:
    default = ['AB', 'AC', 'BC', 'BD', 'CD', 'CE', 'DE', 'DA', 'EA', 'EB']
    if name is None:
        return default
    elif is_str(name):
        return [name]
    else:
        return list(name)


def handle_transition_pair_names(name: Optional[Iterable[str]]) -> List[str]:
    default = ['A', 'B', 'C', 'D', 'E']
    if name is None:
        return default
    elif is_str(name):
        return [name]
    else:
        return list(name)


def load_spikes(
    ses: Union[Session, List[Session]],
    trans: Optional[Union[str, Sequence[str]]] = None,
    block: IndexLike = slice(None),
) -> Mapping:
    import sys
    from time import perf_counter as clock

    sys.stdout.write('Loading spikes... ')
    t_start = clock()

    ses = [ses] if isinstance(ses, Session) else ses

    if not trans:
        ok_names = ses[0].data.transitions.keys()
    elif is_str(trans):
        ok_names = [trans]
    else:
        ok_names = trans

    for s in ses:
        for name in ok_names:
            tr = s.data.transitions[name]
            tr.data = s.data.split(tr.events, target="spikes", block=block)

    t_stop = clock()
    string = 'completed in {:.2f} seconds \n'.format(t_stop - t_start)
    sys.stdout.write(string)


def itertransitions(
    ses: Union[Session, List[Session]],
    name: Optional[str] = None,
) -> Iterator:
    ses = [ses] if isinstance(ses, Session) else ses

    if name is None:
        ok_names = get_transitions().keys()
    elif is_str(name):
        ok_names = [name]
    else:
        ok_names = name

    for s in ses:
        for n in ok_names:
            yield s.data.transitions[n]


def iterpairs(
    ses: Union[Session, List[Session]],
    name: Optional[str] = None,
) -> Iterator:
    ses = [ses] if isinstance(ses, Session) else ses
    if name is None:
        ok_names = get_transition_pairs().keys()
    elif is_str(name):
        ok_names = [name]
    else:
        ok_names = name

    for s in ses:
        for n in ok_names:
            yield s.data.transition_pairs[n]


sessions_FOC_3_1 = [
    open_session("611486-1", "2021-07-13", fs="ssd"),
    open_session("611521-1", "2021-07-14", fs="ssd"),
    open_session("619739-1", "2021-08-03", fs="ssd"),
    open_session("619745-2", "2021-10-19", fs="ssd"),
    open_session("619766-4", "2021-10-19", fs="ssd"),
    open_session("619816-2", "2021-11-02", fs="ssd"),
]

sessions_FOC_3_2 = [
    open_session("12488-1", "2022-02-09", "1", fs="ssd"),
    # open_session("12489-1", "2022-02-09", "1", fs="ssd"), # bad quality?
    open_session("19075-3", "2022-02-09", "1", fs="ssd"),
    open_session("19088-2", "2022-02-09", "1", fs="ssd"),
    open_session("19089-1", "2022-02-09", "1", fs="ssd"),
]

all_sessions = sessions_FOC_3_1 + sessions_FOC_3_2


def get_sessions(name: Optional[str] = None) -> List[Session]:
    if name is None:
        return all_sessions
    if name == "3.1":
        return sessions_FOC_3_1
    if name == "3.2":
        return sessions_FOC_3_2
    raise ValueError(name)


def pool_flat(lst):
    return np.hstack([np.asarray(mat).reshape(-1) for mat in lst])


class GetBlocks:

    def __init__(self, blocks=None):
        if blocks is None or blocks == slice(None):
            self._blocks = None
        elif is_int(blocks):
            self._blocks = [blocks]
        else:
            self._blocks = blocks

    def __call__(self, arr: xr.DataArray) -> xr.DataArray:
        blocks = self._blocks
        if blocks is None or blocks == slice(None):
            return arr
        block_ids = np.arange(arr.block_id.max().item() + 1)
        valid_blocks = block_ids[blocks]
        tf = np.in1d(arr.block_id, valid_blocks)
        return arr[tf]


class Mean:

    def __init__(self, dim: str):
        self.dim = dim

    def __call__(self, arr: xr.DataArray) -> xr.DataArray:
        return arr.mean(self.dim)


class ISel:

    def __init__(self, dim: str, slc=slice(None)):
        self.dim = dim
        self.slc = slc

    def __call__(self, arr: xr.DataArray) -> xr.DataArray:
        if self.slc is None or self.slc == slice(None):
            return arr
        return arr.isel({self.dim: self.slc})


@ensure_iterable_first_argument
def make_transition_scores(
    ses: Iterable[Session],
    score_fn: Callable = lambda pre, post: post,
    blocks: Optional[IndexLike] = [2, 3, 4],
    pre_sel: Optional[IndexLike] = None,
    post_sel: Optional[IndexLike] = None,
    name: Optional = None,
) -> Mapping[str, Transition]:

    block_filter = GetBlocks(blocks)
    pre_funcs = [block_filter, ISel(pre_sel), Mean('time'), Mean('trial')]
    post_funcs = [block_filter, ISel(post_sel), Mean('time'), Mean('trial')]
    ok_names = handle_transition_names(name)

    # make scores for each transition
    for s in ses:
        for key in ok_names:
            tr = s.data.transitions[key]
            pre = toolz.pipe(tr.pre, *pre_funcs)
            post = toolz.pipe(tr.post, *post_funcs)
            tr.scores = score_fn(pre, post)

    # pool them
    T = get_transitions()
    for key in T.keys():
        if key in ok_names:
            lst = []
            for s in ses:
                lst.append(s.data.transitions[key].scores)
            T[key].scores = xr.concat(lst, "roi")
        else:
            del T[key]
    return T


@ensure_iterable_first_argument
def make_transition_pair_scores(
    ses: Iterable[Session],
    score_fn: Callable = lambda high, low: low - high,
    name: Optional = None,
) -> Mapping[str, TransitionPair]:

    ok_names = handle_transition_pair_names(name)

    # make scores for each transition pair
    for s in ses:
        for key in ok_names:
            p = s.data.transition_pairs[key]
            p.scores = score_fn(p.high.scores, p.low.scores)

    # pool them
    P = get_transition_pairs()
    for key in P.keys():
        if key in ok_names:
            lst = []
            for s in ses:
                lst.append(s.data.transition_pairs[key].scores)
            P[key].scores = xr.concat(lst, "roi")
        else:
            del P[key]
    return P

