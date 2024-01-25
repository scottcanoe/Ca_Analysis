import copy
import datetime
import logging
import dataclasses
import json
from numbers import Number
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
from typing import (
    Any,
    Callable,
    ClassVar,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)
import fsspec
import h5py
import logging
from natsort import natsorted
import pandas as pd
from matplotlib.animation import FFMpegWriter
import matplotlib.colors as mpc
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from scipy.stats import zscore
import xarray as xr

from ca_analysis import *
from ca_analysis.grouping import *
from ca_analysis.plot import *
from ca_analysis.traces import *
from ca_analysis.stats import *
from ca_analysis.roi import ROIFileStore
from main import get_transitions, open_session, schema
from ca_analysis.io.h5 import *
from processing import import_session

"""
-------------------------------------------------------------------------------------
- utils
"""


def get_splits(arr: xr.DataArray) -> List[xr.DataArray]:
    # Split array by event
    labels = [int(ev) for ev in arr.coords["event"]]
    inds = np.squeeze(np.argwhere(np.ediff1d(labels, to_begin=0) != 0))
    if inds.ndim == 0:
        # one split
        locs = [inds.item()]
    else:
        locs = [] if np.array_equal(inds, []) else list(inds)
    locs.insert(0, 0)
    locs.append(None)

    # split into chunks
    out = []
    for i in range(len(locs) - 1):
        chunk = arr.isel(time=slice(locs[i], locs[i + 1]))
        out.append(chunk)
    return out

"""
-------------------------------------------------------------------------------------
- plots
"""

def plot_heatmap_and_histogram(
    s: Session,
    trials: Optional[slice] = None,
    heatmap_path=None,
    histogram_path=None,
    ):
    """

    Parameters
    ----------
    s: Session
    trials: slice (optional)
        Provide a slice with start/stop values to use a subset of trials. start/stop values
    are normalized (i.e., slice(1/3, 2/3) would use only middle third of trials).
    heatmap_path: path-like (optional)
        Where to save heatmap.
    histogram_path: path-like (optional)
        Where to save histogram.

    Returns
    -------
    None
    """

    # construct trial-averaged arrays
    s.transitions = get_transitions(s)
    for T in s.transitions.values():
        data = T.arr
        n_trials = data.sizes["trial"]
        if trials is not None or trials != slice(None):
            start = round(trials.start * n_trials) if trials.start else None
            stop = round(trials.stop * n_trials) if trials.stop else None
            data = data.isel(trial=slice(start, stop))
        T.plot_arr = data.mean("trial")
        T.plot_arr = T.plot_arr.transpose("roi", "time")

    # Initialize figure and axes.
    chunklen = 6

    nrows, ncols = len(s.transitions) // 2, 3
    width, height = 9, 3 * nrows

    # preprocess plot data (sorting by oddball score, making difference data, etc.)
    specs = [("BC", "AC"), ("CD", "BD"), ("DE", "CE"), ("EA", "DA"), ("AB", "EB")]
    rows = []

    for i in range(nrows):
        sp = specs[i]
        # grab transitions, and build "difference transition"
        high = s.transitions[sp[0]].plot_arr
        high.name = sp[0]
        high.attrs["P"] = 0.9 - 0.1 * i
        high.attrs["Ptype"] = "HP"

        low = s.transitions[sp[1]].plot_arr
        low.name = sp[1]
        low.attrs["P"] = 0.1 + 0.1 * i
        low.attrs["Ptype"] = "LP"

        diff = low - high
        diff.name = f"{low.name} - {high.name}"
        diff.coords["event"] = high.coords["event"]
        diff.attrs["P"] = high.attrs["P"] - low.attrs["P"]
        diff.attrs["Ptype"] = "diff"

        # grab subregions of transitions for computing oddball score
        labels = [int(ev) for ev in high.coords["event"]]
        ind = np.squeeze(np.argwhere(np.ediff1d(labels, to_begin=0) != 0)).item()
        chunk_1 = diff.isel(time=slice(max(0, ind - chunklen), ind))  # before onset
        chunk_2 = diff.isel(time=slice(ind, ind + chunklen))  # after onset

        # compute oddball score: total absolute deflection (subtractive, not ratio)
        c1 = chunk_1.mean("time")
        c2 = chunk_2.mean("time")
        scores = c2 - c1

        # sort based on oddball score
        sorting_scores = chunk_2.mean("time")
        sort_order = np.flipud(np.squeeze(np.argsort(sorting_scores)))

        row = {
            "high": high,
            "low": low,
            "diff": diff,
            "scores": scores,
            "sort_order": sort_order,
        }
        rows.append(row)

    sort_order = rows[0]["sort_order"]
    for i, row in enumerate(rows):
        row["high"] = row["high"].isel(roi=sort_order)
        row["low"] = row["low"].isel(roi=sort_order)
        row["diff"] = row["diff"].isel(roi=sort_order)

    # build colormaps from pooled data
    reg_pooled, diff_pooled = [], []
    for i, row in enumerate(rows):
        reg_pooled.append(row["high"].data.flatten())
        reg_pooled.append(row["low"].data.flatten())
        diff_pooled.append(row["diff"].data.flatten())
    reg_pooled = np.hstack(reg_pooled)
    reg_vlim = np.percentile(reg_pooled, (2.5, 97.5))
    reg_norm = mpc.Normalize(vmin=reg_vlim[0], vmax=reg_vlim[1])
    reg_cmap = "inferno"

    diff_pooled = np.hstack(diff_pooled)
    diff_vlim = np.percentile(diff_pooled, (2.5, 97.5))
    diff_norm = mpc.Normalize(vmin=diff_vlim[0], vmax=diff_vlim[1])
    diff_cmap = "coolwarm"

    fig = plt.figure(figsize=(width, height))
    fig.tight_layout(pad=0)
    gs = fig.add_gridspec(nrows, ncols)

    def plot_one(arr, row, col, cmap, norm):
        ax = fig.add_subplot(gs[row, col])
        im = ax.imshow(
            arr,
            cmap=cmap,
            norm=norm,
            aspect="auto",
            interpolation="hanning",
        )

        midpoint = np.mean(ax.get_xlim())
        ax.axvline(midpoint, color="gray", ls="--")
        ax.set_xticks([])
        ax.set_yticks([])
        Ptype = arr.attrs.get("Ptype", None)
        if Ptype in {"HP", "LP"}:
            P = arr.attrs["P"]
            title = "$" + f"{Ptype}_{{" + "{:.1f}".format(P) + "}$" + f" ({arr.name})"
            ax.set_title(title)
        elif Ptype == "diff":
            P = arr.attrs["P"]
            title = "$\Delta P" + f"_{{" + "{:.1f}".format(P) + "}$" + f" ({arr.name})"
            ax.set_title(title)

        arr.attrs["ax"] = ax
        arr.attrs["im"] = im

    for i, row in enumerate(rows):
        high, low, diff = row["high"], row["low"], row["diff"]
        plot_one(high, i, 0, reg_cmap, reg_norm)
        plot_one(low, i, 1, reg_cmap, reg_norm)
        plot_one(diff, i, 2, diff_cmap, diff_norm)

    plt.show()
    if heatmap_path is not None:
        fig.savefig(heatmap_path)

    plt.rcParams.update({"font.size": 8})
    fig = plt.figure(figsize=(10, 8))
    fig.tight_layout()
    ax = fig.add_subplot()
    colors = ["black", "red", "teal", "blue", "gray"]
    linestyles = ["-", "-", "-", "-", "--"]
    for i, row in enumerate(rows):
        scores = row["scores"]
        X, Y = gaussian_kde(scores)
        arr = row["diff"]
        P = arr.attrs["P"]
        label = "$\Delta P" + f"_{{" + "{:.1f}".format(P) + "}$" + f" ({arr.name})"
        ax.plot(X, Y, color=colors[i], ls=linestyles[i], label=label)

    ax.legend()
    ax.set_xlim([-6, 6])
    ax.set_xlabel('$r$')
    ax.set_ylabel('density')
    plt.show()
    if histogram_path is not None:
        fig.savefig(histogram_path)

    # import scipy.stats
    # print("Rank Sum Tests (nov. excess ratios)\n------------------------")
    # chance_scores = rows[-1]["scores"]
    # for i in range(4):
    #     test_scores = rows[i]["scores"]
    #     pvalue = scipy.stats.ranksums(test_scores, chance_scores).pvalue
    #     name = rows[i]["diff"].name
    #     print(f"{name}: pvalue = {pvalue}")
    #
    # for i, row in enumerate(rows):
    #     scores = row["scores"]
    #     sc = scores.mean()
    #     print(f"{row['diff'].name} : {sc}")


def init_gach1_segmentation(s):


    s.segmentation_class = ROIFileStore
    seg = s.seg


    roilocs = []

    # define entire image mask
    roilocs.append((slice(0, 1024), slice(0, 1024)))

    # define central mask (all fluorescent regions)
    if not s.fs.exists("scratch/mask.npy"):
        create_mask(s)
    mask = np.load(s.fs.getsyspath("scratch/mask.npy"))
    Y, X = np.where(~mask)
    roilocs.append((Y, X))

    # if isinstance(ypix, slice) or isinstance(xpix, slice):
    #     yarr = np.arange(ypix.start, ypix.stop)
    #     xarr = np.arange(xpix.start, xpix.stop)
    #     Y, X = np.meshgrid(yarr, xarr)
    #     ypix, xpix = Y.flatten(), X.flatten()
    # else:
    #     ypix, xpix = np.asarray(ypix), np.asarray(xpix)

    # define cell masks
    roilocs.append((slice(415, 445), slice(320, 350)))
    roilocs.append((slice(278, 278+25), slice(500, 500+25)))
    roilocs.append((slice(329, 329+25), slice(578, 578+25)))
    roilocs.append((slice(326, 326+25), slice(606, 606+25)))

    # get time and movie arrays
    time = s.events.tables["frame"]["time"]
    with h5py.File(s.fs.getsyspath("mov.h5"), "r") as f:
        dset = f["data"]
        mov = dset[:len(time)]

    # build coordinates
    t = xr.DataArray(time, dims=("t",))
    y = xr.DataArray(np.arange(mov.shape[1]), dtype=np.intp, dims=("y",))
    x = xr.DataArray(np.arange(mov.shape[2]), dtype=np.intp, dims=("x",))
    seg.update_coords(t=t, y=y, x=x)

    # create summary images
    meanimg = mov.mean(axis=0)
    maximg = mov.max(axis=0)
    seg.add_image("meanimg", meanimg)
    seg.add_image("maximg", maximg)

    root = Path(s.fs.getsyspath(""))

    # create fluorescence traces
    F = np.zeros([mov.shape[0], len(roilocs)])
    for i in range(len(roilocs)):
        Y, X = roilocs[i]
        F_i = mov[:, Y, X]
        # F_i = np.array([mov[i, Y, X].mean() for i in range(mov.shape[0])])
        F[:, i] = F_i
        seg.add_roi(Y, X, series=dict(F=F_i))

    seg.save()


def create_mask(s: Session) -> np.ndarray:
    import imageio
    import subprocess as sp
    from skimage.color import gray2rgb

    # Find a movie we can use for the background image.
    scratch_dir = Path(s.reg.scratch)

    # Save the background as a jpeg.
    background = s.seg["meanimg"]
    background = np.asarray(background)
    background = (250 * (background / background.max())).astype(np.uint8)
    rgba = gray2rgb(background).astype(np.uint8)
    imageio.imsave(scratch_dir / "background.jpeg", rgba)

    np.save(scratch_dir / "background.npy", background)

    # Launch krita
    msg = """
    Mask Creation
    ------------
    - 1. Open {}/'background.jpeg'
    - 2. Use the Elliptical Selection Tool to select the ROI (probably everything
         that is window)
    - 3. Right click on the selection, and select 'Cut selection to new layer'.
    - 4. Make only the new layer visible.
    - 5. Save as '{}/mask.tif' and quit krita.

    """.format(scratch_dir, scratch_dir)
    print(msg, flush=True)

    tif_path = scratch_dir / "mask.tif"
    if tif_path.exists():
        tif_path.unlink()
    proc = sp.run(["krita", "--nosplash"], stdout=sp.PIPE, stderr=sp.STDOUT)
    if not proc:
        msg = "There was an error with krita: \n"
        raise RuntimeError(msg)

    # Load mask.
    mat = imageio.imread(tif_path)
    mask = np.zeros((mat.shape[0], mat.shape[1]), dtype=bool)
    mask[mat[:, :, -1] == 0] = True
    np.save(scratch_dir / "mask.npy", mask)

    return mask


def walk_and_print(path):
    for node in walk_h5(path):
        parts = node.name.strip("/").split("/")
        depth = len(parts) - 1
        name = parts[-1]
        attrs = dict(node.attrs)
        if isinstance(node, h5py.Group):
            cls_name = "Group"
            s = f"<Group '{name}': attrs={attrs}>"
        else:
            s = f"<Dataset '{name}': shape={node.shape}, dtype={node.dtype}, attrs={attrs}>"

        s = "  " * depth + s
        print(s)


__version__ = "0.0"

class Meta(NamedTuple):
    TYPE: str
    HANDLES: Optional[Union[type, Tuple[type, ...]]] = None

class Coder:

    meta: ClassVar[Meta]

    @classmethod
    def encode_attrs(cls, attrs: Mapping, **kw) -> bytes:
        attrs = {
            "TYPE": cls.meta.TYPE,
            "name": name if name else "",
        }
        return attrs

    @classmethod
    def decode_attrs(cls, bts: bytes, **kw) -> Mapping:
        attrs = {
            "TYPE": cls.meta.TYPE,
            "name": name if name else "",
        }
        return attrs


    @classmethod
    def create_attrs(cls, name: Optional[str] = None) -> Mapping:
        attrs = {
            "TYPE": cls.meta.TYPE,
            "name": name if name else "",
        }
        return attrs

    @classmethod
    def require_group(
        cls,
        parent: h5py.Group,
        key: str,
        attrs: Optional[Mapping] = None,
        clean: bool = True,
        **kw,
    ):
        group = parent.require_group(key)
        if clean:
            group.clear()
        if attrs:
            group.attrs.update(attrs)
        return group

    @classmethod
    def require_dataset(
        cls,
        data: ArrayLike,
        parent: h5py.Group,
        key: str,
        attrs: Optional[Mapping] = None,
        clean: bool = True,
        exact: bool = True,
        **kw,
    ):

        if key in parent and clean:
            del parent[key]
        try:
            dset = parent.require_dataset(key, data=data, exact=exact, **kw)
        except TypeError:
            parent.pop(key, None)
            dset = parent.create_dataset(key, data=data, **kw)

        if attrs:
            dset.attrs.update(attrs)
        return dset

    @classmethod
    def dump(
        cls,
        obj: Any,
        parent: h5py.Group,
        key: Optional[str] = None,
        **kw,
    ) -> h5py.Dataset:
        raise NotImplementedError

    @classmethod
    def load(cls, obj: Any, *args, **kw):
        raise NotImplementedError

_typename_to_coder = {}
_handled_to_coder = {}

def register_coder(cls: type) -> type:

    typename = cls.meta.TYPE
    handles = cls.meta.HANDLES
    _typename_to_coder[typename] = cls
    if handles:
        handles = (handles,) if isinstance(handles, type) else handles
    for h in handles:
        _handled_to_coder[h] = cls
    return cls


def get_coder(obj: Any, *args, **kw) -> Coder:

    # If simply given a name, look it up.
    if isinstance(obj, str):
        return _typename_to_coder[obj]

    # If it's on disk, look at the TYPE attribute for the name.
    if isinstance(obj, (h5py.Group, h5py.Dataset)):
        typename = obj.attrs.get("TYPE")
        return _typename_to_coder[typename]

    h = type(obj)
    return _handled_to_coder[h]


def dump(obj: Any, *args, **kw) -> None:
    coder = get_coder(obj, *args, **kw)
    return coder.dump(obj, *args, **kw)


def load(obj, *args, **kw) -> Any:
    coder = get_coder(obj, *args, **kw)
    return coder.load(obj, *args, **kw)




def generate_key(group: h5py.Group, prefix: str) -> str:
    count = 0
    key = f"{prefix}{count}"
    while key in group:
        count += 1
        key = f"{prefix}{count}"
    return key



@register_coder
class NDArrayCoder(Coder):

    meta = Meta(
        TYPE="numpy:ndarray",
        HANDLES=(np.ndarray,),
    )

    @classmethod
    def dump(
        cls,
        arr: np.ndarray,
        parent: h5py.Group,
        key: str,
        **kw,
    ) -> h5py.Group:

        attrs = cls.create_attrs()
        attrs["CLASS"] = arr.__class__.__name__
        data = np.asarray(arr)
        return cls.require_dataset(data, parent, key, attrs, **kw)

    @classmethod
    def load(cls, obj: h5py.Dataset, *args, **kw) -> np.ndarray:

        attrs = dict(obj.attrs)
        data = obj[:]
        return data



@register_coder
class IndexCoder(Coder):

    meta = Meta(
        TYPE="pandas:index",
        HANDLES=(
            pd.Index,
            pd.RangeIndex,
            pd.Float64Index,
            pd.Int64Index,
            pd.UInt64Index,
            pd.IntervalIndex,
        ))

    @classmethod
    def dump(
        cls,
        index: pd.Index,
        parent: h5py.Group,
        key: Optional[str] = None,
        **kw,
    ) -> h5py.Dataset:

        attrs = cls.create_attrs(name=index.name)
        attrs["CLASS"] = index.__class__.__name__
        key = key or attrs["name"]
        if not key:
            raise ValueError("object must be named or you must provide a key")

        if isinstance(index, pd.RangeIndex):
            data = np.array([index.start, index.stop, index.step])
        elif isinstance(index, (pd.Float64Index, pd.Int64Index, pd.UInt64Index)):
            data = index.to_numpy()
        elif isinstance(index, pd.IntervalIndex):
            data = np.vstack([index.left, index.right]).T
            attrs["closed"] = index.closed
        else:
            raise NotImplementedError

        dset = cls.require_dataset(
            data,
            parent,
            key,
            attrs=attrs,
            **kw,
        )
        return dset


    @classmethod
    def load(cls, obj: h5py.Dataset, *args, **kw) -> pd.Index:

        attrs = dict(obj.attrs)
        data = obj[:]
        CLASS = attrs["CLASS"]

        if CLASS == "RangeIndex":
            index = pd.RangeIndex(data[0], data[1], data[2])
        elif CLASS in ("Float64Index", "Int64Index", "UInt64Index"):
            cls = getattr(pd, CLASS)
            index = cls(data)
        elif CLASS == "IntervalIndex":
            left, right = data[:, 0], data[:, 1]
            closed = attrs["closed"]
            index = pd.IntervalIndex.from_arrays(
                left, right, closed=closed,
            )
        else:
            raise NotImplementedError
        index.name = attrs.pop("name", None) or None
        return index


@register_coder
class SeriesCoder(Coder):

    meta = Meta(
        TYPE="pandas:series",
        HANDLES=(pd.Series,),
    )

    @classmethod
    def dump(
        cls,
        s: pd.Series,
        parent: h5py.Group,
        key: Optional[str] = None,
        **kw,
    ) -> h5py.Group:

        attrs = cls.create_attrs(name=s.name)
        attrs["CLASS"] = s.__class__.__name__
        key = key or attrs["name"]
        if not key:
            raise ValueError("object must be named or you must provide a key")

        group = cls.require_group(parent, key, attrs, **kw)
        IndexCoder.dump(s.index, group, "index", **kw)
        NDArrayCoder.dump(s.to_numpy(), group, "array", **kw)
        return group


    @classmethod
    def load(cls, obj: h5py.Group, *args, **kw) -> pd.Series:

        attrs = dict(obj.attrs)
        index = IndexCoder.load(obj["index"])
        data = NDArrayCoder.load(obj["array"])
        name = attrs.pop("name", None) or None
        s = pd.Series(data, index, name=name)
        s.attrs.update(attrs)
        return s


@register_coder
class DataFrameCoder(Coder):

    meta = Meta(
        TYPE="pandas:dataframe",
        HANDLES=(pd.DataFrame,),
    )

    @classmethod
    def dump(
        cls,
        df: pd.DataFrame,
        parent: h5py.Group,
        key: str,
        **kw,
    ) -> h5py.Group:

        attrs = cls.create_attrs(name=key)
        attrs["CLASS"] = df.__class__.__name__
        key = key or attrs["name"]
        if not key:
            raise ValueError("object must be named or you must provide a key")

        group = cls.require_group(parent, key, attrs, **kw)
        IndexCoder.dump(df.index, group, "index", **kw)
        for col in df.columns:
            data = df[col].to_numpy()
            dset = NDArrayCoder.dump(data, group, col, **kw)
            d = {
                "TYPE": "pandas.column",
                "CLASS": "numpy.ndarray",
                "name": col,
            }
            dset.attrs.update(d)
        return group


    @classmethod
    def load(cls, group: h5py.Group, *args, **kw) -> pd.Series:

        attrs = dict(group.attrs)
        name = attrs.pop("name", None)
        index = IndexCoder.load(group["index"])
        df = pd.DataFrame(index=index)
        df.attrs.update(attrs)

        for key, dset in group.items():

            if dset.attrs.get("TYPE") == "pandas.column":
                d = dict(dset.attrs)
                arr = dset[:]
                # if attrs["name"] != key:
                #     print('something funny happened')
                df[key] = arr
                s = df[key]
                s.attrs.update(d)

        return df


def check_series(a, b):
    pd.testing.assert_series_equal(
        a,
        b,
        check_index_type=True,
        check_exact=True,

    )

def check_frames(a, b):
    pd.testing.assert_frame_equal(
        a,
        b,
        check_index_type=True,
        check_exact=True,

    )



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

    @property
    def arr(self) -> xr.DataArray:
        if self._arr is None:
            self._arr = xr.concat(self.splits, dim="time")
        return self._arr

    @property
    def splits(self) -> List[xr.DataArray]:
        if self._splits is None:
            self.parse()
        return self._splits

    def clear(self):
        self._session = None
        self._locs = None
        self._splits = None
        self._arr = None

    def bind(self, s: Session) -> None:
        self.clear()
        self._session = s

    def parse(self, mat) -> None:
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

        start_col = event_df["start"].values

        self._splits = []
        for i, ev in enumerate(self._events):
            starts = start_col[self._locs + i]
            fps = s.attrs["capture"]["frame"]["rate"]
            stops = starts + round(fps * ev.duration)
            data = np.stack([mat[x:y] for x, y in zip(starts, stops)])
            ev_coords = np.array([ev] * data.shape[1], dtype=object)
            split = xr.DataArray(
                data,
                dims=("trial", "time", "roi"),
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

if __name__ == "__main__":

    s = gach1 = open_session("25451-1", "2021-12-07", fs="ssd")  # gach mouse
    s.segmentation_class = ROIFileStore
    # init_gach1_segmentation(s)

    s.segmentation_class = ROIFileStore
    seg = s.seg

    roilocs = []

    # define entire image mask
    roilocs.append((slice(0, 1024), slice(0, 1024)))

    # define central mask (all fluorescent regions)
    if not s.fs.exists("scratch/mask.npy"):
        create_mask(s)
    mask = np.load(s.fs.getsyspath("scratch/mask.npy"))
    Y, X = np.where(~mask)
    roilocs.append((Y, X))

    # define cell masks
    roilocs.append((slice(415, 445), slice(320, 350)))
    roilocs.append((slice(278, 278 + 25), slice(500, 500 + 25)))
    roilocs.append((slice(329, 329 + 25), slice(578, 578 + 25)))
    roilocs.append((slice(326, 326 + 25), slice(606, 606 + 25)))

    # get time and movie arrays
    # time = s.events.tables["frame"]["time"]
    # with h5py.File(s.fs.getsyspath("mov.h5"), "r") as f:
    #     dset = f["data"]
    #     mov = dset[:len(time)]

    # build coordinates
    # t = xr.DataArray(time, dims=("t",))
    # y = xr.DataArray(np.arange(mov.shape[1]), dtype=np.intp, dims=("y",))
    # x = xr.DataArray(np.arange(mov.shape[2]), dtype=np.intp, dims=("x",))
    # seg.update_coords(t=t, y=y, x=x)

    # create summary images
    # meanimg = mov.mean(axis=0)
    # maximg = mov.max(axis=0)
    # seg.add_image("meanimg", meanimg)
    # seg.add_image("maximg", maximg)

    # root = Path(s.fs.getsyspath(""))
    # arrs = []
    # T = mov.shape[0]
    # R = len(roilocs)
    # # create fluorescence traces
    # F = np.zeros([T, R])
    # for i in range(len(roilocs)):
    #     Y, X = roilocs[i]
    #     F_i = np.array([mov[i, Y, X].mean() for i in range(mov.shape[0])])
    #     F[:, i] = np.array([mov[i, Y, X].mean() for i in range(mov.shape[0])])
    #     arrs.append(F)

    # F = np.vstack(F)
    # path = root / "F.npy"
    # np.save(path, F)
    # chat1 = open_session("19087-1", "2021-12-07", fs="ssd"),  # chat mouse

    F = np.load(s.fs.getsyspath("F.npy"))
    F = F[1:]
    F = zscore(F, axis=0)
    frames = s.events.tables["frame"]
    time = frames["time"]
    transitions = get_transitions(s)
    for key, val in transitions.items():
        val.parse(F)


    mode = 0
    chunklen = 8

    # for T in transitions.values():
    #     data = T.arr
    #     n_trials = data.sizes["trial"]
    #     if mode == 0:
    #         data = data
    #     elif mode == 1:
    #         data = data.isel(trial=slice(0, n_trials // 3))  # front third
    #     elif mode == 2:
    #         data = data.isel(trial=slice(n_trials // 3, 2 * n_trials // 3))  # middle third
    #     elif mode == 3:
    #         data = data.isel(trial=slice(2 * n_trials // 3, None))  # back third
    #     else:
    #         raise ValueError
    #     T.temp_arr = data

    # mean activity on target element for low- vs high-probability transitions
    high_prob_names = ["BC", "CD", "DE", "EA", "AB"]
    low_prob_names = ["AC", "BD", "CE", "DA", "EB"]
    HP, LP = [], []
    for i in range(len(high_prob_names)):

        data = transitions[high_prob_names[i]].splits[1]
        n_trials = data.sizes["trial"]
        if mode == 0:
            data = data
        elif mode == 1:
            data = data.isel(trial=slice(0, n_trials // 3))  # front third
        elif mode == 2:
            data = data.isel(trial=slice(n_trials // 3, 2 * n_trials // 3))  # middle third
        elif mode == 3:
            data = data.isel(trial=slice(2 * n_trials // 3, None))  # back third
        else:
            raise ValueError
        data = data.isel(time=slice(chunklen))
        means = data.mean(dim="time").data.flatten()
        HP.append(means)

        data = transitions[low_prob_names[i]].splits[1]
        n_trials = data.sizes["trial"]
        if mode == 0:
            data = data
        elif mode == 1:
            data = data.isel(trial=slice(0, n_trials // 3))  # front third
        elif mode == 2:
            data = data.isel(trial=slice(n_trials // 3, 2 * n_trials // 3))  # middle third
        elif mode == 3:
            data = data.isel(trial=slice(2 * n_trials // 3, None))  # back third
        else:
            raise ValueError
        data = data.isel(time=slice(chunklen))
        means = data.mean(dim="time").data.flatten()
        LP.append(means)

    H = np.hstack(HP)
    L = np.hstack(LP)

    fig, ax = plt.subplots()
    X = np.linspace(-2, 2, 1000)
    X, Y = gaussian_kde(H, X)
    ax.plot(X, Y, color='k', label='HP')

    X, Y = gaussian_kde(L, X)
    ax.plot(X, Y, color='r', label='LP')

    ax.legend()
    ax.set_ylabel("density")
    ax.set_xlabel("Fluorescence (z-scored)")
    ax.set_xlim([-2, 2])
    fig.savefig("/home/scott/update_plots/ach.eps")
    plt.show()

