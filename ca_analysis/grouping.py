import copy
from numbers import Number
from typing import (
    Callable,
    Iterable,
    List,
    Optional,
    Sequence,
    Union,
)

import h5py
import pandas as pd
import numpy as np
import xarray as xr

from .common import *
from .event import *

__all__ = [
    "Grouper",
    "Group",
    "Pad",
    "FPad",
    "BPad",
]


class Grouper:
    """
    Tool for grouping data by event/sequence-based data into rectangular arrays.
    
    """

    def __init__(self, event_model: EventModel):

        self._event_model = event_model
        self._schema = self._event_model.schema
        self._dfs = {}
        self._fps = None

    def group(
        self,
        data: ArrayLike,
        specs: Union[str, "Group", Iterable[Union[str, "Group"]]],
        align: Union[int, str, "Quantity", Callable] = "fps",
        new_dim: str = "trial",
    ) -> Union[xr.DataArray, List[xr.DataArray]]:

        """
        Note that align only applies to events, not padding. Also, any selections
        manually created with align specified will retain their align.
        
        """

        # Get fps as a regular number or `None`. This may or not be used, but
        # doing this here prevents downstream methods from having to do this
        # step repeatedly.

        # ---------------------------------------------------------------------------#
        # Get grouping specs as a flat list of empty Group objects.
        dimsize = data.shape[0]
        fps = self._get_fps()

        groups = self._init_groups(specs, align, dimsize, fps=fps)
        self._init_pads(groups, align, dimsize, fps=fps)

        # Extract and align data for each group.
        self._extract(groups, data)

        # Rewrap data.
        if isinstance(data, xr.DataArray):
            new_dims = [new_dim] + list(data.dims)
            new_coords = dict(data.coords)
        else:
            new_dims = [new_dim] + [f"dim_{i + 1}" for i in range(data.ndim)]
            new_coords = {}

        out = []
        for g in groups:
            arr = xr.DataArray(g.data, dims=new_dims, coords=new_coords, name=g.name)
            out.append(arr)

        if len(out) == 1:
            return out[0]

        return out

    def group_frames(
        self,
        data: ArrayLike,
        specs: Union[str, "Group", Iterable[Union[str, "Group"]]],
        align: Union[int, str, "Quantity", Callable] = "fps",
        new_dim: str = "trial",
    ) -> Union[xr.DataArray, List[xr.DataArray]]:

        out = self.group(data, specs, align=align)
        out = self._wrap_output(out, ["trial", "time", "ypix", "xpix"])
        return out

    def group_traces(
        self,
        data: ArrayLike,
        specs: Union[str, "Group", Iterable[Union[str, "Group"]]],
        align: Union[int, str, "Quantity", Callable] = "fps",
    ) -> Union[xr.DataArray, List[xr.DataArray]]:

        out = self.group(data, specs, align=align)
        out = self._wrap_output(out, ["trial", "time", "roi"])
        return out

    """
    - Indexer building
    ---------------------------------
    """

    def _init_groups(
        self,
        specs: Union[str, "Group", Iterable[Union[str, "Group"]]],
        align: Optional[Union[int, str, "Quantity", Callable]],
        dimsize: int,
        fps: Optional[Number] = None,
    ) -> List["Group"]:

        """
        Initialize list of group objects associated with a given a grouping spec.
        
        Converts one or more group specs to a list of group objects, all of them
        having a 'name' and 'event' attribute (except for Pad objects which also
        have a 'ref' attribute)        
        """

        if not hasattr(specs, "__iter__") and not isinstance(specs, str):
            specs = [specs]

        groups = []
        schema = self._schema
        for i, obj in enumerate(specs):

            if isinstance(obj, Group):
                groups.append(copy.copy(obj))
                continue
            try:
                event = schema.get(event=obj)
                groups.append(Group(event.name))
            except:
                seq = schema.get(sequence=obj)
                groups.extend([Group(ev.name) for ev in seq])

        for g in groups:
            if not isinstance(g, Pad):
                g.event = schema.get(event=g.name)
                g.name = g.event.name
                g.duration = g.event.duration
                g.df = self._get_df(g.name)
                if g.align is None:
                    g.align = align

                starts, stops = g.df["start"].values, g.df["stop"].values

                # If alignment policy is "fps", determine length given duration
                # specified by the corresponding event in the event model.                
                if is_int(g.align):
                    length = g.align
                elif g.align == "fps" and fps is not None and g.duration is not None:
                    length = round(g.duration * fps)
                elif isinstance(g.align, str):
                    raise NotImplementedError
                elif callable(g.align):
                    length = g.align(stops - starts)
                else:
                    length = np.median(stops - starts)
                length = round(length)
                stops = starts + length
                g.indexers = self._finalize_indexers(starts, stops, dimsize)

        return groups

    def _init_pads(
        self,
        groups: List["Group"],
        align: Optional[Union[int, str, "Quantity", Callable]],
        dimsize: int,
        fps: Optional[Number] = None,
    ) -> List["Group"]:

        # - Resolve references to target groups.
        # --------------------------------------
        targets = {g.name: g for g in groups if not isinstance(g, Pad)}
        for i, g in enumerate(groups):
            if not isinstance(g, Pad) or isinstance(g.ref, Group):
                continue

            # Target is an event name.
            try:
                g.ref = targets[g.ref]
                continue
            except KeyError:
                pass

            # No target specified; use context to determine it.
            if g.ref is None:
                if i == 0:
                    assert g.side != "back"
                    g.ref = groups[i + 1]
                elif i == len(groups) - 1:
                    assert g.side != "front"
                    g.ref = groups[i - 1]
                else:
                    g.ref = groups[i - 1] if g.side == "front" else groups[i + 1]
                continue

            raise ValueError("Pad ref must be `None`, a string, or a `Group`.")

        # - Determine sizes of slices
        # ---------------------------

        for g in groups:
            if not isinstance(g, Pad):
                continue

            ref_starts = np.array([ix.start if isinstance(ix, slice) else ix[0] \
                                   for ix in g.ref.indexers]
                                  )
            ref_stops = np.array([ix.stop if isinstance(ix, slice) else ix[-1] \
                                  for ix in g.ref.indexers]
                                 )

            if g.align is None:
                g.align = align
            if is_int(g.align):
                length = g.align
            elif g.align == "fps" and fps is not None and g.duration is not None:
                length = round(g.duration * fps)
            else:
                length = np.median(ref_stops - ref_starts)
            length = round(length)

            # Apply offsets.
            if g.side == "front":
                starts, stops = ref_starts - length, ref_starts
            else:
                starts, stops = ref_stops, ref_stops + length

            g.indexers = self._finalize_indexers(starts, stops, dimsize)

    def _finalize_indexers(
        self,
        starts: np.ndarray,
        stops: np.ndarray,
        dimsize: int,
    ) -> None:

        """
        Generates slices/integer-arrays from the dataframe containing unaligned
        start/stop indices.
        
        Since cropping/extending ragged data is a bit of a pain, the strategy
        taken here is to modify to indexers so that the chunks they extract will
        be guaranteed to have the correct shape.
                
        
        """

        indexers = []
        for i, (a, b) in enumerate(zip(starts, stops)):
            if a >= 0 and b <= dimsize:
                indexers.append(slice(a, b))
            else:
                indexers.append(np.clip(np.arange(a, b, dtype=int), 0, dimsize - 1))
        return indexers

    """
    extraction
    """

    def _extract(
        self,
        groups: List["Group"],
        data: ArrayLike,
    ) -> None:

        # Extract data
        if isinstance(data, (np.ndarray, xr.DataArray)):
            self._array_extract(groups, data)
        elif isinstance(data, h5py.Dataset):
            self._h5dataset_extract(groups, data)
        else:
            raise NotImplementedError

    def _array_extract(
        self,
        groups: List["Group"],
        data: ArrayLike,
    ) -> None:
        """
        Extract data from an ndarray.
        """
        # Extract data
        for g in groups:
            arrays = []
            for ix in g.indexers:
                arr = data[ix]
                arrays.append(arr)
            g.data = np.stack(arrays)

    def _h5dataset_extract(
        self,
        groups: List["Group"],
        data: ArrayLike,
    ) -> None:

        """
        Extract data from an h5 dataset. Can't use _array_extract since
        indexing with integer arrays is not supported for h5 datasets.
        """

        # Extract data
        for g in groups:
            arrays = []
            for ix in g.indexers:
                if isinstance(ix, slice):
                    arr = data[ix]
                else:
                    arr = np.stack([data[ival] for ival in ix])
                arrays.append(arr)
            g.data = np.stack(arrays)

    """
    utils
    """

    def _get_df(
        self,
        val: Union[str, "Group"],
    ) -> pd.DataFrame:

        """
        Get dataframe for a group/event.
        """
        name = val.name if isinstance(val, Group) else val
        try:
            return self._dfs[name]
        except:
            pass

        # Find schema object associated with argument to get numeric id.
        ev = self._schema.get(event=name)
        table = self._event_model.tables["events"]
        df = table[table.event_id == ev.id]
        self._dfs[name] = df
        return df

    def _get_fps(self):
        """
        Lazily compute framerate.
        """
        if self._fps is None:
            time = self._event_model.tables["frames"]["time"].values
            self._fps = 1 / np.median(np.ediff1d(time))
        return self._fps

    def _wrap_output(
        self,
        out: Union[xr.DataArray, List[xr.DataArray]],
        dims: Sequence[str],
    ) -> Union[xr.DataArray, List[xr.DataArray]]:
        """
        Wrap output data with one or more xr.DataArray objects having
        predefined dimensions.
        """
        out = out if isinstance(out, list) else [out]
        for i, arr in enumerate(out):
            out[i] = xr.DataArray(arr.data, dims=dims, name=arr.name)
        return out[0] if len(out) == 1 else out


class Group:
    """
    Intermediate object used to store data used throughout the grouping process.
    Can also be used for constructing a spec to supply to Grouper.group().
    
    A sequence of individually contiguous indexable (i.e., sliceable) regions.
    Instances of these classes are placeholders for arrays likely destined to
    be members of an array sequence.
            
    """

    name: str
    align: Optional[Union[int, str, Callable]]
    event: Optional["Event"]
    df: Optional[pd.DataFrame]
    indexers: Optional[List]
    data: ArrayLike

    def __init__(
        self,
        name: str,
        align: Optional[Union[int, str, Callable]] = None,
    ):
        self.name = name
        self.align = align
        self.event = None
        self.df = None
        self.indexers = None
        self.data = None


class Pad(Group):
    """
     `align` behaves differently than it does for regular, non-pad groups.
        None: match length of referent.
        int: number of frames.
        ureg.Quantity: fixed duration, independent of framerate.
    
    """

    #: Group this object is extending from.
    ref: Optional[Union[str, Group]]

    #: Padding side. Must be "front" or "back".
    side: str

    #: Duration in seconds.
    duration: Optional[Number]

    def __init__(
        self,
        side: str,
        ref: Optional[Union[str, Group]] = None,
        duration: Optional[Number] = None,
        align: Optional[int] = None,
    ):

        super().__init__("", align=align)

        self.ref = ref

        if side.lower() in {"front", "f"}:
            self.side = "front"
        elif side.lower() in {"back", "b"}:
            self.side = "back"
        else:
            raise ValueError(f"invalid pad side: {side}")

        self.duration = duration


class FPad(Pad):

    def __init__(
        self,
        ref: Optional[Union[str, Group]] = None,
        duration: Optional[Number] = None,
        align: Optional[Union[int, str, "Quantity", Callable]] = None,
    ):
        super().__init__("front", ref, duration, align=align)


class BPad(Pad):

    def __init__(
        self,
        ref: Optional[Union[str, Group]] = None,
        duration: Optional[Number] = None,
        align: Optional[Union[int, str, "Quantity", Callable]] = None,
    ):
        super().__init__("back", ref, duration, align=align)
