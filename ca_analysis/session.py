import datetime
import subprocess as sp
import warnings
from collections import UserList
from typing import (
    Any, Iterable, List, Mapping, Optional,
    Sequence, Tuple, Type,
    Union,
)

import dask.array as da
import fs.errors
import h5py
from fs.base import FS
import numpy as np
import xarray as xr

from .common import *
from .environment import *
from .event import *
from .io.resources import ResourceCatalog
from .persistence import PersistentMapping
from .roi import *

__all__ = [
    "FluorescenceData",
    "LFPData",
    "MovieData",
    "Session",
    "SessionData",
    "SessionGroup",
]


class Session:
    """
    Interface to imaging session data. Lots of lazy attributes.

    
    ``Session(path)``
    ``Session(mouse, date, run [,fs=...])``


    """
    fs: Optional["FS"]
    mouse: Optional[str]
    date: Optional[datetime.date]
    run: Optional[str]
    sources: Mapping[str, Any]

    _attrs: PersistentMapping
    _events: "EventModel"
    _registry: "ResourceRegistry"
    _segmentation: "Segmentation"

    events_class: Type[EventModel] = EventModel
    registry_class: Type[ResourceCatalog] = ResourceCatalog
    segmentation_class: Optional[type] = None

    _rois: Optional[ROIGroup] = None

    def __init__(
        self,
        fs: Optional[FS] = None,
        mouse: Optional[str] = None,
        date: Optional[Union[str, datetime.date]] = None,
        run: Optional[Union[str, int]] = None,
    ):

        self.fs = fs
        self.mouse = mouse
        self.date = as_date(date) if date is not None else None
        self.run = str(run) if run is not None else None

        if not isinstance(self.fs, FS) and \
                all(val is not None for val in [self.mouse, self.date, self.run]):
            self.fs = get_fs(fs).opendir(self.stem)

        self.sources = {}
        self._attrs = None
        self._events = None
        self._registry = None
        self._segmentation = None
        self.mov_file = None

    @property
    def stem(self) -> str:
        return f'sessions/{self.mouse}/{self.date}/{self.run}'

    @property
    def exp(self) -> Optional[str]:
        warnings.warn("deprecated 'exp'. use 'run' instead")
        return self.run

    @property
    def exp(self, val: Optional[str]) -> None:
        warnings.warn("deprecated 'exp'. use 'run' instead")
        self.run = val

    @property
    def name(self) -> str:
        if self.mouse is None or self.date is None or self.run is None:
            raise ValueError('no mouse/date/run string available')
        return f"{self.mouse}/{self.date}/{self.run}"

    @property
    def attrs(self) -> PersistentMapping:
        if self._attrs is None:
            try:
                path = self.fs.getsyspath("attrs.yaml")
            except fs.errors.ResourceNotFound:
                path = self.fs.getsyspath("") + "/attrs.yaml"
            self._attrs = PersistentMapping(path, load=True)
        return self._attrs

    @property
    def registry(self) -> ResourceCatalog:
        if self._registry is None:
            self._registry = self.registry_class(self.fs)
        return self._registry

    reg = registry
    cat = registry

    @property
    def segmentation(self) -> "Segmentation":
        if self._segmentation is None:
            path = self.registry["segmentation"].resolve()
            self._segmentation = self.segmentation_class(path, session=self)
        return self._segmentation

    seg = segmentation

    # @property
    # def rois(self) -> ROIGroup:
    #
    #     if self._rois is None:
    #         seg = self.segmentation
    #         iscell = seg["iscell"]
    #         self._rois = ROIGroup([ROI(seg, ix) for ix in range(len(iscell))])
    #     return self._rois
    #
    # @property
    # def cells(self):
    #     return ROIGroup([r for r in self.rois if r["iscell"]])

    @property
    def events(self) -> EventModel:
        if self._events is None:
            path = self.fs.getsyspath('events')
            self._events = self.events_class(path)
        return self._events

    def close(self) -> None:
        for key, src in self.sources.items():
            src.close()
        self.sources.clear()

    def get_fs(
        self,
        name: Optional[Union[int, str]],
        require: bool = False,
    ) -> "FS":

        stem = f"sessions/{self.mouse}/{self.date}/{self.run}"
        branch = get_fs(name)
        if not branch.exists(stem):
            if require:
                branch.makedirs(stem)
            else:
                raise fs.errors.ResourceNotFound(stem)
        ses_fs = branch.opendir(stem)
        return ses_fs

    def switch_fs(
        self,
        name: Optional[Union[int, str]],
        require: bool = False,
    ) -> None:

        stem = f"sessions/{self.mouse}/{self.date}/{self.run}"
        branch = get_fs(name)
        if not branch.exists(stem):
            if require:
                branch.makedirs(stem)
            else:
                raise fs.errors.ResourceNotFound(stem)
        ses_fs = branch.opendir(stem)
        self.fs = ses_fs


    def play_movie(self):
        import dask.array as da
        import napari

        mov_path = self.fs.getsyspath("mov.h5")
        self.mov_file = h5py.File(mov_path, "r")
        dset = self.mov_file['data']

        arr = da.from_array(dset, chunks=(1, -1, -1))
        if 'mask' in self.mov_file.keys():
            mask = self.mov_file['mask'][:]
            y, x = np.where(mask)

            def mask_block(array):
                out = np.asarray(array)
                if out.ndim == 2:
                    out[y, x] = np.nan
                else:
                    out[:, y, x] = np.nan
                return out

            arr = arr.map_blocks(mask_block, dtype='float32')

        self.mov_arr = arr
        self.mov_viewer = napari.view_image(self.mov_arr)

        return self.mov_viewer

    def pull(
        self,
        src: Union[int, str] = -1,
        dst: Union[int, str] = 0,
        **kw,
    ) -> None:

        self.push(src, dst, **kw)


    def push(
        self,
        src: Union[int, str] = 0,
        dst: Union[int, str] = -1,
        **kw,
    ) -> None:

        import fs as pyfs

        stem = f'sessions/{self.mouse}/{self.date}/{self.run}'
        src_fs = get_fs(src)
        dst_fs = get_fs(dst)
        walker = pyfs.walk.Walker(**kw) if kw else None
        pyfs.copy.copy_dir_if(
            src_fs,
            stem,
            dst_fs,
            stem,
            "newer",
            walker=walker,
            preserve_time=True,
        )
    # -------------------------------------------------------------------------#

    def __repr__(self) -> str:
        return f"<Session: mouse={self.mouse}, date={self.date}, run={self.run}>"


class SessionData:
    """
    Accessor class assigned to each session instance during`open_session`.
    Specialized to handle slicing and grouping of transitions.
    """

    _data: Optional[ArrayLike] = None
    _prepared: bool = False

    def __init__(
        self,
        session: Session,
        data: Optional[ArrayLike] = None,
    ):
        self._session = session
        self._data = data

    @property
    def session(self) -> Session:
        return self._session

    @property
    def data(self) -> Any:
        self.check_prepared()
        return self._data

    @data.setter
    def data(self, obj) -> None:
        self._data = obj
        self._prepared = True

    def clear(self) -> None:
        self._data = None
        self._prepared = False

    def close(self) -> None:
        pass

    def load(self) -> None:
        raise NotImplementedError

    def find_sequences(
        self,
        obj,
    ) -> xr.DataArray:
        """
        Find where sequences occur.

        Parameters
        ----------
        obj

        Returns
        -------
        A DataArray of slices with dimensions (trial, event). The 'trial'
        dimension has coordinate array 'event' with one event object per event.

        """
        from ca_analysis.indexing import find_sequences

        # setup/initialize
        ev_df = self._session.events['events']
        samplerate = self._session.attrs['samplerate']
        events = self._parse_events_arg(obj)
        ev_ids = [int(ev) for ev in events]
        ev_lengths = [round(ev.duration * samplerate) for ev in events]

        # find indices of matching sequences
        matches = find_sequences(ev_ids, ev_df.event.values)
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
        coords = {"event": xr.DataArray(events, dims=("event",))}
        return xr.DataArray(slices, dims=("trial", "event"), coords=coords)

    def split(
        self,
        obj,
        lpad: Optional[Union[int, units.Quantity]] = None,
        rpad: Optional[Union[int, units.Quantity]] = None,
        concat: Union[bool, str] = False,
    ) -> Union[xr.DataArray, List[xr.DataArray]]:
        """
        Given a sequence of elements, return a list of arrays (one for each
        item) having dimensions ('trial', 'time', ...). If the data to be
        split is not an xarray.DatArray, then it's zeroth dimension will
        be assumed to be 'time'. Otherwise, it will be transposed accordingly.

        Parameters
        ----------
        events

        Returns
        -------

        """
        # init
        schema = self._session.events.schema
        samplerate = self._session.attrs["samplerate"]

        # Get matrix of slices, where each column contains slices for
        # all trials for a particular event.
        slices = self.find_sequences(obj)
        events = slices.coords['event']
        n_trials = slices.sizes["trial"]
        dims = ("trial",) + self.data.dims

        # prepare target data
        data = self.data
        if isinstance(self.data, xr.DataArray):
            if self.data.dims[0] != "time":
                self.data = self.data.transpose("time", ...)

        # Use the slices to extract the data from a target array for each event.
        t_start = 0
        splits = []
        for i, ev in enumerate(events):

            # find shape/dims of chunk to create
            ev_slices = slices.isel(event=i)
            if n_trials:
                n_timepoints = int(ev_slices[0].item().stop - ev_slices[0].item().start)
            else:
                n_timepoints = 0
            shape = (n_trials, n_timepoints) + self.data.shape[1:]

            # extract data
            mat = np.zeros(shape, dtype=self.data.dtype)
            for j, slc in enumerate(ev_slices):
                mat[j] = self.data[slc.item()]

            # create coordinates
            ev_coord = xr.DataArray(np.full(n_timepoints, ev), dims=('time',))
            t_coord = t_start + np.arange(n_timepoints) / samplerate
            t_start = t_coord[-1] + 1 / samplerate
            coords = dict(event=ev_coord, time=t_coord)

            # append array
            splits.append(xr.DataArray(mat, dims=dims, coords=coords))

        if lpad is not None:

            # extract data
            n_timepoints = round(lpad.to("sec").m * samplerate) if \
                isinstance(lpad, units.Quantity) else lpad
            shape = (n_trials, n_timepoints) + self.data.shape[1:]
            mat = np.zeros(shape, dtype=self.data.dtype)
            target_slices = slices.isel(event=0)
            for j, slc in enumerate(target_slices):
                slc = slc.item()
                start, stop = slc.start - n_timepoints, slc.start
                chunk = self.data[slice(start, stop)]
                if chunk.shape[0] == mat.shape[0]:
                    mat[j] = chunk
                else:
                    warnings.warn('padding front of array')
                    new_chunk = np.broadcast_to(chunk[0], mat.shape[1:])
                    new_chunk = np.array(new_chunk)
                    new_chunk[-chunk.shape[0]:] = chunk
                    mat[j] = new_chunk

            # create coordinates
            ev = schema.get(event=0)
            ev_coord = xr.DataArray(np.full(n_timepoints, ev), dims=('time',))
            t_coord = 1 / samplerate + np.arange(n_timepoints) / samplerate
            t_coord = np.flipud(-t_coord)
            coords = dict(event=ev_coord, time=t_coord)

            # prepend array
            splits.insert(0, xr.DataArray(mat, dims=dims, coords=coords))

        if rpad is not None:

            # extract data
            n_timepoints = round(rpad.to("sec").m * samplerate) if \
                isinstance(rpad, units.Quantity) else rpad
            shape = (n_trials, n_timepoints) + self.data.shape[1:]
            mat = np.zeros(shape, dtype=self.data.dtype)
            target_slices = slices.isel(event=-1)
            for j, slc in enumerate(target_slices):
                slc = slc.item()
                start, stop = slc.stop, slc.stop + n_timepoints
                chunk = self.data[slice(start, stop)]
                if chunk.shape[0] == mat.shape[0]:
                    mat[j] = chunk
                else:
                    warnings.warn('padding end of array')
                    new_chunk = np.broadcast_to(chunk[-1], mat.shape[1:])
                    new_chunk = np.array(new_chunk)
                    new_chunk[0:chunk.shape[0]] = chunk
                    mat[j] = new_chunk


            # create coordinates
            ev = schema.get(event=0)
            ev_coord = xr.DataArray(np.full(n_timepoints, ev), dims=('time',))
            t_coord = t_start + np.arange(n_timepoints) / samplerate
            coords = dict(event=ev_coord, time=t_coord)

            # append array
            splits.append(xr.DataArray(mat, dims=dims, coords=coords))

        if len(splits) == 1:
            return splits[0]

        if concat:
            if concat == True:
                return xr.concat(splits, 'time')
            return xr.concat(splits, concat)
        return splits


    def _parse_events_arg(self, obj) -> List[Event]:

        schema = self._session.events.schema

        if isinstance(obj, Event):
            events = [obj]
        elif isinstance(obj, EventSequence):
            events = list(obj)
        else:
            if is_int(obj):
                events = [schema.get(event=obj)]
            elif is_str(obj):
                if schema.tables['sequences'] is not None and \
                        obj in schema.tables['sequences']['name'].values:
                    events = list(schema.get(sequence=obj))
                else:
                    events = [schema.get(event=obj)]
            else:
                events = [schema.get(event=elt) for elt in obj]

        return events

    def check_prepared(self, force: bool = False) -> None:
        if self._prepared and not force:
            return
        self.prepare()

    def prepare(self) -> None:
        self._prepare()
        self._prepared = True

    def _prepare(self) -> None:
        """
        Reimplement this to open/load data.

        Returns
        -------

        """
        pass


class SessionGroup(UserList):

    def __init__(
        self,
        sessions: Optional[Iterable[Session]] = None,
        name: Optional[str] = None,
    ):
        super().__init__(sessions)
        self.name = name

    @property
    def sessions(self) -> List[Session]:
        return self.data

    @sessions.setter
    def sessions(self, obj: Iterable[Session]) -> None:
        self.data = obj

    def split(
        self,
        target: str,
        arg,
        lpad: Optional[Union[int, units.Quantity]] = None,
        rpad: Optional[Union[int, units.Quantity]] = None,
        concat: Union[bool, str] = False,
    ) -> Union[xr.DataArray, List[xr.DataArray]]:

        splits = []
        for s in self.sessions:
            data = getattr(s, target)
            splits.append(data.split(arg, lpad=lpad, rpad=rpad, concat='time'))

        if concat:
            return xr.concat(splits, concat)
        return splits



class LFPData(SessionData):
    """
    Accessor class assigned to each session instance during`open_session`.
    Specialized to handle slicing and grouping of transitions.
    """

    def _prepare(self) -> None:
        h5_path = self._session.fs.getsyspath("data.h5")
        with h5py.File(h5_path, "r") as f:
            adData = f["adData"][:]
            adTimestamps = f["adTimestamps"][:].squeeze()
            adChannels = f["adChannels"][:].squeeze().astype(int)
            self._data = xr.DataArray(
                adData,
                dims=("time", "channel"),
                coords={"time": adTimestamps, 'channel': adChannels}
            )


class MovieData(SessionData):

    file: Optional[h5py.File] = None

    def __init__(
        self,
        session: Session,
        path: PathLike = "mov.h5",
        key: str = "data",
    ):
        super().__init__(session)
        self.path = path
        self.key = key

    def clear(self) -> None:
        self._data = None
        self._prepared = False
        self.close()

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None

    def load(self) -> None:
        self.close()
        with h5py.File(self._session.fs.getsyspath(self.path), 'r') as f:
            arr = f[self.key][:]
        self.data = xr.DataArray(arr, dims=("time", "y", "x"))

    def _prepare(self):
        self.file = h5py.File(self._session.fs.getsyspath(self.path), 'r')
        dset = self.file[self.key]
        arr = da.from_array(dset, chunks=(10, -1, -1))
        self._data = xr.DataArray(arr, dims=("time", "y", "x"))


class FluorescenceData(SessionData):

    def __init__(self, session: Session, key: str, cells: bool = True):
        super().__init__(session)
        self._key = key
        self._cells = cells

    def filter_by(self):
        pass

    def _prepare(self):
        self._data = self._session.segmentation[self._key]
        if self._cells:
            try:
                iscell = self._session.segmentation['iscell']
            except KeyError:
                iscell = None
            if iscell is not None:
                self._data = self._data.isel(roi=iscell)
