from pathlib import Path
import time
from typing import (
    Callable,
    List, Mapping,
    Optional,
    Sequence,
    Union,
)
import dask.array
from fnmatch import fnmatch
from fs.multifs import MultiFS
import h5py
import numpy as np
import xarray as xr

from .common import *
from .indexing import *
from .io import URL
from .event import EventSchema
from .environment import get_fs

__all__ = [
    "get_event_schema",
    "get_ufunc",
    "play_movie",
    "split_by_event",
    "timeit",
]


def get_event_schema(
    name: str,
    fs: Optional[Union[int, str]] = None,
) -> EventSchema:

    if not name.endswith(".yaml"):
        name += ".yaml"
    path = f"event_schemas/{name}"
    fs = get_fs(fs)
    if isinstance(fs, MultiFS):
        for entry in fs.iterate_fs():
            subfs = entry[1]
            if subfs.exists(path):
                return EventSchema(subfs.getsyspath(path))
        raise FileNotFoundError(name)
    else:
        return EventSchema(fs.getsyspath(path))



def get_ufunc(obj: Union[str, Callable]) -> Callable:
    name = obj.__name__ if callable(obj) else obj
    return getattr(np, name)


def play_movie(obj: Union[PathLike, ArrayLike], chunks: int = 10):

    import napari

    if isinstance(obj, (np.ndarray, xr.DataArray, h5py.Dataset)):
        _chunks = list(obj.shape)
        _chunks[0] = chunks
        arr = dask.array.from_array(obj, chunks=_chunks)
    else:
        url = URL(obj)
        path = Path(url.path)
        if h5py.is_hdf5(path):
            from ca_analysis.io.h5 import H5DatasetSource
            src = H5DatasetSource(url, chunks=chunks)
            arr = src.to_dask()
        elif fnmatch(path.name, "Image*.raw"):
            from ca_analysis.io.thorlabs import ThorImageArraySource
            src = ThorImageArraySource(url.path, chunks=chunks)
            arr = src.to_dask()
        else:
            raise ValueError(f'unsupported filetype: {url}')

    viewer = napari.view_image(arr)
    return viewer


def split_by_event(arr: xr.DataArray) -> List[xr.DataArray]:

    labels = arr.coords['event'].data.astype(int)
    inds = argwhere(labels[:-1] != labels[1:]) + 1
    inds = np.r_[0, inds, len(labels)]
    chunks = []
    for i in range(len(inds) - 1):
        start, stop = inds[i], inds[i + 1]
        chunks.append(arr.isel(time=slice(start, stop)))
    return chunks


def timeit(
    fn: Callable,
    args: Optional[Sequence] = None,
    kw: Optional[Mapping] = None,
    n_iter: int = 100,
) -> np.ndarray:
    """
    Get execution times for a function call.
    """

    args = args or []
    kw = kw or {}

    times = np.zeros(n_iter)
    for i in range(n_iter):
        t_start = time.perf_counter()
        fn(*args, **kw)
        times[i] = time.perf_counter() - t_start

    times = np.array(times)
    t_mean = np.mean(times)

    msg = f"timeit: mean execution time (n_iter={n_iter}) = "
    msg += format_timedelta(t_mean)
    print(msg)

    return times
