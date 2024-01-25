from typing import (
    Any,
    Iterator,
    List, Literal,
    Mapping, Optional,
    Sequence, Tuple,
    Union,
)

import ndindex as nd
import numpy as np
from ndindex.ndindex import NDIndex
import pandas as pd
from xarray.core.indexing import expanded_indexer

from .common import *

__all__ = [
    "argwhere",
    "IndexLike",
    "find_repeats",
    "find_sequences",
    "get_chunk_edges",
    "iterchunks",
    "NDIndexLike",
]

IndexLike = Optional[
    Union[
        int,  # basic
        slice,  # basic
        ArrayLike,  # 1d bool array (fancy)
        Literal[Ellipsis],
        NDIndex,
    ]
]
NDIndexLike = Union[IndexLike, Tuple[IndexLike]]


def argwhere(arr: ArrayLike) -> ArrayLike:
    """
    Wraps numpy.argwhere so that 1d input arrays will result in 1d output.

    Parameters
    ----------
    arr

    Returns
    -------

    """

    arr = np.asarray(arr)
    ndim = arr.ndim

    inds = np.argwhere(arr)
    if ndim == 1:
        if inds.size == 0:
            return np.array([], dtype=np.intp)
        return np.atleast_1d(inds.squeeze())

    return inds



def find_repeats(
    obj: ArrayLike,
    val: Optional[Any] = MISSING,
) -> pd.DataFrame:
    """
    Find runs of consecutive items in an array, returning a dataframe
    showing where contiguous sections are (in order).
    Author: Alistair Miles

    Returns
    -------
    Dataframe with columns: 'start', 'stop', and 'value'

    URL: https://gist.github.com/alimanfoo/c5977e87111abe8127453b21204c1065
    """

    # ensure array
    arr = np.asarray(obj)
    if arr.ndim != 1:
        raise ValueError('only 1D arrays supported')
    dimlen = arr.shape[0]

    # handle empty array
    if dimlen == 0:
        empty = np.array([], dtype=int)
        return pd.DataFrame(
            dict(start=empty, stop=empty, value=np.array([], dtype=arr.dtype)),
        )

    # find run starts
    loc_start = np.empty(dimlen, dtype=bool)
    loc_start[0] = True
    np.not_equal(arr[:-1], arr[1:], out=loc_start[1:])
    starts = np.nonzero(loc_start)[0]

    # find run values and run lengths
    values = arr[loc_start]
    lengths = np.diff(np.append(starts, dimlen))
    stops = starts + lengths

    # optionally filter out non-requested values
    if val is not MISSING:
        inds = argwhere(values == val)
        values = values[inds]
        starts = starts[inds]
        stops = stops[inds]

    df = pd.DataFrame({
        'start': starts,
        'stop': stops,
        'value': values,
    })

    return df


def find_sequences(
    pattern: ArrayLike,
    arr: ArrayLike,
) -> pd.DataFrame:
    """
    Find where particular sequences occur.
    Parameters
    ----------
    pattern
    arr

    Returns
    -------
    Dataframe with columns: 'start', 'stop', and 'value'

    """
    pattern = np.asarray(pattern)
    if pattern.ndim != 1:
        raise ValueError('only 1D arrays supported')

    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError('only 1D arrays supported')

    pat_len = len(pattern)
    possible_starts = argwhere(arr == pattern[0])
    starts, stops = [], []
    for a in possible_starts:
        b = a + pat_len
        if np.array_equal(pattern, arr[a:b]):
            starts.append(a)
            stops.append(b)

    df = pd.DataFrame({'start': starts, 'stop': stops})
    return df


def get_chunk_edges(
    shape: Union[int, Sequence[int]],
    size: Optional[Union[int, Tuple[int]]] = None,
) -> Tuple:

    def get_edges(
        dimlen: int,
        chk: Optional[int] = None,
    ) -> NDArray:

        if chk is None or chk == -1:
            return np.array([0, dimlen], dtype=int)
        if is_int(chk) and chk > 0:
            return np.r_[np.arange(0, dimlen, chk), [dimlen]]
        raise ValueError(f'invalid chunks argument: {chk}')

    if is_int(shape) or len(shape) == 1:
        return get_edges(shape, size)
    size = np.broadcast_to(size, len(shape,))
    return tuple(get_edges(shape[i], size[i]) for i, dim in enumerate(shape))


def iterchunks(
    arr: ArrayLike,
    size: int,
    axis: int = 0,
) -> Iterator[Union[int, Union[nd.Slice, nd.Tuple], ArrayLike]]:
    """
    Yields tuples of (chunk num, chunk slice, chunk). Useful for moving
    chunks of data from one array to another.

    Parameters
    ----------
    arr
    size
    axis

    Returns
    -------

    """
    is_1d = arr.ndim == 1
    edges = get_chunk_edges(arr.shape[axis], size)
    index_list = list(nd.ndindex(slice(None)).expand(arr.shape).args)
    for i in range(len(edges) - 1):
        slc = slice(edges[i], edges[i + 1], 1)
        if is_1d:
            idx = nd.Slice(slc)
        else:
            index_list[axis] = slc
            idx = nd.Tuple(*tuple(index_list))
        chunk = arr[idx.raw]
        yield i, idx, chunk

