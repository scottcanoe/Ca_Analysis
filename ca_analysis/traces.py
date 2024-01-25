"""
Defines functions used for manipulating trace data.

"""

from typing import (
    Callable,
    Optional,
    Sequence,
    Union,
)

#import bottleneck as bn
import numpy as np
import xarray as xr

from .common import *

__all__ = [
    "argranked",
    # "butter_highpass_filter",
    "dFF",
    "deconvolve",
    "ranked",
]


def dFF(
    F: Union[xr.DataArray, np.ndarray],
) -> Union[xr.DataArray, np.ndarray]:
    """
    Performs dF/F process on one or more traces. This implementation uses
    a smoothed rolling min to establish a baseline activity within some
    reasonable window (the default window size is about 7-15 seconds wide when
    imaging at 15-30 Hz.). Smoothing is applied to the baseline prior to
    subtracting in order to compensate for the sudden jumps that often occur
    when computing rolling mins/maxs/medians on this data.

    Baseline is the pseudo-rolling minimum quantile. Baseline is subtracted, and
    then used as a scaling factor:

    dFF(x) = (x - baseline) / baseline


    If axis is None, then...

        ndim = 1 -> axis = 0   single trace
        ndim = 2 -> axis = 1   trace pool, and traces are in rows.


    Parameters
    ----------

    data : array-like
        1D, 2D, or 3D array.

    window_size : int
        Width of window used for computation.



    Returns
    -------

    result : ndarray
        Has the same shape and dtype as input.

    """

    is_xarray = isinstance(F, xr.DataArray)
    if is_xarray:
        data = F
    else:
        dims = ['time'] + [f'dim_{i}' for i in range(1, F.ndim)]
        data = xr.DataArray(F, dims=dims)

    baselines = data.min('time')
    heights = data.quantile(0.999, 'time')
    out = (data - baselines) / heights
    if is_xarray:
        out.attrs.update(F.attrs)
        return out
    return out.data


def deconvolve(arr: ArrayLike, **kw):

    from caiman.source_extraction.cnmf.deconvolution import constrained_foopsi

    if isinstance(arr, xr.DataArray):
        orig_dims = arr.dims
        if orig_dims[0] != "time":
            arr = arr.transpose("time", ...)
        data = arr.data
    else:
        data = np.asarray(arr)

    if data.ndim == 1:
        deconv = constrained_foopsi(data, **kw)
    else:
        deconv = np.vstack([constrained_foopsi(row, **kw) for row in data])

    if isinstance(arr, xr.DataArray):
        arr_out = arr.copy(data=deconv)
        if orig_dims[0] != "time":
            arr_out = arr_out.transpose(*orig_dims)
        return arr_out

    return deconv


def argranked(
    arr: xr.DataArray,
    method: Union[str, Callable],
    descending: bool = True,
) -> xr.DataArray:
    orig_dims = arr.dims
    orig_axis = orig_dims.index("time")
    if orig_axis != 0:
        arr = arr.transpose("time", "roi")

    scores = None
    if isinstance(method, str):
        try:
            fn = getattr(xr.DataArray, method)
            scores = fn(arr, "time")
        except AttributeError:
            pass

    elif callable(method):
        scores = method(arr)

    if scores is None:
        labels = arr.coords["label"]
        indic = -1 * np.ones(len(labels))
        event_names = [method] if isinstance(method, str) else method
        for name in event_names:
            indic[labels.data == name] = 1
        indic = indic[:, np.newaxis]
        scores = (arr * indic).sum("time")

    order = np.argsort(scores)
    if descending:
        order = np.flipud(order)
    return order


def ranked(
    arr: xr.DataArray,
    method: Union[str, Sequence[str], Callable],
    descending: bool = True,
) -> xr.DataArray:
    order = argranked(arr, method=method, descending=descending)
    out = arr.isel(roi=order)
    return out
