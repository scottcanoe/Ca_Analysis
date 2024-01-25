from numbers import Number
from typing import (
    Optional,
    Tuple,
    Union,
)

import numpy as np
from .common import *
from .environment import *
import xarray as xr

__all__ = [
    "resample_labels",
    "resample_time",
    "resample_mov",
    "resample1d",
]



def resample_labels(
    labels: ArrayLike,
    num: Number,
    factor: bool = False,
    t: Optional[ArrayLike] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    
    """
    Resample categorical/discrete values (i.e., labels) using nearest neighbor method. 
    Supply `num` for number of output points or, if `factor=True`, the resampling
    factor. (e.g. factor=2 doubles sampling rate).
    
    Like scipy.signal.resample, optionally providing `t` (associated timepoints)
    will return a tuple (new_time, new_labels). Otherwise, only new_labels
    are returned.
    
    Parameters
    ----------
    labels: array-like
    num: int, optional
    factor: number, optional
    t: array-like, optional.
    
    Returns
    -------
    np.ndarray or (np.ndarray, np.ndarray)
    
    """
    from scipy.interpolate import interp1d
    
    num_in = len(labels)
    if factor:
        factor, num_out = num, round(num_in * num)
    else:
        factor, num_out = num / num_in, num
    del num
    
    # Map categorical values to integer IDs.
    unique_labels = set(labels)
    label_to_id = {elt: i for i, elt in enumerate(unique_labels)}
    id_to_label = {val: key for key, val in label_to_id.items()}
    ids = [label_to_id[elt] for elt in labels]
    
    # Generate new coordinates.
    if t is None:
        coords_in = np.arange(num_in)
        coords_out = np.arange(0, num_in, 1 / factor)
    else:
        coords_in = np.asanyarray(t)
        coords_out = resample_time(t, num_out)
        
    # Interpolate.
    fn = interp1d(
        coords_in,
        ids,
        kind="nearest",
        fill_value="extrapolate",
        assume_sorted = True,
    )
    new_ids = fn(coords_out).astype(int)
    
    # Map back.
    new_labels = np.array([id_to_label[elt] for elt in new_ids])
    
    if t is None:
        return new_labels
    return coords_out, new_labels


def resample_time(
    t: ArrayLike,
    num: Number,
    factor: bool = False,
    ) -> np.ndarray:
    
    """
    Resample a time coordinate array.

    """
    from scipy.interpolate import interp1d
    
    t = np.asanyarray(t)
    num_in = len(t)    
    if factor:
        factor, num_out = num, round(num_in * num)
    else:
        factor, num_out = num / num_in, num
    del num
    
    coords_in = np.arange(0, num_in)
    coords_out = np.arange(0, num_in, 1 / factor)
    assert len(coords_out) == num_out
    
    fn = interp1d(
        coords_in,
        t,
        kind="linear",
        fill_value="extrapolate",
        assume_sorted=True,
    )
    new_t = fn(coords_out)
    return new_t
    



def resample_mov(
    mov: ArrayLike,
    num: Number,
    factor: bool = False,
    t: Optional[ArrayLike] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    
    """
    Resample movie data along time dimension.
    Supply `num` for number of output points or `factor` (e.g. factor=2 doubles 
    sampling rate).
    
    Like scipy.signal.resample, optionally providing `t` (associated timepoints)
    will return a tuple (new_time, new_labels). Otherwise, only new_labels
    are returned.
    
    Parameters
    ----------
    labels: array-like
    num: int, optional
    factor: number, optional
    t: array-like, optional.
    
    Returns
    -------
    np.ndarray or (np.ndarray, np.ndarray)
    
    """

    import scipy.signal
    from .ndimage import add_mask
    
    num_in = len(mov)
    if factor:
        factor, num_out = num, round(num_in * num)
    else:
        factor, num_out = num / num_in, num
    del num
    
    if np.ma.is_masked(mov):
        out = scipy.signal.resample(mov.data, num_out, axis=0)
        out = add_mask(out, mov.mask[0], fill_value=mov.fill_value)
    else:
        out = scipy.signal.resample(mov, num_out, axis=0)
    
    if t is None:
        return mov
    t = resample_time(t, num=num_out)
    return t, mov


def resample1d(
    data: xr.DataArray,
    dim: str,
    num: Optional[int] = None,
    factor: Optional[bool] = None,
    method: str = "linear",
    fill_value: Union[Number, str] = "extrapolate",
    preserve_ints: bool = True,
) -> xr.DataArray:

    """
    Resample a DataArray. Extends xarray's resampling methods, including
    preserving integer-typed values.

    Parameters
    ----------
    data
    dim
    num
    factor
    method
    fill_value
    preserve_ints

    Returns
    -------

    """
    from scipy.interpolate import interp1d

    if num is None:
        if factor is None:
            raise ValueError("must provide either 'num' or 'factor'")
        num = round(factor * data.sizes[dim])

    if dim in data.coords:
        keep_coord = True
        x, y = np.arange(len(data.coords[dim])), data.coords[dim].values
    else:
        keep_coord = False
        x = y = np.arange(data.sizes[dim])
        data[dim] = xr.DataArray(x, dims=(dim,))

    f = interp1d(x, y, kind="linear", fill_value="extrapolate")
    pts = f(np.linspace(x[0], x[-1], num))
    kwargs = {dim: pts, "method": method, "kwargs": {"fill_value": fill_value}}
    out = data.interp(**kwargs)
    if not keep_coord:
        del out.coords[dim]

    if preserve_ints:
        dtype = data.dtype
        if np.issubdtype(dtype, np.integer):
            int_out = out.astype(dtype)
            int_out[:] = np.round(out.data).astype(dtype)
            out = int_out

        for name, coord in out.coords.items():
            dtype = data.coords[name].dtype
            if np.issubdtype(dtype, np.integer):
                if name in coord.coords:
                    del coord.coords[name]
                int_coord = coord.astype(dtype)
                int_coord[:] = np.round(coord.data).astype(dtype)
                out.coords[name] = int_coord

    return out
