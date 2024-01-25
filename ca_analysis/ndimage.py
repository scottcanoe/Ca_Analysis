"""
Remember: numpy masked arrays is `True` to indicate which values are masked.
This is counterintuitive since a bitmask would have zeros where the data
is masked which evaluates to `False`.

"""
import copy
from typing import (
    Iterable,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from numpy.ma.core import MaskedArray
import scipy.ndimage
from skimage.morphology import binary_dilation, binary_erosion
import xarray as xr

from .common import *

__all__ = [
    "add_mask",
    "gaussian_filter",
    "luminance_filter",
    "median_filter",
    "percentile",
    "shift",
    "PixelMask",
]


def add_mask(
    arr: ArrayLike,
    mask: ArrayLike,
    **kw,
) -> np.ndarray:
    """
    Mask an array with a mask having possibly fewer dimensions than the
    underlying data. This is accomplished by broadcasting, and its main use
    case is to apply a 2d mask from an image to an entire stack of images.
    
    If the input data is already masked, the resulting mask will be the union
    of the original and new masks, and it will be modified in-place.
    """

    out = np.ma.asanyarray(arr)
    mask = np.array(mask, dtype=bool)
    if out.ndim != mask.ndim:
        mask = np.broadcast_to(mask, np.shape(out))
    out.mask = out.mask | mask

    # restore fill_value
    out.fill_value = kw.get("fill_value", getattr(arr, "fill_value", None))

    return out


def is_masked(arr: ArrayLike, nans: bool = False) -> bool:
    """
    This might be broken or not useful.

    Parameters
    ----------
    arr
    nans

    Returns
    -------

    """
    if isinstance(arr, MaskedArray):
        return True

    if nans:
        return np.isnan(arr).any()

    return False


def gaussian_filter(
    data: Union[xr.DataArray, np.ndarray],
    sigma: Union[Number, Sequence[Number]],
    *args,
    **kw,
) -> Union[xr.DataArray, np.ndarray]:
    """
    If input is a masked array, filtering is performed on unmasked data.
    This is done to prevent scipy from shrinking the unmasked region
    by the size of the kernel. The original mask will then be applied
    to the output array prior to returning.

    """
    is_dataarray = isinstance(data, xr.DataArray)
    if is_dataarray:
        array = data
        data = array.data

    kw["mode"] = kw.get("mode", "nearest")

    if isinstance(data, MaskedArray):
        out = scipy.ndimage.gaussian_filter(data.filled(), sigma, *args, **kw)
        out = np.ma.array(out, mask=data.mask, fill_value=data.fill_value)
        return out

    out = scipy.ndimage.gaussian_filter(data, sigma, *args, **kw)
    if is_dataarray:
        out_array = array.copy()
        out_array.data = out
        return out_array

    return out


def luminance_filter(
    mov: np.ndarray,
    posinf: Optional[Number] = 0.0,
    neginf: Optional[Number] = 0.0,
) -> np.ndarray:
    """
    "Correct" pixels by subtracting global activity. Converts raw pixel
    values to pixel-wise fractions of means minus global fraction of mean.


    Mask should be True

    pix_corrected = pix_t / mean(pix) - global_t / mean(global)

    mask: `True` indicates pixel is masked.

    """

    # Handled masked input array.
    if is_masked(mov):
        out = np.ma.zeros(mov.shape, dtype=np.float32)
        out.mask = mov.mask
        out.fill_value = mov.fill_value
    else:
        out = np.zeros_like(mov, dtype=np.float32)

    """
    T_f: mean luminance for each frame (1d)
    G: mean luminance for each frame divided by its mean (1d).
    S: pixel-wise mean luminance (2d).
    
    """
    T_f = np.array([im.mean() for im in mov])
    G = T_f / np.nanmean(T_f)
    S = np.nanmean(mov, axis=0)

    for i in range(mov.shape[0]):
        frame = mov[i] / S - G[i]
        if posinf is not None or neginf is not None:
            frame = np.nan_to_num(frame, posinf=posinf, neginf=neginf)
        out[i] = frame

    return out


def median_filter(
    data: np.ndarray,
    size: int,
    *args,
    fill_value: Optional[Number] = None,
    **kw,
) -> np.array:
    """
    If input is a masked array, filtering is performed on unmasked data.
    This is done to prevent scipy from shrinking the unmasked region
    by the size of the kernel. The original mask will then be applied
    to the output array prior to returning.

    """

    if not np.ma.is_masked(data) and not hasattr(data, 'mask'):
        return scipy.ndimage.median_filter(data, size, *args, **kw)

    # Run filter on unmasked data, then remask it.
    arr = data.filled(fill_value=fill_value)
    filtered = scipy.ndimage.median_filter(arr, size, *args, **kw)
    out = np.ma.array(filtered, mask=data.mask, fill_value=data.fill_value)
    return out


def percentile(
    data: ArrayLike,
    q: Union[Number, Sequence[Number]],
    maxsize: Optional[int] = 524_288_000,
) -> Union[Number, np.array]:
    """
    Percentile compatible with masked arrays.
    """

    # We need to check the size (and possibly the shape) of the data,
    # so make sure data has those attributes.
    if isinstance(data, xr.DataArray):
        data = data.data
    else:
        data = np.asanyarray(data)

    # Handle masked data.
    if is_masked(data):
        return percentile(data.compressed(), q, maxsize)

    # Extract a subset of the data if we're over the limit.
    size = data.size
    if maxsize is not None and size > maxsize:
        real_dims = [n for n in data.shape if n > 1]
        if len(real_dims) == 1:
            arr = data[:maxsize]
        else:
            chunk_size = np.prod(real_dims[1:])
            arr = data[:maxsize // chunk_size]
    else:
        arr = data

    return np.percentile(arr, q)


def shift(
    im: np.ndarray,
    shift: Union[Number, Sequence[Number]],
    *args,
    **kw,
) -> np.ndarray:
    """
    Shift an array. Extends scipy.ndimage.shift by allowing for
    treating the shift argument in a slice-like way. For example,
    if the input array is 5-dimensional, then shift arguments
    will be coerced such that ``(2, 3) -> (2, 3, 0, 0, 0)``,
    ``(2, 3, ..., 4) -> (2, 3, 0, 0, 4)``, and so on.
    """

    from xarray.core.indexing import expanded_indexer

    _shift = expanded_indexer(shift, np.ndim(im))
    _shift = [0 if isinstance(elt, slice) else elt for elt in _shift]

    return scipy.ndimage.shift(im, _shift, *args, **kw)



class PixelMask:
    """
    Interface for pixel masks as represented by arrays of y (rows) and
    x (columns) coordinates and optionally alpha values and colors.
    """
    def __init__(
        self,
        *args,
        dims: Optional[Iterable[str]] = ("y", "x"),
        values: Optional[Union[Number, ArrayLike]] = None,
    ):
        if len(args) == 1 and isinstance(args[0], PixelMask):
            template = args[0]
            self._arrays = tuple([arr.copy() for arr in template.arrays])
            self._dims = copy.copy(template.dims)
            self._values = copy.copy(template.values)

        elif len(args) == 2:
            dims = ("y", "x") if dims is None else tuple(dims)
            assert len(dims) == len(args)
            self._arrays = np.broadcast_arrays(args)
            self._dims = dims
            self._values = values
        else:
            raise ValueError("invalid args")

        self._length = len(self._arrays[0])


    @property
    def arrays(self) -> Tuple[np.ndarray]:
        return tuple(self._arrays)

    @property
    def values(self) -> Optional[Union[Number, ArrayLike]]:
        return self._values

    @values.setter
    def values(self, vals: Optional[Union[Number, ArrayLike]]) -> None:
        if vals is None:
            self._values = None
            return
        if np.isscalar(vals):
            self._values = vals
            return
        val = np.asarray(vals)
        self._values = np.broadcast_to(val, self._arrays[0].shape)

    def get_boundary(self, outer: bool = True, pad: int = 4) -> "PixelMask":

        lims = [(int(a.min()), int(a.max())) for a in self._arrays]
        shape = [p[1] - p[0] + pad for p in lims]
        im = np.zeros(shape, dtype=bool)
        coords = (arr - lims[i][1] for i, arr in enumerate(self._arrays))
        im[coords] = True

        if outer:
            modified = binary_dilation(im)
        else:
            modified = binary_erosion(im)
        coords = np.where(modified & ~im)
        args = [arr - pad + lims[i][0] for i, arr in enumerate(coords)]
        out = PixelMask(*args)
        return out

    def __getattr__(self, key: str) -> np.ndarray:
        return self[key]

    def __getitem__(self, key: Union[int, str]) -> np.ndarray:
        if isinstance(key, str):
            try:
                key = self._dims.index(key)
            except IndexError:
                raise KeyError(key)
        return self._arrays[key]

    def __len__(self) -> int:
        return self._length