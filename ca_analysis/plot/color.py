import copy
from numbers import Number
from typing import Optional, Sequence, Tuple, Union

import dask.array as da
import matplotlib.colors as mpc
import matplotlib.cm
import numpy as np

from ..common import ArrayLike
from ..ndimage import percentile

__all__ = [
    "ColorLike",
    "DIVERGING_COLORMAPS",
    "get_cmap",
    "get_norm",
    "get_smap",
    "ScalarMappable",
    "to_hex",
    "to_rgb",
    "to_rgba",
]

DIVERGING_COLORMAPS = (
    "brbg",
    "bwr",
    "coolwarm",
    "piyg",
    "prgn",
    "puor",
    "rdbu",
    "rdgy",
    "rdylbu",
    "rdylgn",
    "seismic",
    "spectral",
)

ColorLike = Union[str, ArrayLike]


def to_hex(
    obj: ColorLike,
    keep_alpha: bool = False,
) -> str:
    c = mpc.to_hex(obj, keep_alpha=keep_alpha)
    return c


def to_rgb(
    obj: ColorLike,
    bytes: bool = False,
) -> np.ndarray:
    return to_rgba(obj, bytes=bytes)[0:3]


def to_rgba(
    obj: ColorLike,
    alpha: Optional[Number] = None,
    bytes: bool = False,
) -> np.ndarray:
    c = np.array(mpc.to_rgba(obj, alpha=alpha))
    if bytes:
        c = (255 * c).astype(int)
    return c



class ScalarMappable(matplotlib.cm.ScalarMappable):
    """
    Extends matplotlib's ScalarMappable class by adding optional 
    casting to a particular datatype. Note that casting to an integer type 
    automatically returns values in 0-255 rather than floating point values 
    in 0.0-1.0.
    """

    _dtype: np.dtype
    _bytes: bool

    def __init__(
        self,
        norm: Optional[mpc.Normalize] = None,
        cmap: Optional[Union[str, mpc.Colormap]] = None,
        dtype: Union[str, type, np.dtype] = float,
    ):
        super().__init__(norm=norm, cmap=cmap)

        self.dtype = dtype

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @dtype.setter
    def dtype(self, type_: Union[str, type, np.dtype]) -> None:
        self._dtype = np.dtype(type_)
        self._bytes = np.issubdtype(self._dtype, np.integer)

    @property
    def diverging(self) -> bool:
        return self.cmap.name in DIVERGING_COLORMAPS

    def __call__(
        self,
        x: Union[Number, Sequence[Number]],
        alpha: Optional[Number] = None,
        bytes: Optional[bool] = None,
        norm: bool = True,
    ) -> np.ndarray:
        bytes = self._bytes if bytes is None else bytes
        out = self.to_rgba(x, alpha=alpha, bytes=bytes, norm=norm)
        if out.dtype != self._dtype:
            out = out.astype(self._dtype)
        return out

    def __repr__(self) -> str:
        s = "ScalarMappable\n"
        s += f' - cmap: {self.cmap.name}\n'
        s += f' - vlim: [{self.norm.vmin}, {self.norm.vmax}]'
        return s



def get_cmap(
    name: Optional[Union[str, mpc.Colormap]] = None,
    **kw,
) -> mpc.Colormap:
    """
    Returns a colormap, optionally modifying extreme values. Always returns
    a copy of another colormap.

    Available keyword args:
      * bad: color to substitute for NaNs.
      * bad_alpha: alpha value for bad color. Only valid if `bad` specified.
      * over: color to substutute for values over a threshold.
      * over_alpha: alpha value for over color. Only valid if `over` specified.
      * under: color to substutute for values below a threshold.
      * under_alpha: alpha value for under color. Only valid if `under` specified.
    
    Any other keyword args are ignored.

    Much faster (>10x) than pyqtgraph, and it works with masked arrays
    and under/over values.
    """

    cmap = copy.copy(matplotlib.cm.get_cmap(name, lut=kw.get("lut")))

    if "bad" in kw:
        cmap.set_bad(kw["bad"], alpha=kw.get("bad_alpha", None))
    if "over" in kw:
        cmap.set_bad(kw["over"], alpha=kw.get("over_alpha", None))
    if "under" in kw:
        cmap.set_bad(kw["under"], alpha=kw.get("under_alpha", None))

    return cmap


def get_norm(
    vlim: Optional[Tuple[Number, Number]] = None,
    qlim: Optional[ArrayLike] = None,
    data: Optional[ArrayLike] = None,
    clip: bool = False,
    centered: bool = False,
    **kw,
) -> mpc.Normalize:
    """
    Create a Normalize instance from either `vlim` or percentiles (`qlim`) along
    with data.

    If `centered`, qlim/vlim will be interpreted as (vcenter, halfrange)

    If `vlim` is known in advance, use the `vlim` keyword arg to bypass
    percentile computation on data.

    `kw` is ignored.

    """

    if vlim is None and qlim is not None:
        if data is None:
            raise ValueError("must supply data when using 'qlim' argument")
    if centered:
        if qlim is not None:
            vlim = np.nanpercentile(data, qlim)
        return mpc.CenteredNorm(vcenter=vlim[0], halfrange=vlim[1], clip=clip)
    else:
        if qlim is not None:
            vlim = np.nanpercentile(data, qlim)
        return mpc.Normalize(vmin=vlim[0], vmax=vlim[1], clip=clip)


def get_smap(
    cmap: Optional[Union[str, mpc.Colormap]] = None,
    norm: Optional[mpc.Normalize] = None,
    dtype: Optional[Union[str, type, np.dtype]] = float,
    **kw,
) -> matplotlib.cm.ScalarMappable:
    """
    Make a ScalarMappable for a dataset. By default, returns a subclass
    of ScalarMappable that is callable.


    """
    kw["centered"] = kw.get("centered", cmap in DIVERGING_COLORMAPS)
    cmap = get_cmap(cmap, **kw)
    norm = norm if norm else get_norm(**kw)
    out = ScalarMappable(norm=norm, cmap=cmap, dtype=dtype)
    return out
