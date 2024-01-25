from numbers import Number
from typing import (
    Optional,
    Union,
)

import matplotlib as mpl
import numpy as np
import skimage.color
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from .color import *

__all__ = [
    "blend_images",
    "blend_pixels",
    "gray_to_rgb",
    "gray_to_rgba",
    "overlay_roi",
    "overlay_scalebar",
]

"""
Axis control

"""


def blend_images(
    A: np.ndarray,
    B: np.ndarray,
    alpha: Optional[Number] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Blend two RGBA images together.
    """

    if out is None:
        out = np.zeros(A.shape, dtype=float)

    A_rgb = A[:, :, 0:3]
    A_a = np.expand_dims(A[:, :, 3], 0)
    A_mul = A_rgb * A_a

    B_rgb = B[:, :, 0:3]
    B_a = np.expand_dims(B[:, :, 3], 0)
    B_mul = B_rgb * B_a

    out = np.zeros_like(A)
    out[:, :, 0:3] = A_mul + B_mul * (1 - A_a)
    out[:, :, 3] = np.squeeze(A_a + B_a * (1 - A_a))
    return out


def blend_pixels(
    im: np.ndarray,
    row_coords: np.ndarray,
    col_coords: np.ndarray,
    color: np.ndarray,
    alpha: Optional[Union[Number, np.ndarray]] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Blend pixel values with/in an rgba image.

    """

    if out is None:
        out = im

    # split background in rgb and alpha, and multiply them.
    bg = im[row_coords, col_coords]
    bg_rgb = bg[:, 0:3]  # shape: (n_pix, 3)
    bg_a = np.expand_dims(bg[:, 3], 1)  # shape: (n_pix, 1)
    bg_mul = bg_rgb * bg_a

    # get foreground as rgba.
    if isinstance(color, str):
        color = to_rgba(color)
    color = np.asarray(color)

    # if color is 1d, apply same color to all pixels.
    if color.ndim == 1:
        color = np.expand_dims(color, 0)  # shape: (1, 3/4)

    # color doesn't contain alpha, assume it's 1.
    if color.shape[1] == 3:
        fg_rgb = color
        fg_a = np.atleast_2d(1)
    else:
        fg_rgb = color[:, 0:3]
        fg_a = color[:, 3]

    # if given an alpha, use it.
    if alpha is not None:
        alpha = np.array(alpha)
        if alpha.ndim == 0:
            fg_a = np.atleast_2d(alpha)
        else:
            fg_a = np.expand_dims(alpha, 1)

    # multiply foreground rgb by its alpha
    fg_mul = fg_rgb * fg_a

    # perform blending
    out[row_coords, col_coords, :3] = fg_mul + bg_mul * (1 - fg_a)
    out[row_coords, col_coords, -1] = np.squeeze(fg_a + bg_a * (1 - fg_a))

    return out


def gray_to_rgb(
    im: np.ndarray,
    alpha: Optional[Number] = None,
    dtype: Optional[Union[str, type]] = None,
) -> np.ndarray:
    """
    Converts a grayscale image to an rgb image.
    """
    out = skimage.color.gray2rgb(im, alpha=alpha)
    if dtype:
        dtype = np.dtype(dtype)
        if np.issubdtype(dtype, np.integer):
            out = (out * 255).astype(dtype)
    return out


def gray_to_rgba(
    im: np.ndarray,
    alpha: Optional[Number] = None,
    dtype: Optional[Union[str, type]] = None,
) -> np.ndarray:
    """
    Converts a grayscale image to an rgba image.
    """
    out = skimage.color.gray2rgba(im, alpha=alpha)
    if dtype:
        dtype = np.dtype(dtype)
        if np.issubdtype(dtype, np.integer):
            out = (out * 255).astype(dtype)
    return out


def overlay_roi(
    im: np.ndarray,
    roi: "ROI",
    color: ColorLike,
    alpha: Optional[Number] = None,
    boundary: Union[bool, ColorLike] = False,
    boundary_alpha: Optional[Number] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    mask = roi.mask
    rows, cols, lam = mask.y, mask.x, mask.z
    if alpha is None:
        alpha = lam - lam.min()
        alpha = alpha / np.max(alpha)

    blend_pixels(im, rows, cols, color, alpha=alpha, out=out)

    if boundary is not False:
        if mask.shape == im.shape[0:2]:
            b = mask.boundary
        else:
            b = mask.get_boundary(shape=im.shape[0:2])
        blend_pixels(im, b.rows, b.cols, boundary, alpha=boundary_alpha, out=out)

    return out


def overlay_scalebar(
    ax: mpl.axes.Axes,
    pixsize=None,
    size=None,
    label=None,
    loc=3,
    color='white',
    frameon=False,
    pad=0.25,
    sep=2.5,
    fontsize=12,
    **kw,
) -> mpl.axes.Axes:
    """
    Add a scalebar to a field-of-view. Tries to guess how big the scale
    bar should be based on the data.

    Parameters
    ----------

    ax: matplotlib.axes.Axes

    ses: Session

    pixsize: float (optional)

    size: float
        Length of scalebar in microns.


    """

    if size is None:
        xmin, xmax = ax.get_xlim()
        W = (xmax - xmin) * pixsize
        est = W / 10

        if W <= 10:
            arr = np.arange(1, W, 1)

        elif W <= 50:
            arr = np.arange(5, W, 5)

        elif W <= 500:
            arr = np.arange(10, W, 10)

        else:
            arr = np.arange(100, W, 100)

        dist = np.argmin(np.abs(arr - est))
        size = int(arr[dist])
        label = None

    if label is None:
        if size < 1000:
            label = '${}\mu $m'.format(size)
        else:
            label = '${:.1f}$mm'.format(size / 1000)

    fp = kw.get('fontproperties', FontProperties())
    if fontsize:
        fp.set_size(fontsize)

    bar = AnchoredSizeBar(
        ax.transData,
        size,
        label,
        loc,
        color=color,
        frameon=frameon,
        pad=pad,
        sep=sep,
        fontproperties=fp,
        **kw,
    )
    ax.add_artist(bar)
    return ax

