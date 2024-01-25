"""
Motion correction utilities using caiman as a backend. Extends caiman's motion
correction ...


"""
import logging
import os
import time
import warnings
from typing import Optional, Tuple, Union

import h5py
import matplotlib
import numpy as np
from scipy.signal import savgol_filter

from ca_analysis.common import *
from ca_analysis.environment import get_fs

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from caiman.motion_correction import MotionCorrect, apply_shift_online
    from caiman.cluster import setup_cluster, stop_server

from ..common import *
from ..io import *


__all__ = [
    "run_motion_correction",
    "plot_shifts",
    "smooth_shifts",
]


logger = logging.getLogger("ca_analysis.processing.motion_correction")


def run_motion_correction(
        mov: Union[ArrayLike, PathLike],
        out: Optional[Union[ArrayLike, PathLike]] = None,
        max_shifts: Tuple[int, int] = (20, 20),
        dtype: Optional[Union[str, type]] = None,
        backend: str = "multiprocessing",
        n_processes: int = 8,
        single_thread: bool = True,
        smooth: bool = True,
        gSig_filt = None,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:

    """
    Performs motion-correction via caiman. Can run on an array already in-memory
    or an array in an h5 file. Can also return an array in-memory or saved to 
    an h5 file.
    In-memory implementations are provided for convenience when working with
    smaller datasets.

    Parameters
    ----------
    mov: array-like, path-like
        Input source
    out: array-like, path-like, optional
        Where to put motion-corrected data
    max_shifts:
        Max shifts...
    dtype: dtype-like, optional
        Optional specification for output datatype.
    

    """

    """    
    Prepare input
     
     - Handle diffrent input types:
       - case 1) `mov` is path-like: infer the name of the dataset if not given
         as part of the url.
       - case 2) `mov` is an h5py.Dataset: convert to case 1.
       - case 3) `mov` is an array (probably in-memory): store it in an h5 file 
         so caiman can use it.
       
     - Get the dtype and shape of the input data.
     
     
    """

    # os.environ["MKL_NUM_THREADS"] = 1
    # os.environ["OPENBLAS_NUM_THREADS"] = 1
    # os.environ["VECLIB_MAXIMUM_THREADS"] = 1

    tempdir = get_fs(0).mkdir("temp", exist_ok=True)

    # Determine these...
    url_in = None
    url_out = None
    shape = None
    dtype = None if dtype is None else np.dtype(dtype)

    # Prepare input
    if is_pathlike(mov):
        # case 1)
        url_in = URL(mov)
        url_in.query["name"] = url_in.query.get("name", "data")
        with h5py.File(url_in.path, "r") as f:
            dset = f[url_in.query["name"]]
            shape = dset.shape

    elif isinstance(mov, h5py.Dataset):
        # case 1)
        url_in = URL(mov.file.filename, name=mov.name)
        shape = mov.shape

    else:
        # case 3)
        url_in = URL(tempdir.getsyspath("mc_input.h5"), name="data")
        with h5py.File(url_in.path, "w") as f:
            f.create_dataset(url_in.query["name"], data=mov)
            shape = mov.shape

    """
    Prepare output
    
     - case 1) `out` is path-like: if the given dataset exists, delete it if the
       shape and/or dtype doesn't match. On the other hand, if the shape does match
       and the dtype was not specied, we're OK to use it.
     - case 2) `out` is an array-like destination. Raise an error if shapes or dtypes mismatch.
     - case 3) `out` is `None`: will not apply shifts.
     
     - If dtype is none, determine it from preexisting dataset or output array.
       Otherwise, ...
    """

    if isinstance(out, (PathLike, URL)):
        # case 1)
        url_out = URL(out)
        url_out.query["name"] = url_out.query.get("name", "data")
        if os.path.exists(url_out.path):
            with h5py.File(url_out.path, "a") as f:
                key = url_out.query["name"]
                if key in f:
                    dset = f[key]
                    if dset.shape != shape or (dtype and dtype != dset.dtype):
                        f.close()
                        raise ValueError("shape/dtype mismatch for output dataset")
                    dtype = dset.dtype

    elif hasattr(out, "__array__"):
        # case 2)
        if shape != out.shape or (dtype and dtype != out.dtype):
            raise ValueError("shape/dtype mismatch for output array")
        dtype = out.dtype

    elif out is None:
        # case 3)
        url_out = None
    else:
        raise ValueError("invalid output")

    """
    Find shifts
    """

    t_start = time.time()
    client, dview, n_processes = setup_cluster(
        backend=backend,
        n_processes=n_processes,
        single_thread=single_thread,
    )
    mc = MotionCorrect(
        [str(url_in.path)],
        var_name_hdf5=url_in.query["name"],
        max_shifts=max_shifts,
        dview=dview,
        gSig_filt=gSig_filt,
        # use_cuda=True,
    )
    mc.motion_correct()
    shifts = np.array(mc.shifts_rig)
    np.save(tempdir.getsyspath("shifts.npy"), shifts)
    template = mc.total_template_rig
    np.save(tempdir.getsyspath("template.npy"), template)

    if url_out is None:
        return shifts, template

    """
    Apply shifts
    """

    # Open input file and dataset.
    f_in = h5py.File(url_in.path, "r")
    dset_in = f_in[url_in.query["name"]]
    open_files = [f_in]

    # Open output file and dataset. Possibly determine output dtype.
    if hasattr(out, "__array__"):
        dset_out = out
    else:
        f_out = h5py.File(url_out.path, "a")
        dtype = dtype or dset_in.dtype
        dset_out = f_out.require_dataset(url_out.query["name"], shape=shape, dtype=dtype)
        open_files.append(f_out)

    # Apply shifts
    if smooth:
        shifts = smooth_shifts(shifts)
    block_size = 500
    try:
        n_frames = dset_in.shape[0]
        i_frame = 0
        while i_frame < n_frames:
            logger.debug('i_frame = {} / {}'.format(i_frame, n_frames))
            start, stop = i_frame, i_frame + block_size
            block_in = dset_in[start:stop]
            if block_in.size == 0:
                break
            block_out = apply_shift_online(block_in, shifts[start:stop])
            dset_out[start:stop] = block_out
            i_frame += block_size

    except Exception:
        for fp in open_files:
            fp.close()
        stop_server(dview=dview)
        raise

    finally:
        for fp in open_files:
            fp.close()
        stop_server(dview=dview)

    t_tot = time.time() - t_start
    mins, secs = np.divmod(t_tot, 60)
    mins, secs = int(mins), int(secs)
    logger.info(f"motion correction finished in {mins} min, {secs} sec")

    if out is None:
        with h5py.File(url_out.path, "r") as f:
            dset = f[url_out.query["name"]]
            out = dset[:]
        return out, shifts, template

    return shifts, template


def smooth_shifts(
    shifts: np.ndarray,
    window_length: int = 51,
    polyorder: int = 3,
    **kw,
) -> np.ndarray:

    smoothed = np.zeros_like(shifts)
    smoothed[:, 0] = savgol_filter(shifts[:, 0], window_length, polyorder, **kw)
    smoothed[:, 1] = savgol_filter(shifts[:, 1], window_length, polyorder, **kw)
    return smoothed


def plot_shifts(
    s: "Session",
    show: bool = True,
) -> Tuple["matplotlib.figure.Figure", "matplotlib.axes.Axes"]:

    shifts = np.load(s.fs.getsyspath("scratch/shifts.npy"))
    template = np.load(s.fs.getsyspath("scratch/template.npy"))
    fig = matplotlib.figure.Figure()
    ax = [fig.add_subplot(2, 1, 1), fig.add_subplot(2, 1, 2)]

    ax[0].plot(shifts[:, 0], label='x')
    ax[0].plot(shifts[:, 1], label='y')
    ax[0].legend()
    ax[0].set_title('shifts')

    ax[1].imshow(template, cmap="gray")
    ax[1].set_aspect('equal')
    ax[1].set_title('template')
    if show:
        matplotlib.pyplot.show()

    return fig, ax
