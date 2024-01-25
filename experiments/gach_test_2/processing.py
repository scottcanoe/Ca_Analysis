from numbers import Number
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import (
    Optional,
    Tuple,
    Union,
)
import h5py
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# from ca_analysis.io.common import URL

xr.set_options(keep_attrs=True)

from ca_analysis import *
from ca_analysis.io.thorlabs import *
from ca_analysis.plot import *

from ca_analysis.processing.utils import (
    init_attrs,
    make_sample,
    motion_correct,
    process_events,
    thorimage_raw_to_h5,
)

from main import *


def import_session(s: Session) -> Session:

    """
    First step of the pipeline. The sole purpose of this function is to handle
    raw measurements and put them into a standardized form. This includes
    placing the reformatted data according to a file scheme.

    Note that this is not meant to do anything that alters the underlying data
    in any significant way, such as filtering. It's only meant to standardize
    the format of the raw data.

    """

    logger.info('Importing session {} on fs {}.'.format(s, s.fs))

    s.reg.analysis.mkdir(exist_ok=True)
    s.reg.scratch.mkdir(exist_ok=True)

    # Collect attributes from metadata files.
    init_attrs(s, schema=schema.name)

    # Convert raw movie to 'mov_unprocessed.h5'.
    thorimage_raw_to_h5(s)
    if s.reg["mov_unprocessed"].exists():
        s.fs.rm(s.reg["thor_raw"].path, missing_ok=True)

    # Perform motion correction.
    motion_correct(s, src="mov_unprocessed", dst="mov_mc", single_thread=True)
    # if s.reg["mov_mc"].exists():
    #     s.fs.rm(s.reg["mov_unprocessed"].path, missing_ok=True)
    # create_mask(s)
    # Perform luminance correction.
    # run_luminance_filter(s)
    # if s.reg["mov"].exists():
    #     s.fs.rm(s.reg["mov_mc"].path, missing_ok=True)

    # Process event data.
    process_events(s)

    # Make sample mov.
    # make_sample(s)

    return s


def create_mask(s: Session) -> np.ndarray:
    import imageio
    import subprocess as sp
    from skimage.color import gray2rgb

    # Find a movie we can use for the background image.
    background = None
    for key in ("mov", "mov_mc"):
        p = s.reg[key].resolve()
        if p.exists():
            with h5py.File(p, "r") as f:
                background = f["data"][:50]
                background = np.mean(background, axis=0)
            break

    if background is None:
        raise FileNotFoundError("no movie found")

    # Save the background as a jpeg.
    background = (250 * (background / background.max())).astype(np.uint8)
    rgba = gray2rgb(background).astype(np.uint8)
    imageio.imsave(s.reg["background_jpeg"].resolve(), rgba)
    np.save(s.reg["background_npy"].resolve(), background)

    # Launch krita
    msg = """
    Mask Creation
    ------------
    - 1. Open {}/'background.jpeg'
    - 2. Use the Elliptical Selection Tool to select the ROI (probably everything
         that is window)
    - 3. Right click on the selection, and select 'Cut selection to new layer'.
    - 4. Make only the new layer visible.
    - 5. Save as '{}/mask.tif' and quit krita.

    """.format(str(s.reg["scratch"].resolve()), str(s.reg["scratch"].resolve()))
    print(msg, flush=True)

    tif_path = s.reg["mask_tif"].resolve()
    if tif_path.exists():
        tif_path.unlink()
    proc = sp.run(["krita", "--nosplash"], stdout=sp.PIPE, stderr=sp.STDOUT)
    if not proc:
        msg = "There was an error with krita: \n"
        raise RuntimeError(msg)

    # Load mask.
    mat = imageio.v2.imread(tif_path)
    mask = np.zeros((mat.shape[0], mat.shape[1]), dtype=bool)
    mask[mat[:, :, -1] == 0] = True
    np.save(s.reg["mask_npy"].resolve(), mask)

    for key in ("mov", "mov_mc"):
        p = s.reg[key].resolve()
        if p.exists() and p.suffix.lower() in {".h5", ".hdf", ".hdf5"}:
            with h5py.File(p, "r+") as f:
                if "mask" in f.keys():
                    del f["mask"]
                f.create_dataset("mask", data=mask)

    return mask

