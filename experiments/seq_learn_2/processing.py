from numbers import Number
import shutil
from pathlib import Path
from typing import (
    Optional,
    Tuple, Union,
)
import fs as pyfs
import fs.errors
import h5py
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

xr.set_options(keep_attrs=True)

from ca_analysis import *
# from ca_analysis.io.thorlabs import *
from ca_analysis.processing.utils import *


from main import *



def import_session(s: Session, schema: str, day: int) -> Session:
    """
    First step of the pipeline. The sole purpose of this function is to handle
    raw measurements and put them into a standardized form. This includes
    placing the reformatted data according to a file scheme.

    Note that this is not meant to do anything that alters the underlying data
    in any significant way, such as filtering. It's only meant to standardize
    the format of the raw data.

    """

    logger.info(f"Importing {schema} session {s.mouse}/{s.date}/{s.run}")

    s.reg.analysis.mkdir(exist_ok=True)
    s.reg.scratch.mkdir(exist_ok=True)

    # Collect attributes from metadata files.
    init_attrs(s, schema=schema, day=day)

    # Convert raw movie to 'mov_unprocessed.h5'.
    thorimage_raw_to_h5(s)
    if s.reg["mov_unprocessed"].exists():
        s.fs.rm(s.reg["thor_raw"].path, missing_ok=True)

    # Perform motion correction.
    motion_correct(s, n_processes=4, single_thread=True)
    if s.reg["mov"].exists():
        s.fs.rm(s.reg["mov_unprocessed"].path, missing_ok=True)

    # Perform segmentation.
    segment(s)

    # Process event data.
    process_events(s, schema, day)

    # Make sample movie
    make_sample(s)

    return s



def backup_and_delete(
    s: Session,
    name: str,
    src: Union[int, str] = 0,
    dst: Union[int, str] = -1,
    condition: str = "newer",
    missing_ok: bool = False,
) -> None:

    stem = f"sessions/{s.mouse}/{s.date}/{s.run}"
    src_fs = get_fs(src).opendir(stem)
    if not src_fs.exists(name):
        if missing_ok:
            return
        raise fs.errors.ResourceNotFound(name)

    push_session(s, name, src, dst, condition, missing_ok)
    src_fs.delete(name, missing_ok=missing_ok)


def cleanup(s: Session):
    """
    Remove temporary files created during preprocessing.

    Parameters
    ----------
    s

    Returns
    -------

    """

    if s.reg["mov"].exists():
        s.fs.rm(s.reg["mov_unprocessed"].path, missing_ok=True)
        s.fs.rm(s.reg["thor_raw"].path, missing_ok=True)
    if s.reg["mov_unprocessed"].path.exists():
        s.fs.rm(s.reg["thor_raw"].path, missing_ok=True)

