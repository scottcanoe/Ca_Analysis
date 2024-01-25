from numbers import Number
import shutil
from pathlib import Path
from typing import (
    Optional,
    Tuple, Union,
)
import fs as pyfs
import h5py
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

xr.set_options(keep_attrs=True)

from ca_analysis import *
# from ca_analysis.io.thorlabs import *
from ca_analysis.processing.utils import (
    init_attrs,
    make_sample,
    motion_correct,
    process_events,
    segment,
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

    logger.info('Importing session {} on fs {}'.format(s, s.fs))
    s.reg.analysis.mkdir(exist_ok=True)
    s.reg.scratch.mkdir(exist_ok=True)

    # Collect attributes from metadata files.
    init_attrs(s, schema=schema.name)

    # Convert raw movie to 'mov_unprocessed.h5'.
    thorimage_raw_to_h5(s)
    if s.reg["mov_unprocessed"].exists():
        s.fs.rm(s.reg["thor_raw"].path, missing_ok=True)

    # Perform motion correction.
    motion_correct(s, single_thread=True)
    if s.reg["mov"].exists():
        s.fs.rm(s.reg["mov_unprocessed"].path, missing_ok=True)

    # Perform segmentation.
    segment(s)

    # Process event data.
    process_events(s)
    s.events.load()

    # Make sample mov.
    make_sample(s)

    return s


def push(
    s: Session,
    name: str,
    src: Union[int, str] = 0,
    dst: Union[int, str] = -1,
    condition: str = "newer",
    missing_ok: bool = False,
) -> None:

    stem = f"sessions/{s.mouse}/{s.date}/{s.exp}"
    src_fs = get_fs(src).opendir(stem)
    dst_fs = get_fs(dst).opendir(stem)
    if name in s.reg:
        path = s.reg['mov'].path
    else:
        path = name

    if not src_fs.exists(path):
        if missing_ok:
            return
        raise FileNotFoundError(src_fs.getsyspath(path))

    if src_fs.isdir(path):
        pyfs.copy.copy_dir_if(src_fs, path, dst_fs, path, condition)
    else:
        pyfs.copy.copy_file_if(src_fs, path, dst_fs, path, condition)


def pull(
    s: Session,
    name: str,
    src: Union[int, str] = -1,
    dst: Union[int, str] = 0,
    condition: str = "newer",
    missing_ok: bool = False,
) -> None:
    push(s, name, src, dst, condition, missing_ok)


def backup_and_delete(
    s: Session,
    name: str,
    src: Union[int, str] = 0,
    dst: Union[int, str] = -1,
    condition: str = "newer",
    missing_ok: bool = False,
) -> None:

    push(s, name, src, dst, condition, missing_ok)
    stem = f"sessions/{s.mouse}/{s.date}/{s.exp}"
    src_fs = get_fs(src).opendir(stem)
    if name in s.reg:
        path = s.reg['mov'].path
    else:
        path = name
    src_fs.delete(path, missing_ok=missing_ok)


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

