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
    from ca_analysis.processing.utils import thorimage_raw_to_h5

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
    process_events(s, schema)

    # Make sample movie
    make_sample(s)

    return s



def do_import(mouse, date, run, day):

    path = f'sessions/{mouse}/{date}/{run}'
    remote = get_fs(-1)
    local = get_fs(0)
    pyfs.copy.copy_dir_if(remote, path, local, path, "newer", preserve_time=True)
    s = open_session(mouse, date, run, fs=0)
    import_session(s, 'seq_learn_3', day)
    pyfs.copy.copy_dir_if(local, path, remote, path, "newer", preserve_time=True)
    s.fs.rmtree("thorlabs", missing_ok=True)
    s.fs.remove("mov.h5")


if __name__ == "__main__":
    from time import perf_counter as clock
    from ca_analysis.processing.utils import thorimage_raw_to_h5

    s = open_session('61107-1', '2023-01-27', '1', fs=0)

    # thorimage_raw_to_h5(s)
    # with h5py.File(s.fs.getsyspath('scratch/mov_unprocessed_orig.h5'), 'r') as f:
    #     arr = f['data'][:20000]
    # with h5py.File(s.fs.getsyspath('scratch/mov_unprocessed.h5'), 'w') as f:
    #     f.create_dataset('data', data=arr)

    t0 = clock()
    motion_correct(s, n_processes=8)
    segment(s)
    t1 = clock()
    delta = t1 - t0
    print(f'took {format_timedelta(delta)}')
