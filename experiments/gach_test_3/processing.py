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
    from ca_analysis.processing.utils import thorimage_raw_to_h5

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
