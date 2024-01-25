import datetime
import logging
import os
from pathlib import Path
from typing import (
    Optional,
    Union,
)
import fs as pyfs
from fs.errors import *
import numpy as np
import xarray as xr

from ca_analysis import *
from ca_analysis.environment import get_fs

EXPERIMENT = "gach_test_3"
schema = EventSchema(get_fs(0).getsyspath(f"event_schemas/{EXPERIMENT}.yaml"))
logger = logging.getLogger(f"ca_analysis.experiments.{EXPERIMENT}")
logging.basicConfig(level=logging.INFO)


def open_session(
        mouse: str,
        date: Union[str, datetime.date],
        run: Union[int, str] = "1",
        fs: Optional[Union[int, str]] = None,
        create: bool = False,
) -> Session:

    date = as_date(date)
    run = str(run)
    stem = os.path.join("sessions", mouse, str(date), run)

    parent_fs = get_fs(fs)
    try:
        fs = parent_fs.opendir(stem)
    except ResourceNotFound:
        if not create:
            raise
        fs = parent_fs.makedirs(stem)

    s = Session(fs, mouse=mouse, date=date, run=run)
    s.event_class = EventModel
    s.segmentation_class = None

    # thorlabs
    s.reg.add("thorlabs", "thorlabs")
    s.reg.add("thor_md", "thorlabs/Experiment.xml")
    s.reg.add("thor_raw", "thorlabs/Image.raw")
    s.reg.add("thor_sync", "thorlabs/Episode.h5")

    # events
    s.reg.add("events", "events")
    s.reg.add("schema", "events/schema.yaml")
    s.reg.add("frames_table", "events/frames.csv")
    s.reg.add("events_table", "events/events.csv")

    # motion correction
    s.reg.add("mc", "motion_correction")
    s.reg.add("mc_shifts", "motion_correction/shifts.npy")
    s.reg.add("mc_template", "motion_correction/template.npy")
    s.reg.add("mc_summary", "motion_correction/summary.pdf")

    # scratch
    s.reg.add("scratch", "scratch")
    s.reg.add("mov_unprocessed", "scratch/mov_unprocessed.h5")
    s.reg.add("mov_mc", "scratch/mov_motion_corrected.h5")
    s.reg.add("background_jpeg", "scratch/background.jpeg")
    s.reg.add("background_npy", "scratch/background.npy")
    s.reg.add("mask_tif", "scratch/mask.tif")
    s.reg.add("mask_npy", "scratch/mask.npy")
    s.reg.add("mean_frame", "scratch/mean_frame.npy")

    # segmentation
    s.reg.add("segmentation", "segmentation")

    # analysis files
    s.reg.add("analysis", "analysis")

    # etc
    s.reg.add("attrs", "attrs.yaml")
    s.reg.add("mov", "mov.h5")
    s.reg.add("sample", "sample.mp4")

    s.G = GData(s)

    return s


class GData(SessionData):

    def _prepare(self):

        data = np.load(self.session.fs.getsyspath("G.npy"))
        arr = xr.DataArray(data, dims=('time',))
        self.data = arr
