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

from ca_analysis import *
from ca_analysis.environment import get_fs


schema = EventSchema(get_fs(0).getsyspath("event_schemas/retinotopy_2.yaml"))
logger = logging.getLogger(f"ca_analysis.experiments.retinotopy_2")
logging.basicConfig(level=logging.INFO)


def open_session(
        mouse: str,
        date: Union[str, datetime.date],
        run: Union[int, str] = "1",
        fs: Optional[Union[int, str]] = 0,
        require: bool = False,
) -> Session:

    date = as_date(date)
    run = str(run)
    stem = os.path.join("sessions", mouse, str(date), run)

    if fs is None:
        mfs = get_fs()
        try:
            ses_fs = mfs.opendir(stem)
        except ResourceNotFound:
            if not require:
                raise
            mfs.write_fs.makedirs(stem)
            ses_fs = mfs.opendir(stem)
    else:
        fs_branch = get_fs(fs)
        try:
            ses_fs = fs_branch.opendir(stem)
        except ResourceNotFound:
            if not require:
                raise
            ses_fs = fs_branch.makedirs(stem)

    s = Session(ses_fs, mouse=mouse, date=date, run=run)
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

    # segmentation
    s.reg.add("segmentation", "segmentation")

    # analysis files
    s.reg.add("analysis", "analysis")

    # etc
    s.reg.add("attrs", "attrs.yaml")
    s.reg.add("mov", "mov.h5")
    s.reg.add("sample", "sample.mp4")
    s.reg.add("retinotopy", "retinotopy.mp4")

    return s



def push(
    s: Session,
    name: str,
    src: Union[int, str] = 0,
    dst: Union[int, str] = -1,
    condition: str = "newer",
    missing_ok: bool = False,
) -> None:

    stem = f"sessions/{s.mouse}/{s.date}/{s.run}"
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
    stem = f"sessions/{s.mouse}/{s.date}/{s.run}"
    src_fs = get_fs(src).opendir(stem)
    if name in s.reg:
        path = s.reg['mov'].path
    else:
        path = name
    src_fs.delete(path, missing_ok=missing_ok)

