import datetime
import logging
from pathlib import Path
from typing import (
    Optional,
    Union,
)

# common imports
import numpy as np
import pandas as pd
import xarray as xr

from ca_analysis import *
from ca_analysis.environment import get_fs
from ca_analysis.io.suite2p import Suite2PStore

__experiment__ = "seq_learn_3"
schema = get_event_schema(__experiment__)
logger = logging.getLogger("ca_analysis")
logging.basicConfig(level=logging.INFO)



def import_session(s: Session, schema: str, day: int) -> Session:
    """
    First step of the pipeline. The sole purpose of this function is to handle
    raw measurements and put them into a standardized form. This includes
    placing the reformatted data according to a file scheme.

    Note that this is not meant to do anything that alters the underlying data
    in any significant way, such as filtering. It's only meant to standardize
    the format of the raw data.

    """
    from ca_analysis.processing.utils import (
        init_attrs,
        motion_correct,
        segment,
        process_events,
        make_sample,
        thorimage_raw_to_h5,
    )

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



def open_session(
    mouse: str,
    date: Union[str, datetime.date],
    run: Union[int, str] = "1",
    fs: Optional[Union[int, str]] = None,
) -> Session:


    stem = f"sessions/{mouse}/{date}/{run}"
    ses_fs = get_fs(fs).opendir(stem)
    s = Session(fs=ses_fs, mouse=mouse, date=date, run=run)

    s.segmentation_class = Suite2PStore

    # thorlabs
    s.reg["thorlabs"] = "thorlabs"
    s.reg.add("thor_md", "thorlabs/Experiment.xml"),
    s.reg.add("thor_raw", "thorlabs/Image.raw"),
    s.reg.add("thor_sync", "thorlabs/Episode.h5"),

    # events
    s.reg.add("events", "events")
    s.reg.add("schema", "events/schema.yaml")
    s.reg.add("frames_table", "events/frames.csv")
    s.reg.add("events_table", "events/events.csv")
    s.reg.add("sequences_table", "events/sequences.csv")
    s.reg.add("blocks_table", "events/blocks.csv")

    # motion correction
    s.reg.add("mc", "motion_correction"),
    s.reg.add("mc_shifts", "motion_correction/shifts.npy"),
    s.reg.add("mc_template", "motion_correction/template.npy"),
    s.reg.add("mc_summary", "motion_correction/summary.pdf"),

    # scratch
    s.reg.add("scratch", "scratch"),
    s.reg.add("mov_unprocessed", "scratch/mov_unprocessed.h5?name=data"),

    # segmentation
    s.reg.add("segmentation", "suite2p"),

    # analysis files
    s.reg.add("analysis", "analysis"),

    # etc
    s.reg.add("attrs", "attrs.yaml"),
    s.reg.add("mov", "mov.h5?name=data"),
    s.reg.add("sample", "sample.mp4"),

    # data handlers
    s.spikes = FluorescenceData(s, 'spikes')
    s.F = FluorescenceData(s, 'F')
    s.Fneu = FluorescenceData(s, 'Fneu')
    s.mov = MovieData(s)

    return s


"""
--------------------------------------------------------------------------------
"""


def get_sessions(
    filename: PathLike = "sessions.ods",
    fs: Optional[Union[int, str]] = None,
    **filters,
) -> SessionGroup:

    path = Path(filename)
    if not path.is_absolute():
        path = Path(__file__).parent / path

    df = pd.read_excel(path)
    mice = df['mouse'].dropna()
    df = df.loc[mice.index]
    df['date'] = df['date'].astype(str)
    df['run'] = df['run'].astype(int).astype(str)
    if 'day' in df.columns:
        df['day'] = df['day'].astype(int)

    df['has_day_0'] = df['has_day_0'].astype(bool)

    # filter enabled
    df['enabled'] = df['enabled'].astype(bool)
    df = df[df['enabled']]

    # filter others
    for key, val in filters.items():
        df = df[df[key] == val]
    group = SessionGroup()
    for i in range(len(df)):
        row = df.iloc[i]
        group.append(open_session(row.mouse, row.date, row.run, fs=fs))

    return group
