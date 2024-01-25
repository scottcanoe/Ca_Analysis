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
import pandas as pd
import xarray as xr
xr.set_options(keep_attrs=True)

from ca_analysis import *
from ca_analysis.io.thorlabs import *
from ca_analysis.plot import *

from main import *


def import_session(s: Session, schema: str, **attrs) -> Session:


    """
    Pre-processing pipeline:
     - Create session directory
     - create 'raw' directory, and put imaging data in as 'mov.h5' and trigger
       data as 'triggers.mat'.
     - Open MATLAB, and use script ... to create 'scratch/triggers.h5' 
     - Run 'init_attrs()' with appropriate schema.
     - Run 'decode_event_bits()'
     - Run 'process_events()'
     - Write only wanted imaging data to 'mov.h5', truncating to match... 
    """

    import matlab.engine

    logger.info('Importing session {} on fs {}.'.format(s, s.fs))

    s.reg.analysis.mkdir(exist_ok=True)
    s.reg.scratch.mkdir(exist_ok=True)

    # Collect attributes from metadata files.
    init_attrs(s, schema=schema, **attrs)

    # create 'scratch/triggers.h5'
    logger.info('Converting trigger info')
    engine = matlab.engine.start_matlab()
    engine.rewrite_triggers(s.fs.getsyspath(""), nargout=0)
    engine.quit()

    # decode bits to create 'event' dataset in 'scratch/triggers.h5'
    logger.info('Decoding event bits and processing events')
    decode_event_bits(s)

    # process events (creates 'events' directory)
    process_events(s)

    if schema.startswith('retinotopy'):
        fix_retinotopy_events(s)

    # move imaging data
    raw_h5s = list(Path(s.fs.getsyspath('raw')).glob('*.h5'))
    if raw_h5s and not s.fs.exists('mov.h5'):
        logger.info('Moving imaging data')
        move_imaging_data(s)

    return s



def decode_event_bits(s: Session) -> None:

    path = s.fs.getsyspath('scratch/triggers.h5')
    f = h5py.File(path, 'r+')
    time = f['Time'][:].squeeze()
    n_timepoints = len(time)
    # vis = f['VIS'][:].squeeze().astype(int)
    # frame_out = f['VIS'][:].squeeze().astype(int)
    bits = np.zeros([n_timepoints, 4], dtype=int)
    bits[:, 0] = f['BIT0'][:].squeeze().astype(int)
    bits[:, 1] = f['BIT1'][:].squeeze().astype(int)
    bits[:, 2] = f['BIT2'][:].squeeze().astype(int)
    bits[:, 3] = f['BIT3'][:].squeeze().astype(int)

    events = np.zeros(n_timepoints, dtype=int)
    for i in range(n_timepoints):
        row = bits[i]
        string = "".join(row.astype(str))[::-1]
        code = int(string, 2)
        events[i] = code

    f.create_dataset('event', data=events)


def load_sync_data(s: Session) -> pd.DataFrame:
    with h5py.File(s.fs.getsyspath('scratch/triggers.h5'), "r") as f:
        data = {}
        data['time'] = f["Time"][:].squeeze()
        data['Strobe'] = f['VIS'][:].squeeze().astype(np.int8)
        data['FrameOut'] = f['LED470Signal'][:].squeeze().astype(np.int8)
        data['AnalogEvents'] = f['event'][:].squeeze()
        sd = pd.DataFrame(data)
    return sd


def init_attrs(
    s: Session,
    merge: bool = False,
    update: bool = False,
    force: bool = False,
    **extras,
) -> None:


    if s.fs.exists("attrs.yaml") and not force:
        return
    attrs = {}

    # - capture...
    attrs["capture"] = {}
    attrs['capture']['modality'] = 'camera'

    # - schema
    schema_path = Path(s.fs.getsyspath("events/schema.yaml"))
    if schema_path.exists():
        schema = EventSchema(schema_path)
        attrs["schema"] = schema.name

    # - etc.
    parts = s.fs.getsyspath("").split("/")
    attrs["mouse"] = parts[-3]
    attrs["date"] = parts[-2]
    attrs["run"] = parts[-1]
    attrs['samplerate'] = 10.0

    attrs.update(extras)
    if merge:
        s.attrs.merge(attrs)
    elif update:
        s.attrs.update(attrs)
    else:
        s.attrs.clear()
        s.attrs.update(attrs)
    s.attrs.save()


def process_events(
    s: Session,
    schema: Optional[Union[str, EventSchema]] = None,
    force: bool = False,
) -> None:

    from ca_analysis.processing.alignment import SignalParser

    if s.fs.exists("events"):
        if not force:
            return
        s.fs.removetree("events")

    if schema is None:
        schema_name = s.attrs["schema"]
    elif is_str(schema):
        schema_name = schema
    elif isinstance(schema, EventSchema):
        schema_name = schema.name
    else:
        raise ValueError

    logging.info('Processing events')
    s.fs.makedir("events")

    src_file = get_fs().getsyspath(f"event_schemas/{schema_name}.yaml")
    dst_file = s.fs.getsyspath("events/schema.yaml")
    shutil.copy(src_file, dst_file)
    schema = EventSchema(dst_file)
    strobe_anchor = schema.attrs["strobe_anchor"]
    frame_anchor = schema.attrs["frame_anchor"]

    # Run the stimulus alignment tool on the sync_data.
    sync_data = load_sync_data(s)
    parser = SignalParser(schema, sync_data)
    parser.strobe_anchor = strobe_anchor
    parser.frame_anchor = frame_anchor
    parser.parse()
    frame_df = parser.tables["frame"]
    del frame_df["start"]
    del frame_df["stop"]
    del frame_df["mid"]
    frame_df.to_csv(s.fs.getsyspath("events/frames.csv"))
    if "event" in parser.tables:
        event_df = parser.tables["event"]
        event_df.to_csv(s.fs.getsyspath("events/events.csv"))
    if "sequence" in parser.tables:
        seq_df = parser.tables["sequence"]
        seq_df.to_csv(s.fs.getsyspath("events/sequences.csv"))

    strobe_df = parser.tables['strobe']
    strobe_df.to_csv(s.fs.getsyspath('events/strobes.csv'))

    s.events.load()



def move_imaging_data(s: Session) -> None:

    src_path = find_file('*.h5', root_dir=s.fs.getsyspath('raw'), absolute=True)
    f_src = h5py.File(src_path, 'r')
    f_dst = h5py.File(s.fs.getsyspath('mov.h5'), 'w')

    if 'rfp' in f_src and 'norm' in f_src['rfp']:
        src, name = f_src['rfp']['norm'][:], 'ca'
        n_frames = src.shape[0]
        shape = (n_frames, src.shape[2], src.shape[1])
        dst = f_dst.create_dataset(name, shape=shape, dtype=np.float32, maxshape=shape)
        for i in range(n_frames):
            dst[i] = np.fliplr(np.rot90(src[i]))
        src = None

    if 'gfp' in f_src and 'normHD' in f_src['gfp']:
        src, name = f_src['gfp']['normHD'][:], 'ach'
        n_frames = src.shape[0]
        shape = (n_frames, src.shape[2], src.shape[1])
        dst = f_dst.create_dataset(name, shape=shape, dtype=np.float32, maxshape=shape)
        for i in range(n_frames):
            dst[i] = np.fliplr(np.rot90(src[i]))
        src = None

    f_src.close()
    f_dst.close()


def fix_retinotopy_events(s: Session) -> None:
    path = s.fs.getsyspath('events/events.csv')
    df = pd.read_csv(path, index_col=0)
    events = df['event']
    starts = df['start']
    stops = df['stop']
    length = stops - starts
    events[length == 1] = 0
    df = pd.DataFrame({'event': events, 'start': starts, 'stop': stops})
    df.to_csv(path)
