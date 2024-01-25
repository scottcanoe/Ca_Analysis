import logging
import shutil
from numbers import Number
from pathlib import Path
from typing import (
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from ca_analysis import *

__all__ = [
    "finalize_alignment",
    "fix_frame_trigger",
    "init_attrs",
    "make_sample",
    "motion_correct",
    "process_events",
    "pull_session",
    "push_session",
    "segment",
    "thorimage_raw_to_h5",
]


def push_session(
    s: Session,
    name: str = "*",
    src: Union[int, str] = 0,
    dst: Union[int, str] = -1,
    condition: str = "newer",
    missing_ok: bool = False,
) -> None:

    import fs as pyfs

    stem = f"sessions/{s.mouse}/{s.date}/{s.run}"
    logging.info(f"pushing session: {stem}")
    try:
        src_fs = get_fs(src).opendir(stem)
    except pyfs.errors.ResourceNotFound:
        if missing_ok:
            return
        raise

    dst_fs = get_fs(dst).mkdir(stem, exist_ok=True, parents=True)

    if is_str(name):
        if "*" in name:
            paths = find_files("*", root_dir=src_fs.getsyspath(""), relative=True)
        elif name in s.reg:
            paths = [s.reg[name].path]
        else:
            paths = [name]
    else:
        paths = name

    for p in paths:
        if not src_fs.exists(p):
            if missing_ok:
                continue
            else:
                raise ResourceNotFound(p)

        if src_fs.isdir(p):
            pyfs.copy.copy_dir_if(src_fs, p, dst_fs, p, condition)
        else:
            pyfs.copy.copy_file_if(src_fs, p, dst_fs, p, condition)

    logging.info(f"finished pushing session: {stem}")


def pull_session(
    s: Session,
    name: str = "*",
    src: Union[int, str] = -1,
    dst: Union[int, str] = 0,
    condition: str = "newer",
    missing_ok: bool = False,
) -> None:
    push_session(s, name, src, dst, condition, missing_ok)



def thorimage_raw_to_h5(s: Session, force: bool = False) -> None:
    from ca_analysis.io.thorlabs import thorimage_raw_to_h5
    names = ("mov_unprocessed", "mov")
    if any(s.fs.exists(s.reg[p].path) for p in names) and not force:
        return
    src_url = URL(s.reg["thor_raw"].resolve())
    if not Path(src_url.path).exists():
        return
    logging.info("Converting raw imaging data")
    s.reg.scratch.mkdir(exist_ok=True)
    dst_url = URL(s.reg["mov_unprocessed"].resolve(), name="data")
    thorimage_raw_to_h5(src_url, dst_url)


def init_attrs(
    s: Session,
    merge: bool = False,
    update: bool = False,
    force: bool = False,
    **extras,
) -> Mapping:
    from ca_analysis.io.thorlabs import read_thorimage_metadata

    if s.fs.exists("attrs.yaml") and not force:
        return
    attrs = {}

    # - thorlabs attrs
    md_path = Path(s.fs.getsyspath("thorlabs/Experiment.xml"))
    if md_path.exists():
        md = read_thorimage_metadata(md_path)
        attrs["capture"] = md

    # - schema
    schema_path = Path(s.fs.getsyspath("events/schema.yaml"))
    if schema_path.exists():
        schema = EventSchema(schema_path)
        attrs["schema"] = schema.name

    # - etc.
    parts = s.fs.getsyspath("").split("/")
    attrs["mouse"] = parts[-3]
    attrs["date"] = parts[-2]
    attrs["exp"] = parts[-1]
    attrs['samplerate'] = attrs['capture']['frame']['rate']

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
    from ca_analysis.io.thorlabs import read_thorsync_data
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
    sync_data = read_thorsync_data(s.fs.getsyspath("thorlabs/Episode.h5"))
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
    s.events.load()
    finalize_alignment(s)
    s.events.load()


def finalize_alignment(s: Session, tolerance: int = 50) -> None:

    import h5py
    import pandas as pd
    import warnings

    if not s.fs.exists("events/frames.csv"):
        return

    if s.fs.exists("mov.h5"):
        with h5py.File(s.fs.getsyspath("mov.h5"), "r") as f:
            n_frames_data = f["data"].shape[0]
    elif s.fs.exists("suite2p/F.npy"):
        n_frames_data = np.load(s.fs.getsyspath("suite2p/F.npy")).shape[1]
    else:
        return

    frame_df = pd.read_csv(s.fs.getsyspath("events/frames.csv"), index_col=0)
    n_frames_sync = len(frame_df)
    diff = n_frames_data - n_frames_sync
    if diff == 0:
        return

    if abs(diff) > tolerance:
        msg = f"{s}: n_frames_data={n_frames_data}, " \
              f"n_frames_sync={n_frames_sync}, diff={diff}"
        msg += ". Disparity too great. Something went wrong. Skipping adjustment."
        warnings.warn(msg)
        return

    if diff > 0:
        # extend event files
        tail_data = {}
        for colname in frame_df.columns:
            col = frame_df[colname]
            tail_data[colname] = np.full(diff, col.values[-1], dtype=col.dtype)
        tail_index = np.arange(frame_df.index[-1] + 1, frame_df.index[-1] + 1 + diff)
        period = np.median(np.ediff1d(frame_df["time"]))
        tail_data["time"] = period * np.arange(1, diff + 1) + frame_df["time"].values[-1]
        tail_df = pd.DataFrame(tail_data, index=tail_index)
        frame_df = pd.concat([frame_df, tail_df])
        assert len(frame_df) == n_frames_data
        frame_df.to_csv(s.fs.getsyspath("events/frames.csv"))
        if s.fs.exists("events/events.csv"):
            event_df = pd.read_csv(s.fs.getsyspath("events/events.csv"), index_col=0)
            event_df["stop"].values[-1] = len(frame_df)
            event_df.to_csv(s.fs.getsyspath("events/events.csv"))

    elif diff < 0:
        # truncate event files
        frame_df = frame_df.iloc[slice(0, n_frames_data)]
        assert len(frame_df) == n_frames_data
        frame_df.to_csv(s.fs.getsyspath("events/frames.csv"))
        if s.fs.exists("events/events.csv"):
            event_df = pd.read_csv(s.fs.getsyspath("events/events.csv"), index_col=0)
            if event_df["start"].values[-1] >= len(frame_df):
                raise NotImplementedError
            event_df["stop"].values[-1] = len(frame_df)
            event_df.to_csv(s.fs.getsyspath("events/events.csv"))


def fix_frame_trigger(path: PathLike) -> None:
    import h5py
    from ca_analysis.indexing import argwhere
    with h5py.File(path, "r+") as f:
        DI = f["DI"]
        dset = DI["FrameTrigger"]
        if dset[0] != 0 or dset[-1] != 0:
            arr = dset[:].astype(np.int8).clip(0, 1).squeeze()
            arr[-1] = 0
            diffed = np.ediff1d(arr, to_begin=0)
            ixs = argwhere(diffed)
            signs = diffed[ixs]
            if np.array_equal(signs, [-1, 1, -1]):
                arr[:ixs[1]] = 0
                arr = arr.astype(np.uint32)
                arr[arr > 0] = 16
                arr = arr[:, np.newaxis]
                dset[:] = arr


def motion_correct(
    s: Session,
    src: str = "mov_unprocessed",
    dst: str = "mov",
    force: bool = False,
    **kw,
) -> None:
    from ca_analysis.processing.motion_correction import run_motion_correction
    from matplotlib.figure import Figure

    if not s.reg["mov_unprocessed"].exists():
        return
    if s.reg["mov"].exists() and not force:
        return
    size = s.fs.getsize(s.reg["mov_unprocessed"].path) / 10 ** 9
    n_processes = kw.pop("n_processes", 8)
    single_thread = kw.pop("single_thread", None)
    backend = kw.pop('backend', 'multiprocessing')
    if single_thread is None:
        if size > 40:
            single_thread = True
        else:
            single_thread = False

    logging.info(f"performing motion correction (n_processes={n_processes}, " +
                 f"single_thread={single_thread})"
                 )
    url_in = URL(s.reg[src].resolve(), name="data")
    url_out = URL(s.reg[dst].resolve(), name="data")
    shifts, template = run_motion_correction(
        url_in,
        url_out,
        backend=backend,
        n_processes=n_processes,
        single_thread=single_thread,
        dtype=np.float32,
    )
    s.reg["mc"].mkdir(exist_ok=True)
    np.save(s.reg["mc_shifts"].resolve(), shifts)
    np.save(s.reg["mc_template"].resolve(), template)

    fig = Figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(shifts[:, 0], label="x")
    ax.plot(shifts[:, 1], label="y")
    ax.legend()
    fig.savefig(s.reg["mc_summary"].resolve())


def segment(s: Session, force: bool = False) -> None:
    from ca_analysis.processing.segmentation import run_roi_detection
    if s.attrs["capture"]["modality"] == "camera":
        return
    if not s.reg["mov"].exists():
        return
    if s.reg["segmentation"].exists():
        if not force:
            return
        s.fs.rmtree(s.reg["segmentation"].path)

    url = URL(s.reg["mov"].resolve(), name="data")
    run_roi_detection(s, url)


def make_sample(
    s: Session,
    frames: Union[int, slice] = slice(round(15 * 60 * 5 - 5), round(15 * 60 * 5 - 5) + 1000),
    outfile: PathLike = "sample.mp4",
    sigma: Optional[Union[Number, Tuple[Number, ...]]] = 0.5,
    upsample: Optional[int] = None,
    fps: Optional[Number] = None,
    dpi: Number = 190,
    width: Number = 4,
    frameon: bool = True,
    facecolor: "ColorLike" = "black",
    cmap="inferno",
    qlim=(0.01, 99.99),
    force: bool = False,
    **kw,
):
    import h5py
    from matplotlib.figure import Figure
    from matplotlib.animation import FFMpegWriter
    from ca_analysis.plot import get_smap

    if not s.reg.mov.exists():
        print("No movie found to make sample from. Returning.")
        return
    outfile = (Path(s.fs.getsyspath("")) / outfile).with_suffix(".mp4")
    if outfile.exists() and not force:
        return

    logging.info(f"making sample {outfile}")

    # Figure out fps.
    if fps is None:
        fps = s.attrs["capture"]["frame"]["rate"]
    if upsample and upsample != 1:
        fps = fps * upsample

    # Load movie
    if is_int(frames):
        frames = slice(0, frames)
    with h5py.File(s.reg["mov"].resolve(), "r") as f:
        dset = f["data"]
        mov = dset[frames]

    # Get labels.
    schema = s.events.schema
    frame_info = s.events["frames"]
    event_ids = frame_info["event"][frames]
    labels = [schema.get(event=id).name for id in event_ids]
    labels = simplify_labels(labels, replace={"-": "", "gray": ""})

    # Handle upsampling.
    if upsample and upsample != 1:
        mov = resample_mov(mov, factor=upsample)
        labels = resample_labels(labels, factor=upsample)

    # Handle smoothing
    if sigma is not None:
        mov = gaussian_filter(mov, sigma)

    # Setup normalization + colormapping pipeline.
    smap = get_smap(data=mov, qlim=qlim, cmap=cmap, **kw)

    # Determine figure size, convert to inches.
    ypix, xpix = mov[0].shape
    aspect = ypix / xpix
    figsize = (width, width * aspect)

    # Initialize figure.
    fig = Figure(
        figsize=figsize,
        frameon=frameon,
        facecolor=facecolor,
    )
    ax = fig.add_subplot(1, 1, 1, xmargin=0, ymargin=0)
    ax.set_aspect("equal")
    ax.axis('off')
    fig.tight_layout(pad=0)
    im = ax.imshow(np.zeros_like(mov[0]))

    fontdict = {'size': 16, 'color': 'white'}
    label_loc = [0.05, 0.95]
    label_obj = ax.text(
        label_loc[0],
        label_loc[1],
        ' ',
        fontdict=fontdict,
        transform=ax.transAxes,
        horizontalalignment='left',
        verticalalignment='top',
        usetex=False,
    )

    # ---------------
    # Write frames

    writer = FFMpegWriter(fps=fps)
    with writer.saving(fig, str(outfile), dpi):
        for i in range(mov.shape[0]):
            fm = smap(mov[i])
            fm[:, :, -1] = 1
            im.set_data(fm)
            label_obj.set_text(labels[i])
            writer.grab_frame()


def simplify_labels(
    labels: Sequence[str],
    replace: Optional[Mapping[str, str]] = None,
) -> List[str]:
    out = []
    for i, elt in enumerate(labels):
        if elt.endswith(".front") or elt.endswith(".back"):
            elt = ""
        elt = elt.split("_")[0]
        out.append(elt)

    labels = out
    if replace:
        for i, elt in enumerate(labels):
            if elt in replace:
                labels[i] = replace[elt]
    return labels
