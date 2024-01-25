import logging
from numbers import Number
import shutil
from typing import (
    List,
    Optional,
    Tuple,
    Union,
)

import h5py
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

xr.set_options(keep_attrs=True)

from ca_analysis import *
from ca_analysis.io.thorlabs import *
from ca_analysis.processing.alignment import SignalParser
from ca_analysis.plot import *
from main import *
from ca_analysis.plot.color import ColorLike
logger = logging.getLogger("ca_analysis.pipelines.FOC_3.1")

def import_session(
        *args: List,
        fs: Optional[str] = None,
) -> None:
    """
    First step of the pipeline. The sole purpose of this function is to handle
    raw measurements and put them into a standardized form. This includes placing
    the reformatted data according to a file scheme. 

    Note that this is not meant to do anything that alters the underlying data in
    any significant way, such as filtering. It's only meant to standardize the
    format of the raw data.

    """
    if isinstance(args[0], Session):
        s = args[0]
    else:
        s = open_session(*args, fs=fs)

    logger.info('Importing session {} on fs {}.'.format(s, s.fs))
    s.reg.analysis.syspath.mkdir(exist_ok=True)
    s.reg.scratch.syspath.mkdir(exist_ok=True)

    # Collect attributes from metadata files.
    init_attrs(s)

    # Process event data.
    process_events(s)

    # Convert raw movie to 'mov_unprocessed.h5'.
    convert_raw(s)
    if s.reg.mov_unprocessed.exists():
        raw_path = s.reg.thor_raw.syspath
        if raw_path.exists():
            raw_path.unlink()

    # Perform motion correction.
    motion_correct(s)

    # Perform segmentation.
    segment(s)

    # Make sample mov.
    make_sample(s)

    return s


def init_attrs(s: Session) -> None:
    attrs = {}

    # - thorlabs attrs
    if s.reg.thor_md.exists():
        md = ThorImageMetadataSource(s.reg.thor_md.syspath).read()
        attrs["capture"] = md

    # - schema
    attrs["schema"] = schema.name

    # - etc.
    relparts = s.fs._sub_dir.split("/")
    mouse, date, exp = relparts[-3:]
    attrs["mouse"] = mouse
    attrs["date"] = date
    attrs["exp"] = exp

    s.attrs.merge(attrs)
    s.attrs.save()


def process_events(
        s: Session,
        force: bool = False,
) -> None:

    from ca_analysis.io.thorlabs import ThorSyncSource

    evdir = s.registry["events"]
    if s.fs.exists(evdir.path):
        if force:
            s.fs.removetree(evdir.path)
        else:
            return

    logger.info('Processing events')
    s.fs.makedir(evdir.path)
    src = schema.path
    dst = s.registry["schema"].syspath
    shutil.copyfile(src, dst)

    # Run the stimulus alignment tool on the sync_data.
    sync_data = ThorSyncSource(s.registry["thor_sync"]).read()
    name_map = {
        "AnalogEvents": "event_value",
        "FrameOut": "frame_out",
        "FrameTrigger": "frame_trigger",
        "Strobe": "strobe",

    }
    for key, val in name_map.items():
        sync_data[val] = sync_data[key]
        del sync_data[key]
    parser = SignalParser(schema, sync_data)
    parser.strobe_anchor = "start"
    parser.frame_anchor = "mid"
    parser.parse()
    frames = parser.tables["frame"]
    del frames["start"]
    del frames["stop"]
    del frames["mid"]
    frames.to_csv(s.registry["frames_table"])
    parser.tables["event"].to_csv(s.registry["events_table"])


def convert_raw(s: Session, force: bool = False) -> None:
    from ca_analysis.io.utils import thorimage_raw_to_h5
    if any(s.fs.exists(s.registry[p].path) for p in ("mov_unprocessed", "mov"))\
            and not force:
        return
    logger.info("Converting raw imaging data")
    s.registry["scratch"].syspath.mkdir(exist_ok=True)
    src, dst = s.registry["thor_raw"], s.registry["mov_unprocessed"]
    src = src.syspath
    dst = URL(dst.syspath, key="data")
    thorimage_raw_to_h5(src, dst)


def motion_correct(s: Session, force: bool = False) -> None:
    from ca_analysis.processing.motion_correction import (
        run_motion_correction,
    )

    if s.registry["mov_unprocessed"].exists():
        if s.registry["mov"].exists() and not force:
            return
        size = s.fs.getsize(s.registry["mov_unprocessed"].path) / 10 ** 9
        if size > 40:
            n_processes = 8
            single_thread = True
        else:
            n_processes = 8
            single_thread = False

        logger.info(f"performing motion correction (n_processes={n_processes}, " + \
                    f"single_thread={single_thread})")
        url_in = URL(s.registry["mov_unprocessed"].syspath, key="data")
        url_out = URL(s.registry["mov"].syspath, key="data")
        shifts, template = run_motion_correction(
            url_in,
            url_out,
            backend="sandbox",
            n_processes=n_processes,
            single_thread=single_thread,
            dtype=np.float32,
        )
        s.registry["mc"].syspath.mkdir(exist_ok=True)
        np.save(s.registry["mc_shifts"].syspath, shifts)
        np.save(s.registry["mc_template"].syspath, template)

        fig, ax = plt.subplots()
        ax.plot(shifts[:, 0], label="x")
        ax.plot(shifts[:, 1], label="y")
        ax.legend()
        fig.savefig(s.registry["mc_summary"].syspath)


def segment(s: Session, force: bool = False) -> None:
    from ca_analysis.processing.segmentation import run_roi_detection
    if s.attrs["capture"]["modality"] == "camera":
        return
    if not s.registry["mov"].exists():
        return
    if s.registry["segmentation"].exists():
        if not force:
            return
        s.fs.removetree(s.registry["segmentation"].syspath)

    mov = URL(s.registry["mov"].syspath, key="data")
    run_roi_detection(s, mov)


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
        facecolor: ColorLike = "black",
        cmap="inferno",
        qlim=(0.01, 99.99),
        force: bool = False,
        **kw,
):
    from matplotlib.figure import Figure
    from matplotlib.animation import FFMpegWriter

    if not s.reg.mov.exists():
        return
    outfile = (Path(s.fs.getsyspath("")) / outfile).with_suffix(".mp4")
    if outfile.exists() and not force:
        return
    logger.info(f"making sample {outfile}")

    # Figure out fps.
    if fps is None:
        try:
            fps = s.attrs["capture"]["fps"]
        except:
            fps = s.attrs["capture"]["frame"]["rate"]
    if upsample and upsample != 1:
        fps = fps * upsample

    # Load movie
    if is_int(frames):
        frames = slice(0, frames)
    with h5py.File(s.fs / "mov.h5", "r") as f:
        dset = f["data"]
        mov = dset[frames]

    # Get labels.
    schema = s.events.schema
    frame_info = s.events.tables["frame"]
    event_ids = frame_info["event_id"]
    event_ids = event_ids[frames]
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

    plt.close(fig)
