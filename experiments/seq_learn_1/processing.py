import logging
from numbers import Number
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
from typing import (
    Callable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import h5py
from matplotlib.animation import FFMpegWriter
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import xarray as xr

from ca_analysis import *
from ca_analysis.environment import *
from ca_analysis.grouping import *
from ca_analysis.plot import *
from ca_analysis.traces import *


schema = EventSchema(get_fs(0).getsyspath("event_schemas/seq_learn_1.yaml"))
logger = logging.getLogger("ca_analysis.experiments.seq_learn_1")
logging.basicConfig(level=logging.INFO)

PATHS = {
    "attrs" : "attrs.yaml",
    "thor_md" : "thorlabs/Experiment.xml",
    "thor_raw" : "thorlabs/Image_0001_0001.raw",
    "thor_sync" : "thorlabs/Episode001.h5",
    "scratch" : "scratch",
    "unproc" : "scratch/mov_unprocessed.h5",
    "shifts" : "scratch/shifts.npy",
    "mov" : "mov.h5",
    "suite2p" : "suite2p",
    "analysis": "analysis",
    "sample" : "sample.mp4",
}

def import_session(
    *args: List,
    day: Optional[int] = None,
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
        
    logger.info('Importing seq_learn_1 session {} on fs {}.'.format(s, s.fs))
    
    s.fs.requiredir("analysis")
    
    # Collect attributes from metadata files.
    init_attrs(s, day=day)
    
    # Process event data.
    process_events(s)

    # Convert raw movie to 'mov_unprocessed.h5'.
    convert_raw(s)
    
    # Perform motion correction.
    motion_correct(s)

    # Perform segmentation.
    segment(s)

    # Make sample mov.
    make_sample(s)        
    
    return s

def init_attrs(s: Session, day: Optional[int] = None) -> None:
    
    attrs = {}
    
    # - thorlabs attrs
    md_path = "thorlabs/Experiment.xml"
    if s.fs.exists(md_path):
        md = read_thorimage_metadata(s.fs.getsyspath(md_path))
        attrs["capture"] = md
    
    # - schema
    attrs["schema"] = SCHEMA.name
    if day is not None:
        attrs["day"] = day
        
    # - etc.
    relparts = s.fs._sub_dir.split("/")
    mouse, date, exp = relparts[-3:]
    attrs["mouse"] = mouse
    attrs["date"] = date
    attrs["run"] = run
    
    s.attrs.merge(attrs)
    s.attrs.save()
    
def process_events(s: Session, force: bool = False) -> None:
    from ca_analysis.processing.utils import process_events
    if s.fs.exists("events") and not force:
        return
    
    logger.info('Processing events')
    process_events(s, )
    evdir = s.fs.makedir("events")
    src = SCHEMA.path
    dst = evdir.getsyspath("schema.yaml")
    shutil.copyfile(src, dst)

    # Run the stimulus alignment tool on the sync_data.
    sync_data = read_thorsync_data(s.fs / PATHS["thor_sync"])
    parser = EventParser(SCHEMA, sync_data)
    parser.strobe_anchor = "start"
    parser.frame_anchor = "mid"
    parser.parse()
    frames = parser.tables["frames"]
    del frames["start"]
    del frames["stop"]
    del frames["mid"]
    frames.to_csv(evdir.root / "frames.csv")
    parser.tables["events"].to_csv(evdir.root / "events.csv")
    parser.tables["sequences"].to_csv(evdir.root / "sequences.csv")

def convert_raw(s: Session, force: bool = False) -> None:
    if any (s.fs.exists(PATHS[p]) for p in ("unproc", "mov")) and not force:
        return
    logger.info("Converting raw imaging data")
    s.fs.requiredir("scratch")
    src = s.fs.root / PATHS["thor_raw"]
    dst = s.fs.root / PATHS["unproc"]
    thorimage_raw_to_h5(src, dst)

def motion_correct(s: Session, force: bool = False) -> None:
    if s.fs.exists(PATHS["unproc"]):
        if s.fs.exists(PATHS["mov"]) and not force:
            return
        size = s.fs.getsize(PATHS["unproc"]) / 10**9
        if size > 40:
            n_processes = 8
            single_thread = True
        else:
            n_processes = 8
            single_thread = False
        
        logger.info(f"performing motion correction (n_processes={n_processes}, " + \
                    f"single_thread={single_thread})")
        shifts = run_motion_correction(
            s.fs.getsyspath(PATHS["unproc"]),
            s.fs.getsyspath(PATHS["mov"]),
            backend="sandbox",
            n_processes=n_processes,
            single_thread=single_thread,
            dtype=np.float32,
        )
        s.fs.requiredir("scratch")
        np.save(s.fs.getsyspath(PATHS["shifts"]), shifts)
    
def segment(s: Session, force: bool = False) -> None:

    if s.attrs["capture"]["modality"] == "camera":
        return
    if not s.fs.exists(PATHS["mov"]):
        return
    if s.fs.exists(PATHS["suite2p"]):
        if not force:
            return
        s.fs.removetree(PATHS["suite2p"])
        
    run_segmentation(s)

"""
-------------------------------------------------------------------------------------
- plotting utils
"""



def get_grouper_specs(
    s: Session,
    reverse_DCBA: bool = True,    
    truncate_novT: bool = True,
    fpad: bool = True,
    bpad: bool = True,    
    ) -> Mapping[str, List]:
          
    if reverse_DCBA:
        specs = {
            "ABCD": [FPad("A"), "A", "B", "C", "D", BPad("D")],
            "DCBA": [FPad("D_r"), "A_r", "B_r", "C_r", "D_r", BPad("A_r")],
            "ABBD": [FPad("A_g"), "A_g", "B_g", "C_g", "D_g", BPad("D_g")],
            "ABCD_novT": [FPad("A_novT"), "A_novT", "B_novT", "C_novT", "D_novT", BPad("D_novT")],
        }    
    else:
        specs = {
            "ABCD": [FPad("A"), "A", "B", "C", "D", BPad("D")],
            "DCBA": [FPad("D_r"), "D_r", "C_r", "B_r", "A_r", BPad("A_r")],
            "ABBD": [FPad("A_g"), "A_g", "B_g", "C_g", "D_g", BPad("D_g")],
            "ABCD_novT": [FPad("A_novT"), "A_novT", "B_novT", "C_novT", "D_novT", BPad("D_novT")],
        }    

    schema = s.events.schema
    for ev_name in ["A", "B", "C", "D"]:        
        schema[ev_name + "_novT"].duration = schema[ev_name].duration
                    
    if not fpad:
        for key, val in specs.items():
            val.pop(0)
    
    if not bpad:
        for key, val in specs.items():
            val.pop(-1)
    
    if s.attrs["day"] != 5:
        for name in ["DCBA", "ABBD", "ABCD_novT"]:
            del specs[name]
    
    return specs




"""
-------------------------------------------------------------------------------------
- plots
"""

def make_sample(
    s: Session,
    frames: Union[int, slice] = 1000,
    outfile: PathLike = "sample.mp4",    
    sigma: Optional[Union[Number, Tuple[Number, ...]]] = 0.5,
    upsample: Optional[int] = None,
    fps: Optional[Number] = None,
    dpi: Number = 190,
    width: Number = 3,
    frameon: bool = True,
    facecolor: ColorLike = "black",
    cmap = "inferno",
    qlim = (0.01, 99.99),
    force: bool = False,
    **kw,
    ):

    if not s.fs.exists(PATHS["mov"]):
        return
    if s.fs.exists(PATHS["sample"]):
        if force:
            s.fs.remove(PATHS["sample"])
        else:
            return
                
    from matplotlib.figure import Figure
    from matplotlib.animation import FFMpegWriter

    # Handle path.
    outfile = Path(s.fs.getsyspath(PATHS["sample"]))
    outfile = outfile.with_suffix(".mp4")
    logger.info(f"making sample {outfile}")

    # Figure out fps.
    if fps is None:
        fps = s.attrs["capture"]["fps"]
    if upsample and upsample != 1:
        fps = fps * upsample

    # Load movie
    if _isint(frames):
        frames = slice(0, frames)
    with h5py.File(s.fs / "mov.h5", "r") as f:
        dset = f["data"]
        mov = dset[frames]

    # Get labels.
    schema = s.events.schema    
    frame_info = s.events.tables["frames"]
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

    #---------------
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
    
def make_diffmov(
    s: Session,
    outfile="diffmov.mp4",
    cmap = "coolwarm",
    qlim = (0.5, 99.75),
    sigma = 1,
    upsample = 4,
    fps = 30,
    dpi = 220,    
    frameon = True,
    facecolor = None,
    inches_per_plot = 3,
    reverse_DCBA: bool = True,
    ):

    from matplotlib.figure import Figure
    from matplotlib.animation import FFMpegWriter


    outfile = s.fs / outfile
    if not outfile.suffix:
        outfile = outfile.with_suffix(".mp4")

    if reverse_DCBA:
        SPECS = {
            "ABCD": [FPad("A"), "A", "B", "C", "D", BPad("D")],
            "DCBA": [FPad("D_r"), "A_r", "B_r", "C_r", "D_r", BPad("A_r")],
            "ABBD": [FPad("A_g"), "A_g", "B_g", "C_g", "D_g", BPad("D_g")],
            "ABCD_novT": [FPad("A_novT"), "A_novT", "B_novT", "C_novT", "D_novT", BPad("D_novT")],
        }
    else:
        SPECS = {
            "ABCD": [FPad("A"), "A", "B", "C", "D", BPad("D")],
            "DCBA": [FPad("D_r"), "D_r", "C_r", "B_r", "A_r", BPad("A_r")],
            "ABBD": [FPad("A_g"), "A_g", "B_g", "C_g", "D_g", BPad("D_g")],
            "ABCD_novT": [FPad("A_novT"), "A_novT", "B_novT", "C_novT", "D_novT", BPad("D_novT")],
        }

    # Change duration of novel elements.
    schema = s.events.schema
    for ev_name in ("A", "B", "C", "D"):
        base = schema[ev_name]
        novel = schema[ev_name + "_novT"]
        novel.duration = base.duration

    # Group data, and average over trials.
    f = h5py.File(s.root / "mov.h5", "r")
    mov = f["data"]
    grouper = Grouper(s.events, fps=s.attrs["acquisition"]["fps"])
    groups = {}
    for spec_name, spec in SPECS.items():
        lst = grouper.group(mov, spec)
        for i, obj in enumerate(lst):
            lst[i] = obj.mean("trial")
        groups[spec_name] = lst
    f.close()

    # Concatenate, label movies.
    for name, lst in groups.items():
        arr = xr.concat(lst, dim="time")
        labels = []
        for elt in lst:
            labels.extend([elt.name] * len(elt))

        arr.attrs["labels"] = simplify_labels(labels)
        groups[name] = arr


    # Upsample movies.
    if upsample and upsample != 1:
        for name, g in groups.items():
            data = g.data
            labels = g.attrs["labels"]
            n_frames_out = len(data) * upsample
            data_out = scipy.signal.resample(data, n_frames_out, axis=0)
            labels_out = resample_labels(labels, factor=upsample)
            arr = xr.DataArray(data_out, name=name, dims=("time", "ypix", "xpix"))
            arr.attrs["labels"] = labels_out
            groups[name] = arr

    # Smooth movies?

    ABCD = groups["ABCD"]
    DCBA = groups["DCBA"]
    ABBD = groups["ABBD"]
    ABCD_novT = groups["ABCD_novT"]

    items = []

    obj = SimpleNamespace(name="DCBA-ABCD", data=DCBA - ABCD)
    obj.labels = DCBA.attrs["labels"]
    items.append(obj)

    obj = SimpleNamespace(name="ABBD-ABCD", data=ABBD - ABCD)
    obj.labels = ABBD.attrs["labels"]
    items.append(obj)
    obj = SimpleNamespace(name="ABCD_novT-ABCD", data=ABCD_novT - ABCD)
    obj.labels = ABCD_novT.attrs["labels"]
    items.append(obj)


    n_plots = len(items)
    width = inches_per_plot * n_plots
    height = inches_per_plot
    fig = Figure(figsize=(width, height))
    fig.tight_layout(pad=0)

    for i in range(n_plots):
        obj = items[i]
        obj.ax = ax = fig.add_subplot(1, n_plots, i+1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_title(obj.name)
        obj.im = ax.imshow(obj.data[0])
        obj.to_rgba = get_smap(data=obj.data, qlim=qlim, cmap=cmap)
        if sigma:
            obj.data = gaussian_filter(obj.data, sigma)

        fontdict = {'size': 16, 'color': 'black'}
        label_loc = [0.05, 0.95]        
        obj.lbl = ax.text(label_loc[0], label_loc[1], ' ',
                          fontdict=fontdict,
                          transform=ax.transAxes,
                          horizontalalignment='left',
                          verticalalignment='top',
                          usetex=False)



    n_frames = max([obj.data.shape[0] for obj in items])
    writer = FFMpegWriter(fps=fps)
    with writer.saving(fig, str(outfile), dpi):
        for i in range(n_frames):
            for obj in items:
                if i >= obj.data.shape[0]:
                    continue                
                cdata = obj.to_rgba(obj.data[i])
                cdata[:, :, -1] = 1
                obj.im.set_data(cdata)
                obj.lbl.set_text(obj.labels[i])
                writer.grab_frame()

def make_meanmov(
    s: Session,
    outfile="analysis/meanmov.mp4",
    cmap = "inferno",
    qlim = (2.5, 99.5),
    sigma = 1,
    upsample = 4,
    fps = 30,
    dpi = 220,
    frameon = True,
    facecolor = None,
    inches_per_plot = 3,
    reverse_DCBA: bool = True,
    ):

    t0 = time.time()

    from matplotlib.figure import Figure
    from matplotlib.animation import FFMpegWriter
    
    schema = s.events.schema
    day = s.attrs["day"]
    fps = s.attrs["capture"]["fps"]
    outfile = Path(s.fs.getsyspath(outfile)).with_suffix(".mp4")
    
    
    if day == 1:
        SPECS = {"ABCD": [FPad("A"), "A", "B", "C", "D", BPad("D")]}
        
    else:
        if reverse_DCBA:
            SPECS = {
                "ABCD": [FPad("A"), "A", "B", "C", "D", BPad("D")],
                "DCBA": [FPad("D_r"), "A_r", "B_r", "C_r", "D_r", BPad("A_r")],
                "ABBD": [FPad("A_g"), "A_g", "B_g", "C_g", "D_g", BPad("D_g")],
                "ABCD_novT": [FPad("A_novT"), "A_novT", "B_novT", "C_novT", "D_novT", BPad("D_novT")],
            }
        else:
            SPECS = {
                "ABCD": [FPad("A"), "A", "B", "C", "D", BPad("D")],
                "DCBA": [FPad("D_r"), "D_r", "C_r", "B_r", "A_r", BPad("A_r")],
                "ABBD": [FPad("A_g"), "A_g", "B_g", "C_g", "D_g", BPad("D_g")],
                "ABCD_novT": [FPad("A_novT"), "A_novT", "B_novT", "C_novT", "D_novT", BPad("D_novT")],
            }
            
            # Change duration of novel elements.
            for ev_name in ("A", "B", "C", "D"):
                base = schema[ev_name]
                novel = schema[ev_name + "_novT"]
                novel.duration = base.duration

    # Group data, and average over trials.
    f = h5py.File(s.fs.getsyspath(PATHS["mov"]), "r")
    mov = f["data"]
    grouper = Grouper(s.events)
    items = {} # seq_name -> SimpleNamespace
    print(f't1: {time.time() - t0}')
    for name, spec in SPECS.items():
        # Group movies by element type, get trial average, and concatenate them.
        g = SimpleNamespace(name=name)        
        lst = grouper.group(mov, spec)        
        for i, obj in enumerate(lst):
            lst[i] = obj.mean("trial")
        g.arr = xr.concat(lst, dim="dim_1")
        
        # Get time-series of labels.
        labels = []
        for elt in lst:
            labels.extend([elt.name] * len(elt))
        g.labels = simplify_labels(labels)  
        
        items[name] = g
    
    f.close()
    print(f't2: {time.time() - t0}')

    # Optionally upsample.
    if upsample and upsample != 1:
        for name, g in items.items():
            n_frames_out = len(g.arr) * upsample
            arr = scipy.signal.resample(g.arr, n_frames_out, axis=0)
            g.arr = xr.DataArray(arr, name=name, dims=("time", "ypix", "xpix"))
            g.labels = resample_labels(g.labels, factor=upsample)
    print(f't2: {time.time() - t0}')
    
    n_plots = len(items)        
    width = inches_per_plot * n_plots
    height = inches_per_plot
    fig = Figure(figsize=(width, height))
    fig.tight_layout(pad=0)

    count = 0
    for name, g in items.items():
        count += 1
        g.ax = ax = fig.add_subplot(1, n_plots, count)
        remove_ticks(g.ax)
        g.ax.set_title(g.name)
        g.im = g.ax.imshow(g.arr[0])
        g.smap = get_smap(data=g.arr, qlim=qlim, cmap=cmap)
        if sigma:
            g.arr = gaussian_filter(g.arr, sigma)

        fontdict = {'size': 16, 'color': 'white'}
        label_loc = [0.05, 0.95]        
        g.lbl = ax.text(label_loc[0], label_loc[1], ' ',
                          fontdict=fontdict,
                          transform=ax.transAxes,
                          horizontalalignment='left',
                          verticalalignment='top',
                          usetex=False)

    print(f't3: {time.time() - t0}')
    n_frames = max([g.arr.shape[0] for g in items.values()])
    writer = FFMpegWriter(fps=fps)
    with writer.saving(fig, str(outfile), dpi):
        for i in range(n_frames):
            for g in items.values():
                if i >= g.arr.shape[0]:
                    continue                
                cdata = g.smap(g.arr[i])
                cdata[:, :, -1] = 1
                g.im.set_data(cdata)
                g.lbl.set_text(g.labels[i])
                writer.grab_frame()
    print(f't4: {time.time() - t0}')

def make_roi_meanmov(
    s: Session,
    outfile="analysis/roi_meanmov.mp4",
    cmap = "inferno",
    qlim = (2.5, 99.5),
    sigma = 1,
    upsample = 4,
    fps = 30,
    dpi = 220,
    frameon = True,
    facecolor = None,
    inches_per_plot = 3,
    reverse_DCBA: bool = True,
    ):
        
    outfile = Path(s.fs.getsyspath(outfile)).with_suffix(".mp4")
    
    fps = s.attrs["capture"]["fps"]
    cells = s.cells
    iscell = s.segmentation["iscell"]
    spikes = s.segmentation["spikes"]
    spikes = spikes[:, iscell]
    frame_shape = s.attrs["capture"]["frame_shape"]
    
    # Group data, and average over trials.
    SPECS = get_grouper_specs(s, reverse_DCBA=reverse_DCBA, truncate_novT=True)
    grouper = Grouper(s.events)
    items = {} # seq_name -> SimpleNamespace    
    for name, spec in SPECS.items():
        # Group movies by element type, get trial average, and concatenate them.
        g = SimpleNamespace(name=name)
        lst = grouper.group(spikes, spec)
        for i, obj in enumerate(lst):
            lst[i] = obj.mean("trial")
        g.spikes = xr.concat(lst, dim="time")
        
        # Get time-series of labels.
        labels = []
        for elt in lst:
            labels.extend([elt.name] * len(elt))
        g.labels = simplify_labels(labels)
        items[name] = g    

    # Optionally upsample and smooth spike data.
    if upsample and upsample != 1:
        for g in items.values():
            n_frames_out = len(g.spikes) * upsample
            spikes_out = scipy.signal.resample(g.spikes, n_frames_out, axis=0)
            g.spikes = xr.DataArray(spikes_out, dims=("time", "roi"))
            g.labels = resample_labels(g.labels, factor=upsample)
            
    if sigma:
        from scipy.ndimage import gaussian_filter1d
        g.spikes = gaussian_filter1d(g.spikes, sigma, axis=0)        
    
    # Create images.
    for g in items.values():
        g.arr = np.zeros([len(g.spikes), frame_shape[0], frame_shape[1]])
        for i, c in enumerate(cells):
            mask = c.mask
            for t in range(len(g.spikes)):
                g.arr[t, mask.y, mask.x] = g.spikes[t, i]
                
    # Initialize figure and axes.
    n_plots = len(items)
    width = inches_per_plot * n_plots
    height = inches_per_plot
    fig = Figure(figsize=(width, height))
    fig.tight_layout(pad=0)
    count = 0
    for name, g in items.items():
        count += 1
        g.ax = ax = fig.add_subplot(1, n_plots, count)
        remove_ticks(g.ax)
        g.ax.set_title(g.name)
        g.im = g.ax.imshow(g.arr[0])
        g.smap = get_smap(data=g.spikes, qlim=qlim, cmap=cmap)

        fontdict = {'size': 16, 'color': 'white'}
        label_loc = [0.05, 0.95]
        g.lbl = ax.text(label_loc[0], label_loc[1], ' ',
                          fontdict=fontdict,
                          transform=ax.transAxes,
                          horizontalalignment='left',
                          verticalalignment='top',
                          usetex=False)
    
    # Render.
    n_frames = max([g.arr.shape[0] for g in items.values()])
    writer = FFMpegWriter(fps=fps)
    with writer.saving(fig, str(outfile), dpi):
        for i in range(n_frames):
            for g in items.values():
                if i >= g.arr.shape[0]:
                    continue                
                cdata = g.smap(g.arr[i])
                cdata[:, :, -1] = 1
                g.im.set_data(cdata)
                g.lbl.set_text(g.labels[i])
                writer.grab_frame()


def make_heatmaps(
    s: Session,
    outfile: PathLike = "analysis/heatmaps.pdf",
    cmap: str = "inferno",
    qlim: Tuple[Number, Number] = (2.5, 97.5),
    sigma: Optional[Number] = None,
    upsample: Optional[int] = None,
    inches_per_plot: Number = 3,
    reverse_DCBA: bool = True,
    rank_by: Optional[Union[str, Sequence[str], Callable]] = "mean",
    descending: bool = True,
    ):
        
    outfile = Path(s.fs.getsyspath(outfile))
        
    iscell = s.segmentation["iscell"]
    spikes = s.segmentation["spikes"]
    spikes = spikes[:, iscell]    
    
    # Group data, and average over trials.
    SPECS = get_grouper_specs(s, reverse_DCBA=reverse_DCBA, truncate_novT=True)
    grouper = Grouper(s.events)
    items = {} # seq_name -> SimpleNamespace
    
    for name, spec in SPECS.items():
        # Group movies by element type, get trial average, and concatenate them.
        g = SimpleNamespace(name=name)
        lst = grouper.group_traces(spikes, spec)
        for i, obj in enumerate(lst):
            lst[i] = obj.mean("trial")
        g.spikes = xr.concat(lst, dim="time")
        
        # Ensure time is in zeroth axis.
        g.spikes = g.spikes.transpose("time", "roi")
        
        # Get time-series of labels.
        labels = []
        for elt in lst:
            labels.extend([elt.name] * len(elt))
        g.labels = simplify_labels(labels)
        g.spikes.coords['label'] = xr.DataArray(g.labels, dims=("time",))
        items[name] = g

    # Optionally upsample spike data.
    if upsample and upsample != 1:
        for g in items.values():
            n_frames_out = len(g.spikes) * upsample
            data = scipy.signal.resample(g.spikes, n_frames_out, axis=0)
            g.spikes = xr.DataArray(data, dims=("time", "roi"))
            g.labels = resample_labels(g.labels, factor=upsample)
            g.spikes.coords['label'] = xr.DataArray(g.labels, dims=("time",))
    
    # Optionally smooth spike data.
    if sigma:
        from scipy.ndimage import gaussian_filter1d
        data = gaussian_filter1d(g.spikes, sigma, axis=0)
        g.spikes = xr.DataArray(data, dims=("time", "roi"), coords=g.spikes.coords)
        
    # Rank traces.
    if rank_by:
        g = list(items.values())[0]
        roi_order = argranked(g.spikes, rank_by, descending=descending)
        for g in items.values():
            g.spikes = g.spikes.isel(roi=roi_order)

    
    # Put time in first axis to prepare for plotting.
    for g in items.values():
        g.spikes = g.spikes.transpose("roi", "time")
    
    # Initialize figure and axes.
    nrows, ncols = len(items), 2
    width, height = 6, inches_per_plot * nrows        
    fig = Figure(figsize=(width, height))
    fig.tight_layout(pad=0)
    grid = fig.add_gridspec(nrows, ncols)
    
    # Plot ABCD first.        
    g = ABCD = items.pop("ABCD")
    labels = g.labels
    g.ax = fig.add_subplot(grid[0, :])
    smap = get_smap(data=g.spikes, qlim=qlim, cmap=cmap)
    #cdata = smap(g.spikes)
    g.im = g.ax.imshow(
        g.spikes,
        cmap=smap.cmap,
        norm=smap.norm,
        aspect="auto",
        interpolation="hanning",
        
    )
    annotate_heatmap_axes(g.ax, g.name, labels)
    axes = fig.get_axes()
    g.cbar = fig.colorbar(g.im, ax=axes[-1])
    g.cbar.ax.tick_params(labelsize=8)
    
    for i, g in enumerate(items.values()):
        
        # Draw regular plot.
        g.ax1 = fig.add_subplot(grid[i+1, 0])
        smap = get_smap(data=g.spikes, qlim=qlim, cmap=cmap)
        cdata = smap(g.spikes)
        g.ax1.imshow(cdata, aspect="auto", interpolation="hanning")
        annotate_heatmap_axes(g.ax1, g.name, labels)
        
        # Draw difference plot.
        g.ax2 = fig.add_subplot(grid[i+1, 1])
        data = g.spikes - ABCD.spikes
        vmin, vmax = np.percentile(data, [2.5, 97.5])
        i
        if vmin < 0 and vmax <= 0:
            vmax = -vmin
        elif vmin < 0 and vmax >= 0:
            vmin = -vmax
        else:
            #vmin > 0, vmax > 0
            vmin = -vmax
                        
        #g.im = g.ax2.imshow(cdata, aspect="auto", interpolation="hanning")
        g.norm = get_norm(vlim=[vmin, vmax])
        g.cmap = get_cmap("coolwarm")
        g.im = g.ax2.imshow(data, aspect="auto", cmap=g.cmap, norm=g.norm)
        #g.im.set_data(data)
        
        annotate_heatmap_axes(g.ax2, g.name + " - ABCD", labels)
        g.ax2.set_yticks([])
    
        axes = fig.get_axes()
        g.cbar = fig.colorbar(g.im, ax=axes[-1])
        g.cbar.ax.tick_params(labelsize=8)

    
    if nrows == 1:
        pass
    else:
        fig.subplots_adjust(hspace=0.3, wspace=0.05, top=0.95, bottom=0.05)
            
    fig.savefig(outfile)

     
def push(
    s: Session,
    path: PathLike, 
    src: str, 
    dst: str,
    must_exist: bool = False,
    if_newer: bool = False,
    ) -> None:
    
    relpath = f"sessions/{s.attrs['mouse']}/{s.attrs['date']}/{s.attrs['exp']}"
    src_fs = get_fs(src).opendir(relpath)
    src_path = Path(src_fs.getsyspath(path))
    
    dst_fs = get_fs(dst).opendir(relpath)
    dst_path = Path(dst_fs.getsyspath(path))
    
    if not src_path.exists():
        if must_exist:
            raise FileNotFoundError(src_path)
        return
    
    if dst_path.exists():
        if if_newer:            
            raise NotImplementedError
        if dst_path.is_dir():
            shutil.rmtree(dst_path)
        else:
            dst_path.unlink()
    
    if src_path.is_dir():
        shutil.copytree(src_path, dst_path)
    else:
        shutil.copy2(src_path, dst_path)

def get_sessions(
    day: Optional[int] = None,
    fs: Optional[str] = None,
    ) -> List[Session]:
    
    lst = [
        open_session("610259-1", "2021-03-22", fs=fs),
        open_session("610259-1", "2021-03-26", fs=fs),
        
        open_session("610259-2", "2021-03-22", fs=fs),
        open_session("610259-2", "2021-03-26", fs=fs),                
                
        open_session("611486-1", "2021-04-28", fs=fs),
        open_session("611486-1", "2021-05-02", fs=fs),
                
        open_session("611507-1", "2021-05-16", fs=fs),
        open_session("611507-1", "2021-05-20", fs=fs),
                
        open_session("611521-1", "2021-05-16", fs=fs),
        open_session("611521-1", "2021-05-20", fs=fs),
                
        open_session("A-2", "2020-07-02", fs=fs),
        open_session("A-2", "2020-07-08", fs=fs),
                
        open_session("A-3", "2020-07-02", fs=fs),
        open_session("A-3", "2020-07-08", fs=fs),
                
        open_session("B-1", "2020-07-02", fs=fs),
        open_session("B-1", "2020-07-08", fs=fs),
    ]
    if day is not None:
        lst = [s for s in lst if s.attrs["day"] == day]
    return lst
    
    
def finish_import(fs="hdd"):
    
    sessions = [
        open_session("610259-2", "2021-03-26", fs=fs),                        
        open_session("610259-3", "2021-03-27", fs=fs),    
    ]
    for s in sessions:
        import_session(s)
                
        if s.fs.exists(PATHS["mov"]):
            if s.fs.exists(PATHS["unproc"]):
                s.fs.remove(PATHS["unproc"])
            if s.fs.exists(PATHS["thor_raw"]):
                s.fs.remove(PATHS["thor_raw"]) 
    
    

def annotate_heatmap_axes(ax: "Axes", title: str, labels: Sequence[str]) -> None:
    
    # Add title
    ax.set_title(title)
    
    # Add onset indicators.
    xticks = []        
    for i, elt in enumerate(labels[1:]):
        if labels[i] != labels[i-1]:
            xticks.append(i)
    xticklabels = [r'$\Delta$'] * len(xticks)
    xticklabels[-1] = "-"    
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    for x in xticks:
        ax.axvline(x, color="white", ls="--")
        
    
if __name__ == "__main__":

    logging.getLogger("ca_analysis").setLevel(logging.INFO)

    mfs = get_fs()
    ssd = get_fs("ssd")
    #hdd = get_fs("hdd")
    #ext1 = get_fs("ext1")
            
    #day1 = get_sessions(day=1, fs="hdd")
    #day5 = get_sessions(day=5, fs="hdd")
    sessions = day1 + day5
    
    s = open_session("611486-1", "2021-04-28", fs="ssd")
    
    #cells = s.cells
    #spikes = cells.spikes
    #for s in sessions:
        #if s.fs.exists("mov.h5"):
            #mouse, date, exp = s.attrs["mouse"], s.attrs["date"], s.attrs["exp"]    
            #s2 = open_session(mouse, date, exp, fs="gavornik-nas")
            #if s2.fs.exists("mov.h5"):
                #s.fs.remove("mov.h5")
                
    #s = open_session("611486-1", "2021-04-28", fs="hdd")
    #s = open_session("611486-1", "2021-05-02", fs="hdd")
    #for s in sessions:    
        #make_heatmaps(s)
    #make_roi_meanmov(s)
    
    #outfile = "analysis/roi_meanmov2.mp4"
    #cmap = "inferno"
    #qlim = (2.5, 99.5)
    #sigma = None
    #upsample = 4
    #fps = 30
    #dpi = 220
    #frameon = True
    #facecolor = None
    #inches_per_plot = 3
    #reverse_DCBA: bool = True
    
        
    #outfile = Path(s.fs.getsyspath(outfile)).with_suffix(".mp4")
    
    #fps = s.attrs["capture"]["fps"]
    #cells = s.cells
    #iscell = s.roidata["iscell"]
    #spikes = s.roidata["spikes"]
    #spikes = spikes[:, iscell]
    #frame_shape = s.attrs["capture"]["frame_shape"]
    
    ## Group data, and average over trials.
    #SPECS = get_grouper_specs(s, reverse_DCBA=reverse_DCBA, truncate_novT=True)
    #grouper = Grouper(s.events)
    #items = {} # seq_name -> SimpleNamespace
    #for name, spec in SPECS.items():
        ## Group movies by element type, get trial average, and concatenate them.
        #g = SimpleNamespace(name=name)
        #lst = grouper.group(spikes, spec)
        #for i, obj in enumerate(lst):
            #lst[i] = obj.mean("trial")
        #g.spikes = xr.concat(lst, dim="time")
        
        ## Get time-series of labels.
        #labels = []
        #for elt in lst:
            #labels.extend([elt.name] * len(elt))
        #g.labels = simplify_labels(labels)
        #items[name] = g    

    ## Optionally upsample and smooth spike data.
    #if upsample and upsample != 1:
        #for g in items.values():
            #n_frames_out = len(g.spikes) * upsample
            #spikes_out = scipy.signal.resample(g.spikes, n_frames_out, axis=0)
            #g.spikes = xr.DataArray(spikes_out, dims=("time", "roi"))
            #g.labels = resample_labels(g.labels, factor=upsample)
            
    #if sigma:
        #from scipy.ndimage import gaussian_filter1d
        #g.spikes = gaussian_filter1d(g.spikes, sigma, axis=0)
    
    ## Create images.
    #for g in items.values():
        #g.arr = np.zeros([len(g.spikes), frame_shape[0], frame_shape[1]])
        #for i, c in enumerate(cells):
            #mask = c.mask
            #for t in range(len(g.spikes)):
                #g.arr[t, mask.y, mask.x] = g.spikes[t, i]
                
    ## Initialize figure and axes.
    #import matplotlib.colors as mpc
    #bg = s.roidata["meanimg"]
    #vmin, vmax = np.percentile(bg, (1, 99.75))
    #bg = mpc.Normalize(vmin=vmin, vmax=vmax)(bg)
    #template = gray_to_rgba(bg)
    #template = template / 0.8
    #template[..., -1] = 1
    
    #n_plots = len(items)
    #width = inches_per_plot * n_plots
    #height = inches_per_plot
    #fig = Figure(figsize=(width, height))
    #fig.tight_layout(pad=0)
    #count = 0
    #for name, g in items.items():
        #count += 1
        #g.ax = ax = fig.add_subplot(1, n_plots, count)
        #remove_ticks(g.ax)
        #g.ax.set_title(g.name)
        #g.im = g.ax.imshow(g.arr[0])
        #g.smap = get_smap(data=g.spikes, qlim=qlim, cmap=cmap)

        #fontdict = {'size': 16, 'color': 'white'}
        #label_loc = [0.05, 0.95]
        #g.lbl = ax.text(label_loc[0], label_loc[1], ' ',
                        #fontdict=fontdict,
                        #transform=ax.transAxes,
                        #horizontalalignment='left',
                        #verticalalignment='top',
                        #usetex=False)
    
    ## Prepare time series.
    #for g in items.values():        
        #for i, c in enumerate(s.cells):            
            #ts = g.spikes[:, i].data
            ##ts = ts / np.percentile(ts, 99.999)
            #ts = ts / ts.max()
            #c.spikes = ts
    
    ## Render
    #n_frames = max([g.arr.shape[0] for g in items.values()])
    #writer = FFMpegWriter(fps=fps)
    #with writer.saving(fig, str(outfile), dpi):
        #for i in range(n_frames):
            #for g in items.values():
                #if i >= g.arr.shape[0]:
                    #continue
                #im = template.copy()                
                                
                #for j, c in enumerate(s.cells):
                    #spk = g.spikes[i, j]
                    #spk_norm = g.smap.norm(spk)
                    #spk_color = g.smap.cmap(spk_norm)
                    #alpha = c.spikes[i]
                    ##spk = g.spikes[]
                    ##color = g.smap.cmap(g.smap.norm(spk))
                    #overlay_roi(
                    #im,
                    #c,
                    #spk_color,
                    #alpha=alpha,
                    #out=im,
                #)
                    
                #g.im.set_data(im)
                #g.lbl.set_text(g.labels[i])
                #writer.grab_frame()    
