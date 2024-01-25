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

    if not s.reg["mov"].exists():
        # Convert raw movie to 'mov_unprocessed.h5'.
        thorimage_raw_to_h5(s)

        # Perform motion correction.
        if not s.reg["mov_mc"].exists():
            motion_correct(s, src="mov_unprocessed", dst="mov_mc")
            s.fs.rm(s.reg["mov_unprocessed"].path, missing_ok=True)

        # Perform luminance correction.
        run_luminance_filter(s)

    # Process event data.
    process_events(s)

    # Make sample mov.
    make_sample(s)

    # Make retinotopy mov.
    make_retinotopy_video(s)

    return s


def create_mask(s: Session) -> np.ndarray:
    import imageio
    import subprocess as sp
    from skimage.color import gray2rgb

    # Find a movie we can use for the background image.
    background = None
    for key in ("mov", "mov_mc"):
        p = s.reg[key].resolve()
        if p.exists():
            with h5py.File(p, "r") as f:
                background = f["data"][:50]
                background = np.mean(background, axis=0)
            break

    if background is None:
        raise FileNotFoundError("no movie found")

    # Save the background as a jpeg.
    background = (250 * (background / background.max())).astype(np.uint8)
    rgba = gray2rgb(background).astype(np.uint8)
    imageio.imsave(s.reg["background_jpeg"].resolve(), rgba)
    np.save(s.reg["background_npy"].resolve(), background)

    # Launch krita
    msg = """
    Mask Creation
    ------------
    - 1. Open {}/'background.jpeg'
    - 2. Use the Elliptical Selection Tool to select the ROI (probably everything
         that is window)
    - 3. Right click on the selection, and select 'Cut selection to new layer'.
    - 4. Make only the new layer visible.
    - 5. Save as '{}/mask.tif' and quit krita.

    """.format(str(s.reg["scratch"].resolve()), str(s.reg["scratch"].resolve()))
    print(msg, flush=True)

    tif_path = s.reg["mask_tif"].resolve()
    if tif_path.exists():
        tif_path.unlink()
    proc = sp.run(["krita", "--nosplash"], stdout=sp.PIPE, stderr=sp.STDOUT)
    if not proc:
        msg = "There was an error with krita: \n"
        raise RuntimeError(msg)

    # Load mask.
    mat = imageio.imread(tif_path)
    mask = np.zeros((mat.shape[0], mat.shape[1]), dtype=bool)
    mask[mat[:, :, -1] == 0] = True
    np.save(s.reg["mask_npy"].resolve(), mask)

    for key in ("mov", "mov_mc"):
        p = s.reg[key].resolve()
        if p.exists() and p.suffix.lower() in {".h5", ".hdf", ".hdf5"}:
            with h5py.File(p, "r+") as f:
                if "mask" in f.keys():
                    del f["mask"]
                f.create_dataset("mask", data=mask)

    return mask


def run_luminance_filter(s):

    from time import perf_counter as clock
    import dask.array as da

    t_start = clock()

    print(f'processing movie data for {s}')

    f = f_raw = h5py.File(s.fs.getsyspath('scratch/mov_motion_corrected.h5'), 'r')
    dset = f['data']
    raw = da.from_array(dset)

    print('median filtering')
    t = clock()
    f = f_med = h5py.File(s.fs.getsyspath('scratch/mov_median_filtered.h5'), 'w')
    dset = f.create_dataset('data', shape=raw.shape, dtype=np.float32)
    for i in range(raw.shape[0]):
        frame = raw[i].compute()
        frame = median_filter(frame, (3, 3))
        dset[i] = frame
    f_raw.close()
    f_med.close()
    print(f'... finished median filtering in {clock() - t} secs')

    create_mask(s)
    mask = np.load(s.fs.getsyspath('scratch/mask.npy'))

    print('computing means')
    t = clock()
    f = f_med = h5py.File(s.fs.getsyspath('scratch/mov_median_filtered.h5'), 'r')
    dset = f["data"]
    arr = da.from_array(dset)
    mean_frame = arr.mean(axis=0).compute()
    np.save(s.fs.getsyspath("scratch/S.npy"), mean_frame)

    mov = np.ma.masked_array(dset[:], mask=np.broadcast_to(mask, dset.shape))
    T_f = np.array([frame.compressed().mean() for frame in mov])
    np.save(s.fs.getsyspath('scratch/T_f.npy'), T_f)
    print(f'... computed means in {clock() - t} secs')

    print('standardizing frames')
    G = T_f / T_f.mean()
    t = clock()
    f = f_std = h5py.File(s.fs.getsyspath('mov.h5'), 'w')
    dst = f.create_dataset('data', shape=mov.shape, dtype=np.float32)
    for i in range(mov.shape[0]):
        frame = mov[i]
        frame = frame / mean_frame - G[i]
        frame = np.nan_to_num(frame, posinf=0.0, neginf=0.0)
        dst[i] = frame
    f.create_dataset('mask', data=mask)
    f.create_dataset('G', data=T_f / T_f.mean())
    f.create_dataset('mean_frame', data=mean_frame)
    f_std.close()
    print(f'... standardized frames in {clock() - t} secs')

    print(f'finished processing movie data in {clock() - t_start} secs')


def make_retinotopy_video(
    s: Session,
    outfile: PathLike = "retinotopy.mp4",
    cmap="inferno",
    qlim=(0.5, 99.75),
    kernel=None,
    fps=30,
    dpi=220,
    width=8,
    frameon=True,
    facecolor=None,
) -> None:
    """
    Generates a four-panel video, one for each sweeping direction.
    
    """

    from matplotlib.figure import Figure
    from matplotlib.animation import FFMpegWriter
    from ca_analysis.plot import get_smap

    # Load the data.
    infile = s.fs.getsyspath("mov.h5")
    outfile = Path(s.fs.getsyspath(outfile))
    with h5py.File(infile, "r") as f:
        mov = f["data"][:]
        if "mask" in f.keys():
            mask = f["mask"][:]
            mov = add_mask(mov, mask, fill_value=0)

    # Create plot items.
    import pandas as pd
    schema = s.events.schema
    fps_capture = s.attrs['capture']['frame']['rate'] * 5
    # table = s.events.tables["events"]
    table = s.events.tables["event"]
    items = []
    for name in ["right", "up", "left", "down"]:

        obj = SimpleNamespace(name=name)
        obj.event = schema.get(event=schema.get(event=obj.name))

        # - find block to extract
        df = table[table["event"] == obj.event.id]
        starts = df["start"].values
        lengths = round(obj.event.duration * fps_capture)
        stops = starts + lengths
        n_blocks = len(starts)

        # - extract and reduce
        shape = (n_blocks, lengths, mov.shape[1], mov.shape[2])
        stack = np.zeros(shape, dtype=mov.dtype)
        for i, (a, b) in enumerate(zip(starts, stops)):
            try:
                stack[i] = mov[a:b]
            except ValueError:
                pass
        obj.data = np.mean(stack, axis=0)
        if np.ma.is_masked(mov):
            obj.data = add_mask(obj.data, mov.mask[0], fill_value=mov.fill_value)

        # - smooth/filter
        if kernel:
            obj.data = gaussian_filter(obj.data, kernel)

        # - colormapping
        obj.smap = get_smap(data=obj.data, cmap=cmap, qlim=qlim)

        items.append(obj)

    # Setup figure and axes.    
    ypix, xpix = mov[0].shape
    aspect = ypix / xpix
    figsize = (width, width * aspect)
    fig = Figure(
        figsize=figsize,
        frameon=frameon,
        facecolor=facecolor,
    )
    for i, obj in enumerate(items):
        ax = obj.ax = fig.add_subplot(2, 2, i + 1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])

        ax.set_title(obj.name)
        obj.im = obj.ax.imshow(np.zeros_like(mov[0]))
    fig.tight_layout(pad=0)

    # Save to file.
    n_frames = max([obj.data.shape[0] for obj in items])
    writer = FFMpegWriter(fps=fps)
    with writer.saving(fig, str(outfile), dpi):
        for i in range(n_frames):
            for obj in items:
                frame_num = min(i, obj.data.shape[0] - 1)
                cdata = obj.smap(obj.data[frame_num])
                cdata[:, :, -1] = 1
                obj.im.set_data(cdata)
            writer.grab_frame()

