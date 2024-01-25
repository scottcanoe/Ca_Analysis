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


# def process_events(
#     s: Session,
#     force: bool = False,
# ) -> None:
#
#     logger = get_logger()
#
#     evdir = Path(s.reg["events"])
#     if evdir.exists():
#         if force:
#             shutil.rmtree(evdir)
#         else:
#             return
#
#     logger.info('Processing events')
#     evdir.mkdir()
#     src = schema.url
#     dst = s.registry["schema"].path
#     shutil.copyfile(src, dst)
#
#     # Run the stimulus alignment tool on the sync_data.
#     sync_data = ThorSyncSource(s.registry["thor_sync"]).read()
#     parser = SignalParser(schema, sync_data)
#     parser.strobe_anchor = "stop"
#     parser.frame_anchor = "mid"
#     parser.parse()
#     frames_df = parser.tables["frame"]
#     del frames_df["start"]
#     del frames_df["stop"]
#     del frames_df["mid"]
#     frames_df.to_csv(s.registry["frames_table"])
#     events_df = parser.tables["event"]
#     events_df.to_csv(s.registry["events_table"])

#
# def convert_raw(s: Session, force: bool = False) -> None:
#
#     from ca_analysis.io.thorlabs import thorimage_raw_to_h5
#     from ca_analysis.io.h5 import H5URL
#     logger = get_logger()
#
#     if not s.reg["thor_raw"].path.exists():
#         return
#
#     names = ("mov", "mov_unprocessed", "mov_mc")
#     if any(s.reg[n].exists() for n in names):
#         if not force:
#             return
#
#     logger.info("Converting raw imaging data")
#     Path(s.reg["scratch"]).mkdir(exist_ok=True)
#     src = URL(s.reg["thor_raw"]))
#     dst = URL(s.reg["mov_unprocessed"]), name="data")
#     thorimage_raw_to_h5(src, dst)
#

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


def run_luminance_filter(s: Session):

    # Perform luminance correction.
    if s.reg["mov"].exists():
        return

    if not s.reg["mov_mc"].exists():
        return

    if s.reg["mask_npy"].exists():
        mask = np.load(s.reg["mask_npy"].resolve())
    else:
        mask = create_mask(s)

    # Load data, optionally apply mask.
    logger.info("Applying luminance filter")
    with h5py.File(s.reg["mov_mc"].resolve(), "r") as f:
        data = f["data"][:]

    if mask is not None:
        data = add_mask(data, mask, fill_value=0)

    data = median_filter(data, size=(2, 2, 1))
    mov = luminance_filter(data)
    with h5py.File(s.reg["mov"].resolve(), "w") as f:
        if np.ma.is_masked(data):
            f.create_dataset("data", data=mov.data)
            f.create_dataset("mask", data=mov.mask[0])
        else:
            f.create_dataset("data", data=mov)


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
    fps_capture = s.attrs['capture']['frame']['rate']
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



def find_outlier_pixels(data, tolerance=3, worry_about_edges=True):
    # This function finds the hot or dead pixels in a 2D dataset.
    # tolerance is the number of standard deviations used to cutoff the hot pixels
    # If you want to ignore the edges and greatly speed up the code, then set
    # worry_about_edges to False.
    #
    # The function returns a list of hot pixels and also an image with with hot pixels removed

    from scipy.ndimage import median_filter
    blurred = median_filter(data, size=2)
    difference = data - blurred
    threshold = 10 * np.std(difference)

    # find the hot pixels, but ignore the edges
    hot_pixels = np.nonzero((np.abs(difference[1:-1, 1:-1]) > threshold))
    hot_pixels = np.array(hot_pixels) + 1  # because we ignored the first row and first column

    fixed_image = np.copy(data)  # This is the image with the hot pixels removed
    for y, x in zip(hot_pixels[0], hot_pixels[1]):
        fixed_image[y, x] = blurred[y, x]

    if worry_about_edges == True:
        height, width = np.shape(data)

        # Now get the pixels on the edges (but not the corners)###

        # left and right sides
        for index in range(1, height - 1):
            # left side:
            med = np.median(data[index - 1:index + 2, 0:2])
            diff = np.abs(data[index, 0] - med)
            if diff > threshold:
                hot_pixels = np.hstack((hot_pixels, [[index], [0]]))
                fixed_image[index, 0] = med

            # right side:
            med = np.median(data[index - 1:index + 2, -2:])
            diff = np.abs(data[index, -1] - med)
            if diff > threshold:
                hot_pixels = np.hstack((hot_pixels, [[index], [width - 1]]))
                fixed_image[index, -1] = med

        # Then the top and bottom
        for index in range(1, width - 1):
            # bottom:
            med = np.median(data[0:2, index - 1:index + 2])
            diff = np.abs(data[0, index] - med)
            if diff > threshold:
                hot_pixels = np.hstack((hot_pixels, [[0], [index]]))
                fixed_image[0, index] = med

            # top:
            med = np.median(data[-2:, index - 1:index + 2])
            diff = np.abs(data[-1, index] - med)
            if diff > threshold:
                hot_pixels = np.hstack((hot_pixels, [[height - 1], [index]]))
                fixed_image[-1, index] = med

        ###Then the corners###

        # bottom left
        med = np.median(data[0:2, 0:2])
        diff = np.abs(data[0, 0] - med)
        if diff > threshold:
            hot_pixels = np.hstack((hot_pixels, [[0], [0]]))
            fixed_image[0, 0] = med

        # bottom right
        med = np.median(data[0:2, -2:])
        diff = np.abs(data[0, -1] - med)
        if diff > threshold:
            hot_pixels = np.hstack((hot_pixels, [[0], [width - 1]]))
            fixed_image[0, -1] = med

        # top left
        med = np.median(data[-2:, 0:2])
        diff = np.abs(data[-1, 0] - med)
        if diff > threshold:
            hot_pixels = np.hstack((hot_pixels, [[height - 1], [0]]))
            fixed_image[-1, 0] = med

        # top right
        med = np.median(data[-2:, -2:])
        diff = np.abs(data[-1, -1] - med)
        if diff > threshold:
            hot_pixels = np.hstack((hot_pixels, [[height - 1], [width - 1]]))
            fixed_image[-1, -1] = med

    return hot_pixels, fixed_image


