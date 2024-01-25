from main import *
from processing import *
from ca_analysis.processing.utils import fix_frame_trigger
from ca_analysis.processing.utils import pull_session, push_session
import matplotlib.pyplot as plt

import fs as pyfs
from fs.errors import *


def do_import(mouse, date, run="1"):
    path = f"sessions/{mouse}/{date}/{run}"
    pyfs.copy.copy_dir_if(
        get_fs(-1),
        path,
        get_fs(0),
        path,
        'newer',
        preserve_time=True,
    )
    s = open_session(mouse, date, run, fs=0, require=True)
    import_session(s)

    pyfs.copy.copy_dir_if(
        get_fs(0),
        path,
        get_fs(-1),
        path,
        'newer',
        preserve_time=True,
    )
    s.fs.rmtree("thorlabs", missing_ok=True)
    # s.fs.remove("mov.h5")


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
        if 'mask' in f:
            mask = f['mask'][:]
        mov = add_mask(mov, mask, fill_value=0)

    # Get labels.
    schema = s.events.schema
    frame_info = s.events.tables["frame"]
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


# s = open_session("63668-2", "2022-12-06", "1", fs=0)
# make_sample(s, force=True)

# f = h5py.File(s.fs.getsyspath("mov.h5"), "r")
# dset = f['data']
# frame = dset[100]
# chunk = dset[100:200]
# mask = f['mask'][:]
# chunk = add_mask(chunk, mask)
#
# im = np.copy(frame)
# im[mask] = np.nan
# plt.imshow(im)
# plt.show()
# import_session(s)
# do_import("58169-1", "2022-10-31")

# s = open_session("46725-1", "2022-07-06", "1", fs="ssd")
# fix_frame_trigger(s.fs.getsyspath("thorlabs/Episode.h5"))
# import_session(s)
# root = Path.home() / 'ca_analysis_data/sessions'
# dirs = list(root.glob('**/thorlabs'))
# # for d in dirs:
# #     shutil.rmtree(d)
