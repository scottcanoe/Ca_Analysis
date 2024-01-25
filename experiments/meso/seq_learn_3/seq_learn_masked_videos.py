import matplotlib
matplotlib.use('TkAgg')
from matplotlib.animation import FFMpegWriter
from matplotlib.figure import Figure

from experiments.meso.roi_processing import *
from ca_analysis.plot import get_smap

from experiments.meso.main import *
from experiments.meso.seq_learn_3.utils import *


def make_seq_movie(
    s: Session,
    seq_name: str,
    lpad: Optional[int] = None,
    rpad: Optional[int] = None,
    fps: Number = 10,
    qlim: Tuple = (0.01, 99.99),
    cmap: str = 'inferno',
) -> None:

    stim_ca = s.ca_mov.split(seq_name, lpad=lpad, rpad=rpad, concat=True)
    mean_ca = stim_ca.mean('trial')

    stim_ach = s.ach_mov.split(seq_name, lpad=lpad, rpad=rpad, concat=True)
    mean_ach = stim_ach.mean('trial')

    mov_1 = mean_ca
    mov_2 = mean_ach

    fname = f'{s.attrs["mouse"]}_{seq_name}_day{s.attrs["day"]}.mp4'
    outfile = s.fs.getsyspath(fname)

    dpi: Number = 190
    width: Number = 4
    frameon: bool = True
    facecolor: "ColorLike" = "white"

    items = [dict(data=mean_ca, name='Ca'), dict(data=mean_ach, name='ACh')]

    # Figure out fps.
    if fps is None:
        fps = s.attrs["samplerate"]

    # Initialize figure.
    ypix, xpix = items[0]['data'][0].shape
    aspect = ypix / xpix
    figsize = (2 * width, width * aspect)
    fig = Figure(
        figsize=figsize,
        frameon=frameon,
        facecolor=facecolor,
    )
    fig.tight_layout(pad=1)

    # Setup normalization + colormapping pipelinex, axes, and labels
    fontdict = {'size': 16, 'color': 'white'}
    label_loc = [0.05, 0.95]
    for i, dct in enumerate(items):
        data = dct['data']
        dct['smap'] = get_smap(data=data, qlim=qlim, cmap=cmap)
        dct['ax'] = ax = fig.add_subplot(1, 2, i + 1, xmargin=0, ymargin=0)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_title(dct['name'])
        dct['im'] = ax.imshow(np.zeros_like(data[0]))
        dct['label'] = ax.text(
            label_loc[0],
            label_loc[1],
            ' ',
            fontdict=fontdict,
            transform=ax.transAxes,
            horizontalalignment='left',
            verticalalignment='top',
            usetex=False,
        )

    # ------------------------------------------------------------------------------
    # Write frames

    n_frames = items[0]['data'].shape[0]
    writer = FFMpegWriter(fps=fps)
    with writer.saving(fig, str(outfile), dpi):
        for i in range(n_frames):
            for dct in items:
                frame = dct['data'].isel(time=i)
                cdata = dct['smap'](frame)
                cdata[:, :, -1] = 1
                dct['im'].set_data(cdata)
                dct['label'].set_text((frame.coords['event'].item()))
            writer.grab_frame()





"""
--------------------------------------------------------------------------------
"""

def make_masked_seq_movie(
    s: Session,
    sequence: str,
    lpad: Optional[int] = 4,
    masks = ('LH', 'RH'),
    cmap: str = 'inferno',
    qlim: Tuple = (2.5, 97.5),
    fps: Optional[Number] = None,
) -> Path:

    # Do some prep
    fps = s.attrs["samplerate"] if fps is None else fps
    channels = get_channels(s)
    n_channels = len(channels)
    if is_str(masks):
        masks = [masks]

    # Load ach data
    mean_ach = load_sequence_splits(s, 'ABCD', 'ach', lpad=lpad).mean('trial')
    items = [dict(data=mean_ach, name='ACh')]

    # Load cortex mask. True is where good values are.
    mask = np.zeros_like(mean_ach.isel(time=0), dtype=bool)
    with h5py.File(s.fs.getsyspath('masks.h5'), 'r') as f:
        for name in masks:
            group = f[name]
            pixels = group['pixels'][:]
            y_i = pixels[:, 0]
            x_i = pixels[:, 1]
            mask[y_i, x_i] = True
    Y, X = np.where(mask)
    mask_bad = ~mask
    Y_bad, X_bad = np.where(mask_bad)

    # Optionally load calcium data
    if 'ca' in channels:
        mean_ca = load_sequence_splits(s, 'ABCD', 'ca', lpad=lpad).mean('trial')
        items.insert(0, dict(data=mean_ca, name='Ca'))

    # Initialize figure.
    ypix, xpix = items[0]['data'][0].shape
    aspect = ypix / xpix
    width = n_channels * 4
    height = 4 * aspect
    figsize = (width, height)
    fig = Figure(figsize=figsize, frameon=True, facecolor='white')
    fig.tight_layout(pad=1)

    # Setup normalization + colormapping pipelinex, axes, and labels
    for i, dct in enumerate(items):
        data = dct['data']
        values = data.data[:, Y, X].flatten()
        vlim = np.percentile(values, qlim)
        dct['smap'] = get_smap(data=data, vlim=vlim, cmap=cmap)
        dct['ax'] = ax = fig.add_subplot(1, n_channels, i + 1, xmargin=0, ymargin=0)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_title(dct['name'])
        dct['im'] = ax.imshow(np.zeros_like(data[0]))
        dct['label'] = ax.text(
            0.05,
            0.95,
            ' ',
            fontdict={'size': 16, 'color': 'white'},
            transform=ax.transAxes,
            horizontalalignment='left',
            verticalalignment='top',
            usetex=False,
        )

    # ------------------------------------------------------------------------------
    # Write frames
    fname = f'{s.attrs["mouse"]}_day{s.attrs["day"]}_{sequence}.mp4'
    savepath = Path(s.fs.getsyspath(fname))

    n_frames = items[0]['data'].shape[0]
    writer = FFMpegWriter(fps=fps)
    with writer.saving(fig, str(savepath), 190):
        for i in range(n_frames):
            for dct in items:
                frame = dct['data'].isel(time=i)
                cdata = dct['smap'](frame)
                cdata[:, :, -1] = 1
                cdata[Y_bad, X_bad, :] = (0, 0, 0, 1)
                dct['im'].set_data(cdata)
                dct['label'].set_text((frame.coords['event'].item()))
            writer.grab_frame()

    return savepath


sessions = [
    open_session('M172', '2023-05-15', '1', fs=0),
    open_session('M172', '2023-05-19', '1', fs=0),
    open_session('M173', '2023-05-15', '1', fs=0),
    open_session('M173', '2023-05-19', '1', fs=0),
    open_session('M174', '2023-05-15', '1', fs=0),
    open_session('M174', '2023-05-19', '1', fs=0),
]


# s = open_session('M173', '2023-05-15', '1', fs=0)

# for s in sessions:
#     day = s.attrs['day']
#     sequences = ['ABCD', 'ABBD', 'ACBD'] if day == 5 else ['ABCD']
#     for seq in sequences:
#         savepath = make_masked_seq_movie(s, seq)
#         video_dir = Path.home() / 'Desktop/videos_masked'
#         shutil.copyfile(savepath, video_dir / savepath.name)

