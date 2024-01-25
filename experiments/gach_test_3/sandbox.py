import intake
from intake.source.discovery import drivers, load_plugins_from_module
import weakref
from time import perf_counter as clock
from typing import Mapping

import dask.array as da
import h5py
from matplotlib.figure import Figure
from matplotlib.animation import FFMpegWriter
import napari
import numpy as np

from main import *
from processing import *
from ca_analysis.plot import get_smap
from ca_analysis.processing.utils import *



def annotate_onsets(
    ax: "Axes",
    labels: ArrayLike,
    skip_first: bool = False,
    last: Optional[str] = None,
    vline: bool = True,
    color: Union[ArrayLike, str] = "gray",
    ls: str = "--",
    alpha: Number = 1,
) -> None:

    # Add onset indicators.
    xticks = []
    if not skip_first:
        xticks.append(0)
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            xticks.append(i - 0.5)
    xticklabels = [r'$\Delta$'] * len(xticks)
    if last is not None:
        xticklabels[-1] = last
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    # add vertical lines
    if vline:
        for x in xticks:
            ax.axvline(x, color=color, ls=ls, alpha=alpha)


def dFF(
    F: ArrayLike,
    Fneu: Optional[ArrayLike] = None,
    neu_coeff: Number = 0.7,
    window_size: int = 500,
    qmin: Number = 20,
) -> np.ndarray:
    """
    Performs dF/F process on one or more traces. This implementation uses
    a smoothed rolling min to establish a baseline activity within some
    reasonable window (the default window size is about 7-15 seconds wide when
    imaging at 15-30 Hz.). Smoothing is applied to the baseline prior to
    subtracting in order to compensate for the sudden jumps that often occur
    when computing rolling mins/maxs/medians on this data.

    Baseline is the pseudo-rolling minimum quantile. Baseline is subtracted, and
    then used as a scaling factor:

    dFF(x) = (x - baseline) / baseline


    If axis is None, then...

        ndim = 1 -> axis = 0   single trace
        ndim = 2 -> axis = 1   trace pool, and traces are in rows.


    Parameters
    ----------

    data : array-like
        1D, 2D, or 3D array.

    window_size : int
        Width of window used for computation.



    Returns
    -------

    result : ndarray
        Has the same shape and dtype as input.

    """

    from scipy.ndimage import gaussian_filter1d

    using_xarray = isinstance(F, xr.DataArray)
    if using_xarray:
        in_dims = F.dims
        F = F.transpose('time', ...)

    def dFF_one(arr: np.ndarray) -> np.ndarray:
        baseline = np.zeros_like(arr)
        for ix in range(0, len(arr), window_size):
            chunk = arr[ix:ix + window_size]
            baseline[ix:ix + window_size] = np.percentile(chunk, qmin)
        baseline = gaussian_filter1d(baseline, window_size)
        arr = (arr - baseline) / baseline
        return arr

    data = np.asarray(F)

    # Handle neuropil subtraction.
    if Fneu is not None:
        data = data - neu_coeff * np.asarray(Fneu)

    if data.ndim == 1:
        out = dFF_one(data)
    if data.ndim > 2:
        out = np.vstack([dFF_one(row) for row in data])

    if using_xarray:
        out = xr.DataArray(out, dims=F.dims, coords=F.coords)
        out = out.transpose(*in_dims)
    return out

    raise ValueError('Input must be 1D or 2D (with time in zeroth axis).')


def do_import(mouse, date, run):

    path = f'sessions/{mouse}/{date}/{run}'
    remote = get_fs(-1)
    local = get_fs(0)
    pyfs.copy.copy_dir_if(remote, path, local, path, "newer", preserve_time=True)
    s = open_session(mouse, date, run, fs=0)
    import_session(s)
    s.fs.move('scratch/mov_motion_corrected.h5', 'mov.h5')
    s.fs.remove('scratch/mov_unprocessed.h5')
    pyfs.copy.copy_dir_if(local, path, remote, path, "newer", preserve_time=True)
    s.fs.rmtree("thorlabs", missing_ok=True)
    s.fs.remove("mov.h5")


to_import = [
    ('58470-1', '2022-12-20', '1'),
    ('58470-2', '2022-12-20', '1'),
    ('58470-3', '2022-12-20', '1'),
    ('63668-1', '2022-12-20', '1'),
    ('63668-2', '2022-12-20', '1'),
    ('63668-3', '2022-12-20', '1'),
    ('63668-5', '2022-12-20', '1'),
]

def plot_traces():
    to_import = [
        ('58470-1', '2022-12-20', '1'),
        # ('58470-2', '2022-12-20', '1'),
        # ('58470-3', '2022-12-20', '1'),
        # ('63668-1', '2022-12-20', '1'),
        # ('63668-2', '2022-12-20', '1'),
        # ('63668-3', '2022-12-20', '1'),
        # ('63668-5', '2022-12-20', '1'),
    ]
    lpad = 2.5 * units.sec
    rpad = 2.5 * units.sec
    PLOTDIR = Path.home() / "plots/gach_test_3/mean_traces"
    for tup in to_import:
        s = open_session(tup[0], tup[1], tup[2], fs=-1)
        s.G.data = dFF(s.G.data)
        stim = s.G.split('stim', lpad=lpad, rpad=rpad, concat=True)
        arr = stim.mean('trial')
        fig, ax = plt.subplots()
        ax.plot(arr)
        annotate_onsets(ax, arr.coords['event'], skip_first=True)
        title = s.mouse
        ax.set_title(title)
        plt.show()
        savepath = PLOTDIR / f"{s.mouse}.pdf"
        fig.savefig(savepath)


# s = open_session('58470-1', '2022-12-20', '1', fs=0)

# with h5py.File(s.fs.getsyspath('mov.h5'), 'r') as f:
#     dset = f['data']
#     n_frames = dset.shape[0]
#     G = np.zeros(n_frames)
#     for i in range(n_frames):
#         G[i] = np.mean(dset[i, :370, :200])
#     np.save(s.fs.getsyspath('G.npy'), G)
# lpad = 2.5 * units.sec
# rpad = 2.5 * units.sec
# PLOTDIR = Path.home() / "plots/gach_test_3/mean_traces"
# s = open_session('58470-1', '2022-12-20', '1', fs=0)
# s.G.data = dFF(s.G.data)
# stim = s.G.split('stim', lpad=lpad, rpad=rpad, concat=True)
# arr = stim.mean('trial')
# fig, ax = plt.subplots()
# ax.plot(arr)
# annotate_onsets(ax, arr.coords['event'], skip_first=True)
# title = s.mouse
# ax.set_title(title)
# plt.show()
# savepath = PLOTDIR / f"{s.mouse}.pdf"
# fig.savefig(savepath)


def make_summary_images(f: h5py.File) -> None:

    logger.info('creating summary images')
    with h5py.File(s.fs.getsyspath('mov.h5'), 'r+') as f:
        dset = f['data']
        arr = da.from_array(dset, chunks=(1, -1, -1))
        mean = arr.mean(axis=0).compute()
        max = arr.max(axis=0).compute()
        std = arr.std(axis=0).compute()
        if 'images' not in f:
            grp = f.create_group('images')
        else:
            grp = f['images']
        grp.create_dataset('mean', data=mean)
        grp.create_dataset('max', data=max)
        grp.create_dataset('std', data=std)


def open_viewer(s: Session) -> napari.Viewer:

    with h5py.File(s.fs.getsyspath("mov.h5"), "r+") as f:
        if 'images' not in f:
            make_summary_images(f)
        background = f["images"]['mean'][:]

    viewer = napari.view_image(background)
    bg_layer = viewer.layers[0]
    bg_layer.name = "background"
    win = viewer.window
    win.add_plugin_dock_widget('napari-roi')
    layer = napari.layers.Shapes(
        edge_color="red",
        face_color="transparent",
    )
    layer.name = "ROIs"
    viewer.add_layer(layer)
    viewer.window.session = s
    return viewer


def load_roi_data(path) -> Mapping:
    dct = {'name': [], 'data': [], 'shape_type': [], 'y': [], 'x': []}
    with h5py.File(path, "r") as f:
        dset_names = list(f.keys())
        for i, name in enumerate(dset_names):
            group = f[name]
            dct['name'].append(name)
            dct['data'].append(group['data'][:])
            dct['shape_type'].append(str(group.attrs['shape_type']))
            pixels = group['pixels']
            dct['y'].append(pixels[0, :])
            dct['x'].append(pixels[1, :])

    return dct


def extract_fluorescence(s: Session, fname: str = 'rois.h5') -> None:
    mov_path = s.fs.getsyspath('mov.h5')
    roi_path = Path(s.fs.getsyspath(fname)).with_suffix('.h5')
    with h5py.File(mov_path, 'r') as f_mov:
        mov_dset = f_mov['data']
        with h5py.File(roi_path, 'r+') as f_rois:
            for key in f_rois.keys():
                grp = f_rois[key]
                pixels = grp['pixels']
                y, x = pixels[:, 0], pixels[:, 1]
                F = np.zeros(mov_dset.shape[0])
                for i in range(len(F)):
                    frame = mov_dset[i]
                    F[i] = np.mean(frame[y, x])
                dF = dFF(F)

                if 'F' in grp:
                    del grp['F']
                grp.create_dataset('F', data=F)
                if 'dFF' in grp:
                    del grp['dFF']
                grp.create_dataset('dFF', data=dF)


class ManualROISegmentation(SessionData):

    def __init__(self, session: session, key: str, filename: str = "rois.h5"):
        super().__init__(session)
        self.key = key
        self.filename = filename

    def _prepare(self) -> None:
        roi_path = s.fs.getsyspath(self.filename)
        roi_path = Path(roi_path).with_suffix('.h5')
        with h5py.File(roi_path, 'r') as f:
            names, traces = [], []
            for key, group in f.items():
                names.append(key)
                traces.append(group[self.key][:])
        data = np.stack(traces).T
        dims = ('time', 'roi')
        coords = {'roi': names}
        self.data = xr.DataArray(data, dims=dims, coords=coords)


from matplotlib_scalebar.scalebar import ScaleBar


s = open_session('58470-2', '2022-12-20', '1', fs=-1)


# viewer = open_viewer(s)
# extract_fluorescence(s, 'rois.h5')
# extract_fluorescence(s, 'vasculature.h5')


#
# s.F = ManualROISegmentation(s, 'F')
# s.dFF = ManualROISegmentation(s, 'dFF')
# s.vasc = ManualROISegmentation(s, 'dFF', 'vasculature')
#
# lpad = rpad = 2.5 * units.sec
# stim = s.dFF.split('stim', lpad=lpad, rpad=rpad, concat=True)
# arr = stim.mean('trial')
# fig, ax = plt.subplots()
# y = arr.mean('roi') if 'roi' in arr.dims else arr
# ax.plot(y)
# annotate_onsets(ax, arr.coords['event'], skip_first=True)
# title = s.mouse
# ax.set_title(title)
# plt.show()
#
# stim = s.vasc.split('stim', lpad=lpad, rpad=rpad, concat=True)
# arr = stim.mean('trial')
# fig, ax = plt.subplots()
# y = arr.mean('roi') if 'roi' in arr.dims else arr
# ax.plot(y)
# annotate_onsets(ax, arr.coords['event'], skip_first=True)
# title = s.mouse + ' vasculature'
# ax.set_title(title)
# plt.show()

# savepath = PLOTDIR / f"{s.mouse}.pdf"
# fig.savefig(savepath)