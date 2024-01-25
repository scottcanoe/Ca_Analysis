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


def make_mov(
    s: Session,
    indata: Union[ArrayLike, PathLike],
    outfile: PathLike,
    diffmode = False,
    cmap=None,
    width=8,
    dpi=220,
    qlim=None,
    frameon=True,
    facecolor=None,
):
    if is_pathlike(indata):
        if not os.path.isabs(indata):
            indata = s.fs.getsyspath(indata)
        f = h5py.File(indata, 'r')
        dset = f['data']
        indata = da.from_array(dset)
    else:
        f = dset = None
        indata = da.asarray(indata)

    if diffmode:
        cmap = 'coolwarm' if cmap is None else cmap
        qlim = (50, 97.5) if qlim is None else qlim
    else:
        cmap = 'inferno' if cmap is None else cmap
        qlim = (2.5, 97.5) if qlim is None else qlim

    schema = s.events.schema
    fps = s.events.get_fps()
    ev_df = s.events.tables["event"]
    mask = np.load(s.fs.getsyspath('scratch/mask.npy'))
    duration = schema.get(event=1).duration
    n_frames_stim = round(fps * duration)
    n_frames_pre = round(5 * fps)
    n_frames_post = round(5 * fps)
    n_frames_chunk = n_frames_pre + n_frames_stim + n_frames_post

    data = {}
    for ev_id in range(1, 6):
        sub_df = ev_df[ev_df.event == ev_id]
        info = SimpleNamespace(event=schema.get(event=ev_id), difference=False)
        info.name = info.event.name
        stack = []
        for onset in ev_df[ev_df.event == info.event.id].start:
            start = onset - n_frames_pre
            if start < 0:
                continue
            stop = onset + n_frames_stim + n_frames_post
            if stop >= indata.shape[0]:
                continue
            chunk = indata[start:stop]
            if hasattr(chunk, 'compute'):
                chunk = chunk.compute()
            stack.append(chunk)
        info.stack = np.stack(stack)
        info.mov = np.mean(info.stack, axis=0)
        info.mov = add_mask(info.mov, mask, fill_value=0.0)
        info.mov = gaussian_filter(info.mov, (1, 1, 1))
        data[ev_id] = info

    if diffmode:
        full = data[5]
        full_mov = full.mov
        for ev_id in range(1, 5):
            info = data[ev_id]
            info.mov = full_mov - info.mov
            info.difference = True

    frame_shape = data[1].mov.shape[1:]
    ypix, xpix = frame_shape
    aspect = ypix / xpix
    figsize = (width, width * aspect)
    fig = Figure(
        figsize=figsize,
        frameon=frameon,
        facecolor=facecolor,
    )
    for ev_id in range(1, 5):
        ax = fig.add_subplot(2, 2, ev_id)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        im = ax.imshow(np.zeros(frame_shape))
        label_loc = [0.05, 0.95]
        label_obj = ax.text(
            label_loc[0],
            label_loc[1],
            ' ',
            fontdict={'size': 16, 'color': 'white'},
            transform=ax.transAxes,
            horizontalalignment='left',
            verticalalignment='top',
            usetex=False,
        )
        info = data[ev_id]
        info.ax = ax
        info.im = im
        info.smap = get_smap(data=info.mov, cmap=cmap, qlim=qlim)
        info.label = label_obj

    labels = [""] * n_frames_chunk
    for i in range(n_frames_pre, n_frames_pre + n_frames_stim):
        labels[i] = "on"

    n_frames = data[1].mov.shape[0]
    writer = FFMpegWriter(fps=fps)
    if not os.path.isabs(outfile):
        outfile = s.fs.getsyspath(outfile)
    with writer.saving(fig, str(outfile), dpi):
        for i in range(n_frames):
            for ev_id in range(1, 5):
                info = data[ev_id]
                cdata = info.smap(info.mov[i])
                cdata[:, :, -1] = 1
                info.im.set_data(cdata)
                info.label.set_text(labels[i])
            writer.grab_frame()
    if f:
        f.close()

def make_diff_movies(s: Session, movpath='scratch/mov_lum_filtered.h5', diffmode=False):

    f = h5py.File(s.fs.getsyspath(movpath), 'r')
    dset = f['data']
    indata = da.from_array(dset)

    combinations = [
        ('top_left', 'top_right'),
        ('top_left', 'bottom_left'),
        ('top_right', 'bottom_right'),
        ('bottom_left', 'bottom_right'),
    ]
    for events in combinations:
        if diffmode:
            outfile = f'analysis/{events[0]}-{events[1]}_diff.mp4'
        else:
            outfile = f'analysis/{events[0]}-{events[1]}.mp4'
        width = 4
        dpi = 220
        frameon = True
        facecolor = None

        schema = s.events.schema
        fps = s.events.get_fps()
        ev_df = s.events.tables["event"]
        mask = np.load(s.fs.getsyspath('scratch/mask.npy'))
        duration = schema.get(event=1).duration
        n_frames_stim = round(fps * duration)
        n_frames_pre = round(5 * fps)
        n_frames_post = round(5 * fps)
        n_frames_chunk = n_frames_pre + n_frames_stim + n_frames_post

        A = SimpleNamespace(event=schema.get(event=events[0]), difference=False)
        B = SimpleNamespace(event=schema.get(event=events[1]), difference=False)
        Full = SimpleNamespace(event=schema.get(event='full'), difference=False)
        plots = [A, B, Full]
        for i, info in enumerate(plots):
            info.name = info.event.name
            stack = []
            for onset in ev_df[ev_df.event == info.event.id].start:
                start = onset - n_frames_pre
                if start < 0:
                    continue
                stop = onset + n_frames_stim + n_frames_post
                if stop >= indata.shape[0]:
                    continue
                chunk = indata[start:stop]
                if hasattr(chunk, 'compute'):
                    chunk = chunk.compute()
                stack.append(chunk)
            info.stack = np.stack(stack)
            info.mov = info.stack.mean(axis=0)
            info.mov = add_mask(info.mov, mask, fill_value=0.0)
            info.mov = gaussian_filter(info.mov, [2, 2, 2])

        if diffmode:
            A.mov = Full.mov - A.mov
            B.mov = Full.mov - B.mov
            A.difference = B.difference = True

        C = SimpleNamespace(event=None, mov=A.mov - B.mov, difference=True)
        C.name = f"{A.name} - {B.name}"
        plots = [A, B, C]

        frame_shape = A.mov.shape[1:]
        ypix, xpix = frame_shape
        aspect = ypix / xpix
        figsize = (width, 3 * width * aspect)
        fig = Figure(
            figsize=figsize,
            frameon=frameon,
            facecolor=facecolor,
        )
        fig.tight_layout()

        for i, info in enumerate(plots):
            ax = fig.add_subplot(3, 1, i + 1)
            remove_ticks(ax)
            ax.set_aspect("equal")
            ax.set_title(info.name)
            im = ax.imshow(np.zeros(frame_shape))
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
            info.ax = ax
            info.im = im
            info.label = label_obj
            if info.difference:
                info.smap = get_smap(data=info.mov, cmap='coolwarm', qlim=(50, 99))
            else:
                info.smap = get_smap(data=info.mov, cmap='inferno', qlim=(1, 99.9))

        labels = [""] * n_frames_chunk
        for i in range(n_frames_pre, n_frames_pre + n_frames_stim):
            labels[i] = "on"

        n_frames = A.mov.shape[0]
        writer = FFMpegWriter(fps=fps)
        if not os.path.isabs(outfile):
            outfile = s.fs.getsyspath(outfile)
        with writer.saving(fig, str(outfile), dpi):
            for i in range(n_frames):
                for info in plots:
                    cdata = info.smap(info.mov[i])
                    cdata[:, :, -1] = 1
                    info.im.set_data(cdata)
                    info.label.set_text(labels[i])

                writer.grab_frame()


def process_movie(s):

    from time import perf_counter as clock
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
    t = clock()
    f = f_std = h5py.File(s.fs.getsyspath('mov.h5'), 'w')
    dst = f.create_dataset('data', shape=mov.shape, dtype=np.float32)
    for i in range(mov.shape[0]):
        frame = mov[i]
        frame = frame / mean_frame
        frame = np.nan_to_num(frame, posinf=0.0, neginf=0.0)
        dst[i] = frame
    f.create_dataset('mask', data=mask)
    f.create_dataset('G', data=T_f / T_f.mean())
    f.create_dataset('mean_frame', data=mean_frame)
    f_std.close()
    print(f'... standardized frames in {clock() - t} secs')

    # print('making movie')
    # f = f_mov = h5py.File(s.fs.getsyspath('mov.h5'), 'r')
    # dset = f['data']
    # indata = da.from_array(dset)
    # make_mov(s, indata, s.fs.getsyspath('mov.mp4'))
    # f_mov.close()
    # print('... finished making movie')

    print(f'finished processing movie data in {clock() - t_start} secs')



FILES = weakref.WeakValueDictionary()


def open_movie(s: Session, file='mov.h5', name='data'):
    path = s.fs.getsyspath(file)
    file_key = f"{path}:{name}"
    f = h5py.File(path, "r")
    dset = f[name]
    arr = da.from_array(dset)
    FILES[file_key] = f
    return arr



# mouse, date, run = "55709-5", "2022-10-07", "1"
# s = open_session(mouse, date, run, fs="ssd")
# pull_session(s)
# del s
# s = open_session(mouse, date, run, fs="ssd")
# import_session(s)
# s = open_session(mouse, date, run, fs="ssd")
# process_movie(s)

# s = open_session("55709-2", "2022-09-26", "1", fs="ssd")
# make_mov(s, 'mov.h5', 'analysis/mov.mp4')
# make_mov(s, 'mov.h5', 'analysis/mov_diff.mp4', diffmode=True)
# make_diff_movies(s, 'mov.h5')
# print('finished')


def open_viewer(s: Session):

    with h5py.File(s.fs.getsyspath("mov.h5"), "r") as f:
        background = f["mean_frame"][:]
        background = median_filter(background, 2)
        mask = f["mask"][:]
        y, x = np.where(mask)
        background[y, x] = 0
        background = np.ma.array(background, mask=mask)

    viewer = napari.view_image(background)
    bg_layer = viewer.layers[0]
    bg_layer.name = "background"
    vmin, vmax = percentile(background, [5, 95])
    bg_layer.contrast_limits = [vmin, vmax]

    win = viewer.window

    win.add_plugin_dock_widget('napari-roi')
    layer = napari.layers.Shapes(
        edge_color="red",
        face_color="transparent",
    )
    layer.name = "ROIs"
    viewer.add_layer(layer)

    return viewer


def save_roi_data(path, layer, masks: bool = True) -> None:

    path = str(path)
    path = path.rstrip(".csv")
    path = Path(path).with_suffix(".h5")

    names = layer.properties['roi_name']
    if masks:
        mask_data = layer.to_masks()
    else:
        mask_data = None

    with h5py.File(path, "w") as f:
        for i in range(layer.nshapes):
            group = f.create_group(names[i])
            group.create_dataset('data', data=layer.data[i])
            group.attrs["shape_type"] = layer.shape_type[i]
            y, x = np.where(masks[i])
            group.create_dataset('y', data=y)
            group.create_dataset('x', data=x)
            if masks:
                group.create_dataset('mask', data=mask_data[i])


def load_roi_data(path) -> Mapping:
    dct = {'name': [], 'data': [], 'shape_type': [], 'y': [], 'x': []}
    with h5py.File(path, "r") as f:
        dset_names = list(f.keys())
        for i, name in enumerate(dset_names):
            group = f[name]
            dct['name'].append(name)
            dct['data'].append(group['data'][:])
            dct['shape_type'].append(str(group.attrs['shape_type']))
            dct['y'].append(group['y'][:])
            dct['x'].append(group['x'][:])

    return dct


s = open_session("55709-1", "2022-09-26", "1", fs=0)
with h5py.File(s.fs.getsyspath("mov.h5"), "r") as f:
    background = f["mean_frame"][:]
    background = median_filter(background, 2)
    mask = f["mask"][:]
    y, x = np.where(mask)
    background[y, x] = 0
    background = np.ma.array(background, mask=mask)


def extract_fluorescence(s):
    path = s.fs.getsyspath("rois.h5")
    dct = load_roi_data(h5_path)
    n_rois = len(dct['name'])

    f = h5py.File(s.fs.getsyspath("mov.h5"), "r")
    arr = da.from_array(f["data"])
    n_timepoints = arr.shape[0]

    F = np.zeros([n_rois, n_timepoints])
    for i, slc, chunk in iterchunks(arr, 1000):
        slc = slc.raw[0]
        start, stop = slc.start, slc.stop
        chunk = chunk.compute()
        for j in range(n_rois):
            y, x = dct['y'][j], dct['x'][j]
            mat = chunk[:, y, x]
            vals = mat.mean(axis=1)
            F[j, start:stop] = vals

    F = xr.DataArray(F, dims=('roi', 'time'))
    return F


class Player:

    def __init__(self, s: Session):
        self.session = s
        self.files = {'mov': None, 'rois': None}

        f = h5py.File(self.session.fs.getsyspath("mov.h5"), 'r')
        self.files['mov'] = f
        dset = f['data']
        arr = da.from_array(dset, chunks=(10, -1, -1))
        if 'mask' in f.keys():
            mask = f['mask'][:]
            y, x = np.where(mask)

            def mask_block(array):
                out = np.asarray(array)
                if out.ndim == 2:
                    out[y, x] = np.nan
                else:
                    out[:, y, x] = np.nan
                return out

            arr = arr.map_blocks(mask_block, dtype='float64')

        self.arr = arr
        self.viewer = None

    def view(self):
        if self.viewer is None:
            self.viewer = napari.view_image(self.arr)
        return self

    def close(self):
        if self.viewer:
            try:
                self.viewer.close()
            except RuntimeError:
                pass
        for key, val in self.files.items():
            if val:
                val.close()
            self.files[key] = None


s = open_session("55709-1", "2022-09-26", "1", fs='ca-nas')
# v = play_movie(s.fs.getsyspath('thorlabs/Image.raw'))

# cat = intake.open_catalog(s.fs.getsyspath('cat.yaml'))

# finalize_alignment(s)
# df = s.events['frame']
# time = df['time'].values
# path = s.fs.getsyspath('mov.h5')
# src = H5DatasetSource(URL(path, name='data'))
# path = s.fs.getsyspath('mov.h5')

# path = get_fs('ca-nas').getsyspath(f"sessions/55709-1/2022-09-26/1/thorlabs/Image.raw")

# src = ThorImageArraySource(path)

# f = h5py.File(s.fs.getsyspath('mov.h5'), 'r')
# dset = f['data']
# mov = da.from_array(dset, chunks=(1, -1, -1))
#
# a = xr.DataArray(mov, dims=('time', 'y', 'x'))
# a.coords['time'] = time
# t = xr.DataArray(time, dims=('time'))
#
# a.coords['time'] = t
#
# x = a.coords['time']

# p = Player(s)
# p.view()
# std = p.arr.std(axis=0).compute()
# mean = p.arr.mean(axis=0).compute()
# mat = std - mean
# mat = np.abs(mat)
#
# im = plt.imshow(std, cmap='inferno')
# plt.colorbar(im)
# plt.show()

viewer = open_viewer(s)
#
# h5_path = s.fs.getsyspath("rois.h5")
# dct = load_roi_data(h5_path)
# n_rois = len(dct['name'])
#
# F = extract_fluorescence(s)
#
#
# # da_mask_3d = da.broadcast_to(mask, da_arr.shape)
# # da_masked = da.ma.masked_array(da_arr, da_mask_3d)
#
# # chunk = arr[0:200]
# # mask_3d = np.broadcast_to(mask, chunk.shape)
# #
# # m = add_mask(mov, mask)
# # napari.view_image(m)
#
# # F = extract_fluorescence(s)
# fig, axes = plt.subplots(n_rois, 1, figsize=(12, 9))
# start = 5000
# stop = 6000
# for i in range(n_rois):
#     ax = axes[i]
#     ax.plot(F[i, start:stop])
# plt.show()
#
#
