import weakref
from time import perf_counter as clock

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


import napari
from ca_analysis.io.thorlabs import ThorImageArraySource


def do_import(mouse, date, run, schema, day):

    path = f'sessions/{mouse}/{date}/{run}'
    remote = get_fs(-1)
    local = get_fs(0)
    pyfs.copy.copy_dir_if(remote, path, local, path, "newer", preserve_time=True)
    s = open_session(mouse, date, run, fs=0)
    import_session(s)
    pyfs.copy.copy_dir_if(local, path, remote, path, "newer", preserve_time=True)
    s.fs.rmtree("thorlabs", missing_ok=True)
    s.fs.remove("mov.h5")


to_import = [
    ("55708-1", "2022-10-12", "1", 0),
    ("55708-1", "2022-10-17", "1", 0),
]

for tup in to_import:
    do_import(tup[0], tup[1], tup[2], 'seq_learn_3', tup[3])


# fs_branch = get_fs('ca-nas')
# path = fs_branch.getsyspath(f'sessions/{mouse}/{date}/{run}/thorlabs/Image.raw')
# src = ThorImageArraySource(path)
# arr = src.to_dask()
# viewer = napari.view_image(arr)

# if is_array(obj):
#     _chunks = list(obj.shape)
#     _chunks[0] = chunks
#     arr = dask.array.from_array(obj, chunks=_chunks)
# else:
#     url = URL(obj)
#     path = Path(url.path)
#     if h5py.is_hdf5(path):
#         from ca_analysis.io.h5 import H5DatasetSource
#         src = H5DatasetSource(url, chunks=chunks)
#         arr = src.to_dask()
#     elif fnmatch(path.name, "Image*.raw"):
#         from ca_analysis.io.thorlabs import ThorImageArraySource
#         src = ThorImageArraySource(url.path, chunks=chunks)
#         arr = src.to_dask()
#     else:
#         raise ValueError(f'unsupported filetype: {url}')
#
# viewer = napari.view_image(arr)



# mouse, date, run = "55709-5", "2022-10-07", "1"
# s = open_session(mouse, date, run, fs="ssd")
# pull_session(s)
# del s
# s = open_session(mouse, date, run, fs="ssd")
# import_session(s)
# s = open_session(mouse, date, run, fs="ssd")
# process_movie(s)
#
# # s = open_session("55709-2", "2022-09-26", "1", fs="ssd")
# make_mov(s, 'mov.h5', 'analysis/mov.mp4')
# make_mov(s, 'mov.h5', 'analysis/mov_diff.mp4', diffmode=True)
# make_diff_movies(s, 'mov.h5')
# print('finished')
#

# mask = f['ma']
# arr = da.from_array(dset)


# mov = open_movie(s)
# frame = mov[100:200].mean(axis=0).compute()
# smap = get_smap('inferno', data=frame, qlim=(2.5, 97.5))
# image = smap(frame)
# napari.view_image(image)