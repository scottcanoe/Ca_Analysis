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


def make_mov(
    s: Session,
    indata: ArrayLike,
    outfile: PathLike,
    width=8,
    dpi=220,
    qlim=(1, 99.9),
    frameon=True,
    facecolor=None,
    fps=10,
):
    ev_df = s.events.tables["event"]
    ev_df = ev_df[ev_df.event == 1]

    fps = s.events.get_fps()
    duration = s.events.schema.get(event='on').duration
    n_frames_stim = round(fps * duration)
    n_frames_pre = round(5 * fps)
    n_frames_post = round(5 * fps)
    n_frames_chunk = n_frames_pre + n_frames_stim + n_frames_post

    stack = []
    for i in range(len(ev_df)):
        row = ev_df.iloc[i]
        start = row.start - n_frames_pre
        if start < 0:
            continue
        stop = row.start + n_frames_stim + n_frames_post
        if stop >= indata.shape[0]:
            continue
        chunk = indata[start:stop]
        if hasattr(chunk, 'compute'):
            chunk = chunk.compute()
        stack.append(chunk)

    stack = np.stack(stack)
    data = np.mean(stack, axis=0)
    data = gaussian_filter(data, (2, 2, 2))

    smap = get_smap(data=data, cmap='inferno', qlim=qlim)

    ypix, xpix = data[0].shape
    aspect = ypix / xpix
    figsize = (width, width * aspect)
    fig = Figure(
        figsize=figsize,
        frameon=frameon,
        facecolor=facecolor,
    )
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    im = ax.imshow(np.zeros_like(data[0]))

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
    labels = [""] * n_frames_chunk
    for i in range(n_frames_pre, n_frames_pre + n_frames_stim):
        labels[i] = "on"

    writer = FFMpegWriter(fps=fps)
    with writer.saving(fig, str(outfile), dpi):
        for i in range(data.shape[0]):
            cdata = smap(data[i])
            cdata[:, :, -1] = 1
            im.set_data(cdata)
            label_obj.set_text(labels[i])
            writer.grab_frame()

    ts = np.zeros(len(data))
    for i in range(len(ts)):
        ts[i] = data[i].mean()

    fig, ax = plt.subplots()
    X = np.arange(len(ts))
    secs = X / 10
    ax.plot(secs, ts)
    ax.axvline(n_frames_pre, color='black', ls='--')
    ax.axvline(n_frames_pre + n_frames_stim, color='black', ls='--')
    fig.savefig(s.fs.getsyspath('gach.png'))


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

    print('mean frame computing')
    t = clock()
    f = f_med = h5py.File(s.fs.getsyspath('scratch/mov_median_filtered.h5'), 'r')
    med = da.from_array(f["data"])
    mean_frame = med.mean(axis=0).compute()
    np.save(s.fs.getsyspath("scratch/mean_frame.npy"), mean_frame)
    print(f'... finished mean frame computing in {clock() - t} secs')

    print('mean frame subtracting')
    t = clock()
    f = f_mov = h5py.File(s.fs.getsyspath('mov.h5'), 'w')
    dset = f.create_dataset('data', shape=med.shape, dtype=np.float32)
    for i in range(med.shape[0]):
        frame = med[i].compute()
        dset[i] = frame - mean_frame
    f_med.close()
    f_mov.close()
    print(f'... finished mean subtracting in {clock() - t} secs')

    # print('making movie')
    # f = f_mov = h5py.File(s.fs.getsyspath('mov.h5'), 'r')
    # dset = f['data']
    # indata = da.from_array(dset)
    # make_mov(s, indata, s.fs.getsyspath('mov.mp4'))
    # f_mov.close()
    # print('... finished making movie')

    print(f'finished processing movie data in {clock - t_start} secs')


import weakref


FILES = weakref.WeakValueDictionary()


def open_mov(s: Session, name='mov.h5'):
    path = s.fs.getsyspath(name)
    f = h5py.File(path, "r")
    dset = f['data']
    arr = da.from_array(dset)
    FILES[path] = f
    return arr


# s = open_session("55708-2", "2022-09-25", "1", fs="ssd")
# mask = np.load(s.fs.getsyspath("scratch/mask.npy"))
# for match in s.fs.glob("scratch/mov*.h5"):
#     path = s.fs.getsyspath(match.path)
#     with h5py.File(path, 'r+') as f:
#         if 'mask' in f:
#             del f['mask']
#         f.create_dataset('mask', data=mask)
# print('finished')
# mov_path = s.fs.getsyspath('scratch/mov_median_filtered.h5')
# f = h5py.File(mov_path, "r")
# dset = f['data']
# arr = da.from_array(dset)
# mov = dset[:]
#
# S = mov.mean(axis=0)
# np.save(s.fs.getsyspath('scratch/S.npy'), S)
#
# mask = np.load(s.fs.getsyspath("scratch/mask.npy"))
# mask3d = np.broadcast_to(mask, mov.shape)
# mov = np.ma.masked_array(mov, mask=mask3d)

# T_f = np.array([frame.compressed().mean() for frame in mov])
# np.save(s.fs.getsyspath("scratch/T_f.npy"), T_f)
#
# S = np.load(s.fs.getsyspath("scratch/S.npy"))
# T_f = np.load(s.fs.getsyspath("scratch/T_f.npy"))
#
# G = T_f / T_f.mean()
# out = np.zeros(mov.shape, dtype=np.float32)

# posinf = neginf = 0.0
#
# for i in range(mov.shape[0]):
#     out[i] = mov[i] / S - G[i]
#
# if posinf is not None or neginf is not None:
#     for i in range(mov.shape[0]):
#         out[i] = np.nan_to_num(out[i], posinf=posinf, neginf=neginf)
#
# f_out = h5py.File(s.fs.getsyspath("scratch/mov_lum_filtered.h5"), 'w')
# dst = f_out.create_dataset('data', data=out)
# f_out.close()
# print('finished')
#


# diffmode = True
# f = h5py.File(s.fs.getsyspath('mov.h5'), 'r')
# dset = f["data"]
# mov = da.from_array(dset)
# mask = da.from_array(np.load(s.fs.getsyspath('scratch/mask.npy')))
#
# indata = mov
# outfile = s.fs.getsyspath('test_no_diff.mp4')
# width = 8
# dpi = 220
# if diffmode:
#     outfile = s.fs.getsyspath('mean_diff.mp4')
#     cmap = 'coolwarm'
#     qlim = (50, 99.9)
# else:
#     outfile = s.fs.getsyspath('means.mp4')
#     cmap = 'inferno'
#     qlim = (1, 99.9)
# frameon = True
# facecolor = None
#
# schema = s.events.schema
# fps = s.events.get_fps()
# ev_df = s.events.tables["event"]
#
# duration = schema.get(event=1).duration
# n_frames_stim = round(fps * duration)
# n_frames_pre = round(5 * fps)
# n_frames_post = round(5 * fps)
# n_frames_chunk = n_frames_pre + n_frames_stim + n_frames_post
#
# data = {}
# for ev_id in range(1, 6):
#     sub_df = ev_df[ev_df.event == ev_id]
#     info = {'event': schema.get(event=ev_id)}
#     lst = []
#     for i in range(len(sub_df)):
#         row = sub_df.iloc[i]
#         start = row.start - n_frames_pre
#         if start < 0:
#             continue
#         stop = row.start + n_frames_stim + n_frames_post
#         if stop >= indata.shape[0]:
#             continue
#         chunk = indata[start:stop]
#         if hasattr(chunk, 'compute'):
#             chunk = chunk.compute()
#         lst.append(chunk)
#     stack = np.stack(lst)
#     info['stack'] = stack
#     info['mov'] = np.mean(stack, axis=0)
#     info['mov'] = gaussian_filter(info['mov'], (2, 2, 2))
#     data[ev_id] = info
#
# if diffmode:
#     full = data[5]
#     full_mov = full['mov']
#     for ev_id in range(1, 5):
#         info = data[ev_id]
#         info['mov'] = full_mov - info['mov']
#
# # data[1]['mov'] = data[4]['mov'] - data[2]['mov']
#
# # data = gaussian_filter(data, (2, 2, 2))
# frame_shape = data[1]['mov'].shape[1:]
# ypix, xpix = frame_shape
# aspect = ypix / xpix
# figsize = (width, width * aspect)
# fig = Figure(
#     figsize=figsize,
#     frameon=frameon,
#     facecolor=facecolor,
# )
# for ev_id in range(1, 5):
#     ax = fig.add_subplot(2, 2, ev_id)
#     ax.set_aspect("equal")
#     ax.set_xticks([])
#     ax.set_xticklabels([])
#     ax.set_yticks([])
#     ax.set_yticklabels([])
#     im = ax.imshow(np.zeros(frame_shape))
#
#     fontdict = {'size': 16, 'color': 'white'}
#     label_loc = [0.05, 0.95]
#     label_obj = ax.text(
#         label_loc[0],
#         label_loc[1],
#         ' ',
#         fontdict=fontdict,
#         transform=ax.transAxes,
#         horizontalalignment='left',
#         verticalalignment='top',
#         usetex=False,
#     )
#     info = data[ev_id]
#     info['ax'] = ax
#     info['im'] = im
#     info['smap'] = get_smap(data=info['mov'], cmap=cmap, qlim=qlim)
#     info['label'] = label_obj
#
# labels = [""] * n_frames_chunk
# for i in range(n_frames_pre, n_frames_pre + n_frames_stim):
#     labels[i] = "on"
#
# n_frames = data[1]['mov'].shape[0]
# writer = FFMpegWriter(fps=fps)
# with writer.saving(fig, str(outfile), dpi):
#     for i in range(n_frames):
#         for ev_id in range(1, 5):
#             info = data[ev_id]
#             cdata = info['smap'](info['mov'][i])
#             cdata[:, :, -1] = 1
#             info['im'].set_data(cdata)
#             info['label'].set_text(labels[i])
#
#         writer.grab_frame()
#
# print('finished')
