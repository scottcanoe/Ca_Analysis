import dask.array as da
import numpy as np

from main import *
from processing import *
from ca_analysis.processing.utils import fix_frame_trigger
import matplotlib.pyplot as plt

import h5py
from matplotlib.figure import Figure
from matplotlib.animation import FFMpegWriter
import numpy as np

from ca_analysis.plot import get_smap


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

    # if mask is not None:
    #     data = add_mask(data, mask, fill_value=0)

    # data = median_filter(data, size=(2, 2, 1))
    mov = luminance_filter(data)

    with h5py.File(s.reg["mov"].resolve(), "w") as f:
        if np.ma.is_masked(data):
            f.create_dataset("data", data=mov.data)
            f.create_dataset("mask", data=mov.mask[0])
        else:
            f.create_dataset("data", data=mov)


def luminance_filter(
    mov: np.ndarray,
    posinf: Optional[Number] = 0.0,
    neginf: Optional[Number] = 0.0,
) -> np.ndarray:
    """
    "Correct" pixels by subtracting global activity. Converts raw pixel
    values to pixel-wise fractions of means minus global fraction of mean.


    Mask should be True

    pix_corrected = pix_t / mean(pix) - global_t / mean(global)

    mask: `True` indicates pixel is masked.

    """

    # Handled masked input array.
    if np.ma.is_masked(mov):
        print('its masked')
        out = np.ma.zeros(mov.shape, dtype=np.float32)
        out.mask = mov.mask
        out.fill_value = mov.fill_value
    else:
        mov = np.asarray(mov)
        out = np.zeros_like(mov, dtype=np.float32)

    """
    T_f: mean luminance for each frame (1d)
    G: mean luminance for each frame divided by its mean (1d).
    S: pixel-wise mean luminance (2d).

    """
    T_f = np.array([im.mean() for im in mov])
    G = T_f / np.nanmean(T_f)
    S = np.nanmean(mov, axis=0)

    datadir = Path.home() / 'gach'
    np.save(datadir / 'T_f.npy', T_f)
    np.save(datadir / 'G.npy', G)
    np.save(datadir / 'S.npy', S)

    for i in range(mov.shape[0]):
        out[i] = mov[i] / S  # - G[i]

    if posinf is not None or neginf is not None:
        for i in range(mov.shape[0]):
            out[i] = np.nan_to_num(out[i], posinf=posinf, neginf=neginf)

    return out


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
    n_frames_pre = round(14 * fps)
    n_frames_post = round(14 * fps)
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

    f = f_raw = h5py.File(s.fs.getsyspath('scratch/mov_motion_corrected.h5'), 'r')
    dset = f['data']
    raw = da.from_array(dset)

    print('median filtering')
    f = f_med = h5py.File(s.fs.getsyspath('scratch/mov_median_filtered.h5'), 'w')
    dset = f.create_dataset('data', shape=raw.shape, dtype=np.float32)
    for i in range(raw.shape[0]):
        frame = raw[i].compute()
        frame = median_filter(frame, (2, 2))
        dset[i] = frame
    f_raw.close()
    f_med.close()
    print('... finished median filtering')

    print('mean frame computing')
    f = f_med = h5py.File(s.fs.getsyspath('scratch/mov_median_filtered.h5'), 'r')
    med = da.from_array(f["data"])
    mean_frame = med.mean(axis=0).compute()
    print('... finished mean frame computing')

    print('mean frame subtracting')
    f = f_mov = h5py.File(s.fs.getsyspath('mov.h5'), 'w')
    dset = f.create_dataset('data', shape=med.shape, dtype=np.float32)
    for i in range(med.shape[0]):
        frame = med[i].compute()
        dset[i] = frame - mean_frame
    f_med.close()
    f_mov.close()
    print('... finished mean subtracting')

    print('making movie')
    f = f_mov = h5py.File(s.fs.getsyspath('mov.h5'), 'r')
    dset = f['data']
    indata = da.from_array(dset)
    make_mov(s, indata, s.fs.getsyspath('mov.mp4'))
    print('... finished making movie')


s = open_session("55708-2", "2022-09-09", "1", fs="ssd")

# process_movie(s)
#
f = f_mov = h5py.File(s.fs.getsyspath('mov.h5'), 'r')
dset = f['data']
indata = da.from_array(dset)

# arr = np.array([indata[i].mean().compute() for i in range(len(indata))])
# np.save(s.fs.getsyspath('mean.npy'), arr)
# print('finished')

# print('making movie')
# make_mov(s, indata, s.fs.getsyspath('mov.mp4'))
# print('... finished making movie')

ev_df = s.events.tables["event"]
ev_df = ev_df[ev_df.event == 1]

fps = s.events.get_fps()
duration = s.events.schema.get(event='on').duration
n_frames_stim = round(fps * duration)
n_frames_pre = round(14 * fps)
n_frames_post = round(14 * fps)
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
traces = []
for i in range(len(stack)):
    chunk = stack[i]
    arr = np.array([frame.mean() for frame in chunk])
    traces.append(arr)

traces = np.stack(traces)
mean = traces.mean(axis=0)
std = traces.std(axis=0)
upper = mean + std
lower = mean - std

fig, ax = plt.subplots()
X = np.arange(len(mean))
secs = X / 10
ax.fill_between(secs, upper, lower, color='red', alpha=0.4)
ax.plot(secs, mean)
ax.axvline(secs[n_frames_pre], color='black', ls='--')
ax.axvline(secs[n_frames_pre + n_frames_stim], color='black', ls='--')
# fig.savefig(s.fs.getsyspath('gach.png'))
plt.show()


# data = np.mean(stack, axis=0)
# data = gaussian_filter(data, (2, 2, 2))
#
#
# ts = np.zeros(len(data))
# for i in range(len(ts)):
#     ts[i] = data[i].mean()

# fig, ax = plt.subplots()
# X = np.arange(len(ts))
# secs = X / 10
# ax.plot(secs, ts)
# ax.axvline(secs[n_frames_pre], color='black', ls='--')
# ax.axvline(secs[n_frames_pre + n_frames_stim], color='black', ls='--')
# fig.savefig(s.fs.getsyspath('gach.png'))
# plt.show()

# f = f_mov = h5py.File(s.fs.getsyspath('mov.h5'), 'r')
# dset = f['data']
# indata = da.from_array(dset)
#
# arr = np.array([indata[i].mean().compute() for i in range(len(indata))])
# np.save(s.fs.getsyspath('mean.npy'), arr)
# print('finished')


# arr = np.load(s.fs.getsyspath('mean.npy'))
# arr = gaussian_filter(arr, 2)
# ev_df = s.events.tables["event"]
# zeros = ev_df[ev_df.event == 0]
#
# n_axes = 10
# axlen = len(arr) // n_axes
# fig, axes = plt.subplots(n_axes, 1, figsize=(18, 18))
# start = 0
# X = np.arange(len(arr))
# for i in range(n_axes):
#     stop = start + axlen
#     ax = axes[i]
#     ax.plot(X[start:stop], arr[start:stop])
#     start += axlen
# plt.tight_layout()
# fig.savefig(s.fs.getsyspath('a.pdf'))


# make_mov(s, indata, s.fs.getsyspath('mov.mp4'))
# print('... finished making movie')

# ev_df = s.events.tables["event"]
# ev_df = ev_df[ev_df.event == 1]
#
# fps = s.events.get_fps()
# duration = s.events.schema.get(event='on').duration
# n_frames_stim = round(fps * duration)
# n_frames_pre = round(7 * fps)
# n_frames_post = round(7 * fps)
# n_frames_chunk = n_frames_pre + n_frames_stim + n_frames_post
#
# stack = []
# for i in range(len(ev_df)):
#     row = ev_df.iloc[i]
#     start = row.start - n_frames_pre
#     if start < 0:
#         continue
#     stop = row.start + n_frames_stim + n_frames_post
#     if stop >= indata.shape[0]:
#         continue
#     chunk = indata[start:stop]
#     if hasattr(chunk, 'compute'):
#         chunk = chunk.compute()
#     stack.append(chunk)
#
# stack = np.stack(stack)
# data = np.mean(stack, axis=0)
# data = gaussian_filter(data, (2, 2, 2))
#
# smap = get_smap(data=data, cmap='inferno', qlim=qlim)
#
# ypix, xpix = data[0].shape
# aspect = ypix / xpix
# figsize = (width, width * aspect)
# fig = Figure(
#     figsize=figsize,
#     frameon=frameon,
#     facecolor=facecolor,
# )
# ax = fig.add_subplot(1, 1, 1)
# ax.set_aspect("equal")
# ax.set_xticks([])
# ax.set_xticklabels([])
# ax.set_yticks([])
# ax.set_yticklabels([])
# im = ax.imshow(np.zeros_like(data[0]))
#
# fontdict = {'size': 16, 'color': 'white'}
# label_loc = [0.05, 0.95]
# label_obj = ax.text(
#     label_loc[0],
#     label_loc[1],
#     ' ',
#     fontdict=fontdict,
#     transform=ax.transAxes,
#     horizontalalignment='left',
#     verticalalignment='top',
#     usetex=False,
# )
# labels = [""] * n_frames_chunk
# for i in range(n_frames_pre, n_frames_pre + n_frames_stim):
#     labels[i] = "on"
#
# writer = FFMpegWriter(fps=fps)
# with writer.saving(fig, str(outfile), dpi):
#     for i in range(data.shape[0]):
#         cdata = smap(data[i])
#         cdata[:, :, -1] = 1
#         im.set_data(cdata)
#         label_obj.set_text(labels[i])
#         writer.grab_frame()
#
# ts = np.zeros(len(data))
# for i in range(len(ts)):
#     ts[i] = data[i].mean()
#
# fig, ax = plt.subplots()
# ax.plot(ts)
# ax.axvline(n_frames_pre, color='black', ls='--')
# ax.axvline(n_frames_pre + n_frames_stim, color='black', ls='--')
# fig.savefig(s.fs.getsyspath('gach.png'))

#
# print('-------- here ---------')
#
# with h5py.File(s.fs.getsyspath('indata0.h5'), 'w') as f2:
#     dst = f2.create_dataset('data', shape=mov.shape, dtype=np.float32)
#     for i in range(mov.shape[0]):
#         frame = mov[i].compute()
#         frame = median_filter(frame, (3, 3))
#         dst[i] = frame
#
# f.close()
# f = h5py.File(s.fs.getsyspath('indata0.h5'), 'r')
# dset = f['data']
# mov = da.from_array(dset)
# S = mov.mean(axis=0).compute()
#
# with h5py.File(s.fs.getsyspath('indata.h5'), 'w') as f2:
#     dst = f2.create_dataset('data', shape=mov.shape, dtype=np.float32)
#     for i in range(mov.shape[0]):
#         frame = mov[i].compute()
#         frame = frame / S
#         dst[i] = frame
# f.close()
#
# evs = s.events.tables['event']
# f = h5py.File(s.fs.getsyspath('indata.h5'), 'r')
# indata = f["data"]
# outfile = s.fs.getsyspath('mov.mp4')
# make_mov(s, indata, outfile)
# f.close()
# print('-------- finished ---------')

# ev_df = s.events.tables["event"]
# ev_df = ev_df[ev_df.event == 1]
#
# n_frames_pre = 40
# n_frames_post = 80
# stack = []
# for i in range(len(ev_df)):
#     row = ev_df.iloc[i]
#     start = row.start - n_frames_pre
#     if start < 0:
#         continue
#     stop = row.start + n_frames_post
#     chunk = indata[start:stop]
#     stack.append(chunk)
#
# stack = np.stack(stack)
# data = np.mean(stack, axis=0)

# with h5py.File(s.fs.getsyspath('scratch/mov_motion_corrected.h5'), 'r') as f:
#     mov = f['data'][:]
# mov = median_filter(mov, (3, 3, 1))
# T_f = np.array([im.mean() for im in mov])
# G = T_f / np.nanmean(T_f)
# S = np.nanmean(mov, axis=0)

# row = 75
# col = 250
#
# p0 = mov[:, row, col]
# p1 = p0 / S[row, col]
# p2 = p1 - G
#
# def chunk_trace(x):
#     ev_df = s.events.tables["event"]
#     ev_df = ev_df[ev_df.event == 1]
#     n_frames = 40
#     n_frames_pre = 40
#     n_frames_post = 80
#     stack = []
#     for i in range(len(ev_df)):
#         row = ev_df.iloc[i]
#         start = row.start - n_frames_pre
#         if start < 0:
#             continue
#         stop = row.start + n_frames_post
#         chunk = x[start:stop]
#         stack.append(chunk)
#     stack = np.stack(stack)
#     return stack
#
# P0 = chunk_trace(p0)
# P1 = chunk_trace(p1)
# P2 = chunk_trace(p2)
# g = chunk_trace(G)
#
# num = 7
# fig, axes = plt.subplots(4, 1)
# axes[0].plot(P0[num])
# axes[1].plot(P1[num])
# axes[2].plot(P2[num])
# axes[3].plot(g[num])

# axes[0].plot(P0.mean(0))
# axes[1].plot(P1.mean(0))
# axes[2].plot(P2.mean(0))
# axes[3].plot(g.mean(0))

plt.show()
# datadir = Path.home() / "gach"
# T_f = np.load(datadir / 'T_f.npy')
# G = np.load(datadir / 'G.npy')
# S = np.load(datadir / 'S.npy')
#
# ev_df = s.events.tables["event"]
# ev_df = ev_df[ev_df.event == 1]
# n_frames = 40
# n_frames_pre = 40
# n_frames_post = 80
# stack = []
# for i in range(len(ev_df)):
#     row = ev_df.iloc[i]
#     start = row.start - n_frames_pre
#     if start < 0:
#         continue
#     stop = row.start + n_frames_post
#     chunk = G[start:stop]
#     stack.append(chunk)
#
# stack = np.stack(stack)
# mean = np.mean(stack, axis=0)
# fig, ax = plt.subplots()
# for i in range(stack.shape[0]):
#     ax.plot(stack[i])
# ax.plot(mean, ls='--', c='k')
# plt.show()
#
# run_luminance_filter(s)
# outfile = s.fs.getsyspath("test.mp4")
# width = 8
# dpi = 220
# qlim = (1, 99.9)
# frameon = True
# facecolor = None
# fps = 10
#
# with h5py.File(s.fs.getsyspath('mov.h5'), 'r') as f:
#     mov = f['data'][:]
# mask = np.load(s.fs.getsyspath('scratch/mask.npy'))
#
# ev_df = s.events.tables["event"]
# ev_df = ev_df[ev_df.event == 1]
# n_frames = 40
# n_frames_pre = 40
# n_frames_post = 80
# stack = []
# for i in range(len(ev_df)):
#     row = ev_df.iloc[i]
#     start = row.start - n_frames_pre
#     if start < 0:
#         continue
#     stop = row.start + n_frames_post
#     chunk = mov[start:stop]
#     stack.append(chunk)
# n_events = len(stack)
# stack = np.stack(stack)
# data = np.mean(stack, axis=0)
#
# # data = add_mask(data, mask, fill_value=0)
# mov = gaussian_filter(data, 1)
# smap = get_smap(data=data, cmap='inferno', qlim=qlim)
#
# ypix, xpix = mov[0].shape
# aspect = ypix / xpix
# figsize = (width, width * aspect)
# fig = Figure(
#     figsize=figsize,
#     frameon=frameon,
#     facecolor=facecolor,
# )
# ax = fig.add_subplot(1, 1, 1)
# ax.set_aspect("equal")
# ax.set_xticks([])
# ax.set_xticklabels([])
# ax.set_yticks([])
# ax.set_yticklabels([])
# im = ax.imshow(np.zeros_like(mov[0]))
#
# fontdict = {'size': 16, 'color': 'white'}
# label_loc = [0.05, 0.95]
# label_obj = ax.text(
#     label_loc[0],
#     label_loc[1],
#     ' ',
#     fontdict=fontdict,
#     transform=ax.transAxes,
#     horizontalalignment='left',
#     verticalalignment='top',
#     usetex=False,
# )
# labels = [""] * (n_frames_pre + n_frames_post)
# for i in range(n_frames_pre, n_frames_pre + 40):
#     labels[i] = "on"
#
# writer = FFMpegWriter(fps=fps)
# with writer.saving(fig, str(outfile), dpi):
#     for i in range(mov.shape[0]):
#         cdata = smap(mov[i])
#         cdata[:, :, -1] = 1
#         im.set_data(cdata)
#         label_obj.set_text(labels[i])
#         writer.grab_frame()
#
# ts = np.zeros(len(mov))
# for i in range(len(ts)):
#     ts[i] = mov[i].mean()
#
# fig, ax = plt.subplots()
# ax.plot(ts)
# fig.savefig(Path.home() / 'gach.png')
# print('finished')
#
