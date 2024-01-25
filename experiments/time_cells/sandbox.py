from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_DIR = Path.home() / 'time_cells/data/CSVs'
ORIENTATIONS = np.array([0.0, 30.0, 60.0, 90.0, 120.0, 150.0])
BLOCKS = [8, 11, 14]

def get_SIDs(region: str, block: int) -> np.array:
    pattern = f'*{region}_block_{block}*'
    out = [p.stem.split('_')[-1] for p in DATA_DIR.glob(pattern)]
    out = np.array(out, dtype=object)
    return out


def read_csv(region: str, block: int, sid: str) -> pd.DataFrame:
    path = DATA_DIR / f'neuropixels_static_gratings_{region}_block_{block}_session_{sid}.csv'
    df = pd.read_csv(path)
    df = df.dropna()
    df['orientation'] = df['orientation'].astype(int)
    return df

#
# region = 'VISp'
# block = 8
# ori = 0
#
# binsize = 0.01
# bins = np.arange(0, 0.5 + binsize, binsize)
# all_counts = []
# for sid in get_SIDs(region, block):
#     print(sid)
#     sdf = read_csv(region, block, sid)
#     unit_ids = np.unique(sdf['unit_id'])
#     for uid in unit_ids:
#         unit_df = sdf[sdf['unit_id'] == uid]
#         ori_df = unit_df[unit_df['orientation'] == ori]
#
#         # get spikes from target stim
#         ori_offsets = ori_df['time_since_stimulus_presentation_onset'].values
#         # counts, _ = np.histogram(offsets, bins=bins)
#         # all_counts.append(counts)
#
#         # get spikes from subsequent stims
#         stim_ids = np.unique(ori_df['stimulus_presentation_id'])
#         next_ids = stim_ids + 1
#         next_offsets = []
#         for nxt in next_ids:
#             sub_df = unit_df[unit_df['stimulus_presentation_id'] == nxt]
#             offsets = sub_df['time_since_stimulus_presentation_onset'].values + 0.25
#             next_offsets.append(offsets)
#         next_offsets = np.concatenate(next_offsets)
#         all_offsets = np.concatenate([ori_offsets, next_offsets])
#         counts, _ = np.histogram(all_offsets, bins=bins)
#         all_counts.append(counts)
#
# mat = np.stack(all_counts)
#
# inds = np.argsort(np.argmax(mat, axis=1))
# mat = mat[inds]
# vmin, vmax = np.percentile(mat, [2.5, 97.5])

mat2 = mat.copy()
mat2[:, 25:] = mat2[:, 25:] / 5
fig, ax = plt.subplots()
im = ax.imshow(mat2, cmap='inferno', vmin=vmin, vmax=vmax)#, interpolation='none')
plt.colorbar(im)
ax.set_aspect('auto')
ax.set_xticks([0, 10, 20, 30, 40])
ax.set_xticklabels([0, 100, 200, 300, 400])
ax.set_xlabel('time (msec)')
ax.set_ylabel('unit', rotation=90)
fig.tight_layout(pad=0.5)
plt.show()

# uid = unit_ids[4]
# udf = df_all[df_all['unit_id'] == uid]
#
#
# plot_dir = '/home/scott/plots/time_cells/data/PSTHs'
# # fig, axes = plt.subplots(6, 1)
# fig, ax = plt.subplots()
# binsize = 0.025
# bins = np.arange(0, 0.25 + binsize, binsize)
#
# colors = ['black', 'gray', 'red', 'orange', 'blue', 'lightblue']
# for i in range(len(ORIENTATIONS)):
#     ori = ORIENTATIONS[i]
#     odf = udf[udf['orientation'] == ori]
#     offsets = odf['time_since_stimulus_presentation_onset']
#     counts, _ = np.histogram(offsets, bins=bins)
#     # ax = axes[i]
#     # X = 1000 * bins[:-1]
#     X = bins[:-1]
#     X = np.arange(len(counts))
#     # ax.bar(X, counts)
#     ax.plot(X, counts, color=colors[i], label=str(ori))
#     # ax.set_title(ori)
#
# ax.legend()
# fig.tight_layout(pad=0.5)
# plt.show()
#
