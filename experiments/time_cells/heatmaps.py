from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_DIR = Path.home() / 'time_cells/data/CSVs'
ORIENTATIONS = np.array([0, 30, 60, 90, 120, 150])
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


region = 'VISp'
block = 8
ori = 150
binsize = 0.005
bins = np.arange(0, 0.25 + binsize, binsize)

all_counts = []
even = []
odd = []
for sid in get_SIDs(region, block):
    sdf = read_csv(region, block, sid)
    unit_ids = np.unique(sdf['unit_id'])
    for uid in unit_ids:
        udf = sdf[sdf['unit_id'] == uid]
        odf = udf[udf['orientation'] == ori]
        offsets = odf['time_since_stimulus_presentation_onset'].values
        counts, _ = np.histogram(offsets, bins=bins)
        all_counts.append(counts)

        n = len(offsets)
        even_inds = np.arange(0, n, 2, dtype=int)

        even_inds = np.arange(0, n, 2, dtype=int)
        counts, _ = np.histogram(offsets[even_inds], bins=bins)
        even.append(counts)

        odd_inds = np.arange(1, n, 2, dtype=int)
        counts, _ = np.histogram(offsets[odd_inds], bins=bins)
        odd.append(counts)


mat = np.stack(all_counts)
even = np.stack(even)
odd = np.stack(odd)

inds = np.argsort(np.argmax(even, axis=1))
mat = odd[inds]
vmin, vmax = np.percentile(mat, [2.5, 97.5])
# i_maxs = np.argmax(mat, axis=1)

fig, ax = plt.subplots()
im = ax.imshow(mat, cmap='inferno', vmin=vmin, vmax=vmax)#, interpolation='none')
ax.set_aspect('auto')
plt.colorbar(im)
ax.set_xticks([0, 10, 20, 30, 40])
ax.set_xticklabels([0, 50, 100, 150, 200])
ax.set_xlabel('time (msec)')
ax.set_title(f'VISp: ori={ori}, block={block}')
fig.tight_layout(pad=0.6)
plt.show()
fig.savefig(f'VISp_ori_{ori}_block_{block}.png')

