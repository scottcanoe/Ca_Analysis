from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from patsy import dmatrices, dmatrix
import statsmodels.api as sm


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
# binsize = 0.01
# bins = np.arange(0, 0.25 + binsize, binsize)
#
# all_counts = []
# even, odd = [], []
# for sid in get_SIDs(region, block):
#     sdf = read_csv(region, block, sid)
#     unit_ids = np.unique(sdf['unit_id'])
#     for uid in unit_ids:
#         udf = sdf[sdf['unit_id'] == uid]
#         odf = udf[udf['orientation'] == ori]
#         offsets = odf['time_since_stimulus_presentation_onset'].values
#         counts, _ = np.histogram(offsets, bins=bins)
#         all_counts.append(counts)
#
#         inds = np.arange(0, len(offsets), 2, dtype=int)
#         counts, _ = np.histogram(offsets[inds], bins=bins)
#         even.append(counts)
#
#         inds = np.arange(1, len(offsets), 2, dtype=int)
#         counts, _ = np.histogram(offsets[inds], bins=bins)
#         odd.append(counts)
#
# mat = np.stack(all_counts)
# inds = np.argsort(np.argmax(mat, axis=1))
# mat = mat[inds]
#
# even, odd = np.stack(even), np.stack(odd)
# inds = np.argsort(np.argmax(even, axis=1))
# odd = odd[inds]

# uid = 550
# y = mat[550]
# fig, ax = plt.subplots()
# ax.plot(y)
# plt.show()
#
# n_bins = len(y)
# X = np.eye(n_bins)
# dct = {}
# for i in range(n_bins):
#     name = f'bin_{i}'
#     dct[name] = X[:, i]
# rhs = ' + '.join(dct.keys())
# dct['count'] = y
# expr = f"""count ~ {rhs}"""
# df = pd.DataFrame(dct)
# y_train, X_train = dmatrices(expr, df, return_type='dataframe')
# glm = sm.GLM(y_train, X_train, family=sm.families.Poisson())
# res = glm.fit()
#
