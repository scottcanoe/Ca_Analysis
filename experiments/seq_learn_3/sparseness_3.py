from pathlib import Path

import h5py
from matplotlib.figure import Figure
import numpy as np
import xarray as xr
from experiments.seq_learn_3.main import get_sessions
from experiments.seq_learn_3.utils import *

# put path to data.h5 file here


def load(day: int, sequence: str) -> xr.DataArray:
    h5_path = Path(__file__).parent / 'data/data.h5'
    with h5py.File(h5_path, 'r') as f:
        group = f[f'day{day}']
        dset = group[sequence]
        out = xr.DataArray(dset[:], dims=('trial', 'time', 'roi'))
    return out


# roi = 0
# day = 0
# sessions = get_sessions(day=day, fs=0)
#
# baselines = []
# stds = []
#
# s = sessions[0]
# for s in sessions:
#     spikes = s.spikes.data
#     for i in range(spikes.sizes['roi']):
#         sp = spikes.isel(roi=i).isel(time=slice(0, 1800))
#         sp = sp.data.reshape(300, 6)
#         means = np.mean(sp, axis=1)
#         std_front = np.std(means)
#         baseline_front = means.mean()
#
#         length = spikes.sizes['time']
#         sp = spikes.isel(roi=i).isel(time=slice(length - 1800, None))
#         sp = sp.data.reshape(300, 6)
#         means = np.mean(sp, axis=1)
#         std_back = np.std(means)
#         baseline_back = means.mean()
#
#         baselines.append(np.mean([baseline_front, baseline_back]))
#         stds.append(np.mean([std_front, std_back]))
#
# baselines = np.array(baselines)
# stds = np.array(stds)
def run(day: int):


    ABCD_all = load(day, 'ABCD')
    ABBD_all = load(day, 'ABBD')
    ACBD_all = load(day, 'ACBD')

    roi_ids = []
    roi_stims = []

    n_sig = 0
    n_rois = ABCD_all.sizes['roi']
    for roi in range(n_rois):

        ABCD = ABCD_all.isel(roi=roi)
        ABBD = ABBD_all.isel(roi=roi)
        ACBD = ACBD_all.isel(roi=roi)

        ABCD_A = ABCD.isel(time=slice(2, 10)).mean('time')
        ABCD_B = ABCD.isel(time=slice(10, 18)).mean('time')
        ABCD_C = ABCD.isel(time=slice(18, 26)).mean('time')
        ABCD_D = ABCD.isel(time=slice(26, 34)).mean('time')
        ABCD_gray = ABCD.isel(time=slice(34, None)).mean('time')

        ABBD_A = ABBD.isel(time=slice(2, 10)).mean('time')
        ABBD_B1 = ABBD.isel(time=slice(10, 18)).mean('time')
        ABBD_B2 = ABBD.isel(time=slice(18, 26)).mean('time')
        ABBD_D = ABBD.isel(time=slice(26, 34)).mean('time')
        ABBD_gray = ABBD.isel(time=slice(34, None)).mean('time')

        ACBD_A = ACBD.isel(time=slice(2, 10)).mean('time')
        ACBD_C = ACBD.isel(time=slice(10, 18)).mean('time')
        ACBD_B = ACBD.isel(time=slice(18, 26)).mean('time')
        ACBD_D = ACBD.isel(time=slice(26, 34)).mean('time')
        ACBD_gray = ACBD.isel(time=slice(34, None)).mean('time')

        A = np.r_[ABCD_A, ABBD_A, ACBD_A]
        B = np.r_[ABCD_A, ABBD_B1, ABBD_B2, ACBD_B]
        C = np.r_[ABCD_C, ACBD_C]
        D = np.r_[ABCD_D, ABBD_D, ACBD_D]
        gray = np.r_[ABCD_gray, ABBD_gray, ACBD_gray]

        mu_A = A.mean()
        mu_B = B.mean()
        mu_C = C.mean()
        mu_D = D.mean()
        mu_gray = gray.mean()
        mu_stims = np.array([mu_A, mu_B, mu_C, mu_D, mu_gray])

        stim_names = ['A', 'B', 'C', 'D', 'gray']

        maxs = [
            ABCD.mean('trial').max('time'),
            ABBD.mean('trial').max('time'),
            ACBD.mean('trial').max('time'),
        ]
        m = np.max(maxs)
        names = []
        ratio = None
        for j in range(len(stim_names)):
            mu_i = mu_stims[j]
            others = np.r_[mu_stims[:j], mu_stims[j+1:]]
            thresh = others.mean() + 2 * others.std()
            r = float(mu_i - others.mean() / (mu_i + others.mean()))
            if mu_i >= thresh:
                ratio = mu_i / thresh
                if ratio > 1.5:
                    names.append(stim_names[j])
                # ratio = r

        if names:
            n_sig += 1
            string = f'roi {roi}: ' + ', '.join(names) + " - r: {:.2f}".format(ratio)
            print(string)
            roi_ids.append(roi)
            roi_stims.append(','.join(names))
        else:
            roi_ids.append(roi)
            roi_stims.append("")
            print(f'roi {roi}:       - r: {ratio}')

    print(n_sig / n_rois)
    df = pd.DataFrame({'stimulus': roi_stims}, index=roi_ids)
    return df


df = run(5)

# df = pd.read_csv('selectivity/cells_day0_scott.csv', index_col=0).fillna("")

