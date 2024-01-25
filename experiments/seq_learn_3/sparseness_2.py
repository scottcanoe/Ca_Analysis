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

day = 0

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

    ABCD_A = ABCD.isel(time=slice(2, 8)).mean('time')
    ABCD_B = ABCD.isel(time=slice(10, 16)).mean('time')
    ABCD_C = ABCD.isel(time=slice(18, 24)).mean('time')
    ABCD_D = ABCD.isel(time=slice(26, 32)).mean('time')
    ABCD_gray = ABCD.isel(time=slice(34, None)).mean('time')

    ABBD_A = ABBD.isel(time=slice(2, 8)).mean('time')
    ABBD_B1 = ABBD.isel(time=slice(10, 16)).mean('time')
    ABBD_B2 = ABBD.isel(time=slice(18, 24)).mean('time')
    ABBD_D = ABBD.isel(time=slice(26, 32)).mean('time')
    ABBD_gray = ABBD.isel(time=slice(34, None)).mean('time')

    ACBD_A = ACBD.isel(time=slice(2, 8)).mean('time')
    ACBD_C = ACBD.isel(time=slice(10, 16)).mean('time')
    ACBD_B = ACBD.isel(time=slice(18, 24)).mean('time')
    ACBD_D = ACBD.isel(time=slice(26, 32)).mean('time')
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
    names = []
    for j in range(len(stim_names)):
        mu_i = mu_stims[j]
        others = np.r_[mu_stims[:j], mu_stims[j+1:]]
        thresh = others.mean() + 2 * others.std()
        if mu_i >= thresh:
            names.append(stim_names[j])
    if names:
        n_sig += 1
        string = f'roi {roi}: ' + ', '.join(names)
        print(string)
        roi_ids.append(roi)
        roi_stims.append(','.join(names))

    # std = mu_stims.std()
    # mu = float(np.mean([ABCD.mean(), ABBD.mean(), ACBD.mean()]))
    # # mu = mu_stims.mean()
    # # mu = baselines[roi]
    # # std = stds[roi]
    # thresh = mu + 2 * std
    #
    # tf = mu_stims >= thresh
    # sig = bool(tf.sum())
    # if sig:
    #     n_sig += 1
    # A_tf = mu_A > thresh
    # B_tf = mu_B > thresh
    # C_tf = mu_C > thresh
    # D_tf = mu_D > thresh
    # gray_tf = mu_gray > thresh
    # print(f'roi {roi} -- A: {A_tf}, B: {B_tf}, C: {C_tf}, D: {D_tf}, gray: {gray_tf}')

print(n_sig / n_rois)
df = pd.DataFrame({'stimulus': roi_stims}, index=roi_ids)
