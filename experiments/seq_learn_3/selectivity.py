import json

import h5py
import numpy as np
from scipy.stats import ks_2samp
import statsmodels.api as sm

from main import *


def create_roi_filters(s: Session) -> None:
    """
    Find which ROIs fire significantly differently in response to
    stimuli or gray screen.

    Parameters
    ----------
    s

    Returns
    -------

    """
    logger.info(f"Creating roi filters for {s}")

    # get spike data for all cells
    spikes = s.spikes.data

    """
    Extract gray/non-gray indices, and regroup into 8-length chunks. Then throw
    away the last 4 indices for each 8-length chunk. Skipping every four-length
    chunk will help cut down on correlated samples.

    """

    def pull_data(inds: ArrayLike) -> ArrayLike:
        major_chunk, minor_chunk = 8, 4
        n, rem = divmod(len(inds), major_chunk)
        if rem > 0:
            inds = inds[:-rem]
        inds = np.reshape(inds, [n, major_chunk])[:, :minor_chunk]

        samples = []
        for j in range(inds.shape[0]):
            obj = spikes.isel(time=inds[j]).mean("time")
            samples.append(obj)
        data = xr.DataArray(np.vstack(samples).T, dims=('roi', 'sample'))
        data.coords['roi'] = spikes.coords['roi']
        return data

    frames = s.events["frames"]

    gray_inds = frames[frames['stimulus'] == 0].index
    gray_data = pull_data(gray_inds)

    stim_inds = frames[frames['stimulus'] > 0].index
    stim_data = pull_data(stim_inds)

    """
    Loop through each ROI, and perform a ks-test between gray and non-gray data.
    """

    roi_ids = spikes.coords['roi'].data
    grating_ids = []
    gray_ids = []
    visual_ids = []
    non_visual_ids = []
    for roi in roi_ids:
        gr = gray_data.sel(roi=roi)
        st = stim_data.sel(roi=roi)
        stat = ks_2samp(gr, st)
        if stat.pvalue < 0.05:
            visual_ids.append(roi)
            if st.mean() > gr.mean():
                grating_ids.append(roi)
            if st.mean() < gr.mean():
                gray_ids.append(roi)
        else:
            non_visual_ids.append(roi)
    grating_ids = np.array(grating_ids, dtype=int)
    gray_ids = np.array(gray_ids, dtype=int)
    visual_ids = np.array(visual_ids, dtype=int)
    non_visual_ids = np.array(non_visual_ids, dtype=int)

    n_rois = len(roi_ids)
    n_visual = len(visual_ids)
    pct_visual = 100 * n_visual / n_rois
    logger.info('{}/{} ({:.2f}%) visually modulated'.format(
        n_visual, n_rois, pct_visual)
    )

    with h5py.File(s.fs.getsyspath('roi_filters.h5'), 'w') as f:
        f.create_dataset('all', data=roi_ids)
        f.create_dataset('gratings', data=grating_ids)
        f.create_dataset('gray', data=gray_ids)
        f.create_dataset('visual', data=visual_ids)
        f.create_dataset('non_visual', data=non_visual_ids)


def compute_stimulus_selectivity(
    s: Session,
    savepath: PathLike = "roi_info.h5",
) -> None:


    logger.info(f"Computing stimulus selectivity for {s}")

    ev_df = s.events['events']
    spikes = s.spikes.data

    in_data = np.zeros([len(ev_df), spikes.sizes['roi']])
    for i in range(len(ev_df)):
        row = ev_df.iloc[i]
        start, stop = row.start, row.stop
        chunk = spikes.isel(time=slice(start, stop))
        in_data[i] = chunk.mean('time')
    in_data = xr.DataArray(in_data, dims=('time', 'roi'))
    in_data.coords['roi'] = spikes.coords['roi']

    # one-hot encode event stimuli
    stim_table = s.events.schema.tables['stimuli']
    stim_ids = np.array(stim_table.index)
    stim_names = np.array(stim_table.name)
    stim_names[stim_names == ''] = 'gray'

    stim_vec = ev_df['stimulus'].values
    exog_data = {}
    for id, name in zip(stim_ids, stim_names):
        vec = np.zeros(len(ev_df), dtype=int)
        vec[stim_vec == id] = 1
        exog_data[name] = vec
    exog = pd.DataFrame(exog_data)
    exog['constant'] = 1

    n_rois = in_data.sizes['roi']
    for i in range(n_rois):
        endog = in_data.isel(roi=i)
        roi_id = endog.coords['roi'].item()
        model = sm.OLS(endog, exog)
        res = model.fit()

        res_df_data = {}
        res_df_index = res.params.index
        res_df_data['coef'] = res.params.values
        res_df_data['std_err'] = res.HC1_se
        res_df_data['t'] = res.tvalues
        res_df_data['P'] = res.pvalues
        sig = res.pvalues <= 0.05 / 5
        sig[res.params.values <= 0] = False
        res_df_data['sig'] = sig
        res_df = pd.DataFrame(res_df_data, index=res_df_index)

        save_regression_table(s.fs.getsyspath(savepath), roi_id, res_df)
        if i % 10 == 0:
            logger.debug(f'{i} / {n_rois}')


def save_regression_table(
    path: PathLike,
    key: str,
    df: pd.DataFrame,
    mode: str = "a",
) -> None:

    key = str(key)
    with h5py.File(path, mode) as f:
        if key in f:
            del f[key]
        g = f.create_group(key)
        g.create_dataset(
            '__index__', data=df.index.values, dtype=h5py.string_dtype()
        )
        for col_name in df.columns:
            g.create_dataset(col_name, data=df[col_name].values)
        g.attrs['columns'] = json.dumps(list(df.columns))


def load_regression_table(
    path: PathLike,
    key: str,
) -> pd.DataFrame:

    key = str(key)
    with h5py.File(path, 'r') as f:
        g = f[key]
        index = g['__index__'][:].astype(str)
        columns = json.loads(g.attrs['columns'])
        data = {}
        for col_name in columns:
            data[col_name] = g[col_name][:]
        out = pd.DataFrame(data, index=index)
        return out


if __name__ == "__main__":

    for s in get_sessions(day=0, fs=0):
        create_roi_filters(s)

    for s in get_sessions(day=5, fs=0):
        create_roi_filters(s)
