import json
from types import SimpleNamespace
from typing import Iterable, Mapping

import h5py
import matplotlib as mpl
# mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import statsmodels.api as sm
from types import SimpleNamespace

from ca_analysis.plot import *
from main import *
from selectivity import load_regression_table




def analysis(
    sessions: Iterable[Session],
    visual: Optional[bool] = None,
    verbose: bool = False,
):

    alpha = 0.05 / 5

    counts = {
        # 'gray': 0,
        'A': 0,
        'B': 0,
        'C': 0,
        'D': 0,
    }
    counts_1 = {
        # 'gray': 0,
        'A': 0,
        'B': 0,
        'C': 0,
        'D': 0,
    }

    n_rois_total = 0
    n_sig_counts = np.array([0, 0, 0, 0, 0], dtype=int)

    for s in sessions:

        savepath = s.fs.getsyspath('roi_info.h5')

        with h5py.File(savepath, 'r') as f:
            all_roi_ids = np.array(list(sorted(list(f.keys()))), dtype=int)
            visual_ids = np.load(s.fs.getsyspath('visual.npy'))
            non_visual_ids = np.setdiff1d(all_roi_ids, visual_ids)
            if visual is None:
                roi_ids = all_roi_ids
            elif visual is True:
                roi_ids = visual_ids
            else:
                roi_ids = non_visual_ids

            for roi_id in roi_ids:
                df = load_regression_table(s.fs.getsyspath('roi_info.h5'), roi_id)
                df = df.drop('constant')
                df = df.drop('gray')
                df = df[df['P'] < alpha]
                sig_names = set(df.index)
                n_sig_counts[len(df)] += 1
                if len(df) == 0:
                    continue

                for key in counts.keys():
                    if key in sig_names:
                        counts[key] += 1
                if len(df) == 1:
                    for key in counts.keys():
                        if key in sig_names:
                            counts_1[key] += 1
                n_rois_total += 1

    # print('finished')
    # index = ['gray', 'A', 'B', 'C', 'D']
    index = ['A', 'B', 'C', 'D']
    col_1 = np.array([counts[key] for key in index])
    col_2 = 100 * col_1 / n_rois_total
    col_3 = np.array([counts_1[key] for key in index])
    col_4 = 100 * col_3 / n_rois_total
    df = pd.DataFrame({
        'count': col_1,
        'count pct': col_2,
        'count (only 1)': col_3,
        'count (only 1) pct': col_4,
        },
        index=index,
    )
    if verbose:
        print(f'visual: {visual}')
        print(f'n_rois_total: {n_rois_total}')
        print(f'n_sig_counts: {n_sig_counts}')
        df = pd.DataFrame({
            'count': col_1,
            'count pct': col_2,
            'count (only 1)': col_3,
            'count (only 1) pct': col_4,
            },
            index=index,
        )
        print(df)
        print('')
    out = {
        'table': df,
        'n_rois_total': n_rois_total,
        'n_sig_counts': n_sig_counts,
    }

    return out


def get_summary_stats(
    sessions: Iterable[Session],
    visual: Optional[bool] = None,
) -> Mapping:

    # add 1 if a neuron responds significantly to a stimulus
    stim_counts = {
        'gray': 0,
        'A': 0,
        'B': 0,
        'C': 0,
        'D': 0,
    }

    # add 1 if a neuron responds significantly 'only' to a stimulus
    mono_stim_counts = {
        'gray': 0,
        'A': 0,
        'B': 0,
        'C': 0,
        'D': 0,
    }

    # add one for number of stimuli significant
    n_stim_counts = np.array([0, 0, 0, 0, 0, 0], dtype=int)

    n_rois_total = 0
    for s in sessions:
        savepath = s.fs.getsyspath('roi_info.h5')
        with h5py.File(savepath, 'r') as f:
            all_roi_ids = np.array(list(sorted(list(f.keys()))), dtype=int)
            visual_ids = np.load(s.fs.getsyspath('visual.npy'))
            non_visual_ids = np.setdiff1d(all_roi_ids, visual_ids)
            if visual is None:
                roi_ids = all_roi_ids
            elif visual is True:
                roi_ids = visual_ids
            else:
                roi_ids = non_visual_ids

            for roi_id in roi_ids:
                res_df = load_regression_table(s.fs.getsyspath('roi_info.h5'), roi_id)
                res_df = res_df.drop('constant')
                res_df = res_df[res_df['sig']]

                sig_names = list(res_df.index)

                # add to main counts
                for key in sig_names:
                    stim_counts[key] += 1

                # add to single-selective counts
                if len(sig_names) == 1:
                    mono_stim_counts[sig_names[0]] += 1

                # add to n-selectivity counts
                n_stim_counts[len(sig_names)] += 1

                n_rois_total += 1

    out = {
        'stim_counts': stim_counts,
        'mono_stim_counts': mono_stim_counts,
        'n_stim_counts': n_stim_counts,
        'n_rois': n_rois_total,
    }
    return out


def plot_num_stimuli_selectivity():
    plotdir = Path.home() / "plots/seq_learn_3"
    visual = None

    sessions_day_0 = get_sessions(day=0, fs=0)
    sessions_day_5 = get_sessions(day=5, fs=0)
    all_sessions = sessions_day_0 + sessions_day_5

    day_0 = get_summary_stats(sessions_day_0, visual=visual)
    day_5 = get_summary_stats(sessions_day_5, visual=visual)

    counts_0 = day_0['n_stim_counts']
    counts_0_frac = counts_0 / day_0['n_rois']
    a = counts_0_frac[1:]

    counts_5 = day_5['n_stim_counts']
    counts_5_frac = counts_5 / day_5['n_rois']
    b = counts_5_frac[1:]

    figsize = (6.5, 4)
    fig, ax = plt.subplots(figsize=figsize)
    X = np.arange(1, 6)
    width = 0.35
    ylim = [0, 1]

    ax.bar(X - width / 2, a, width, label='day 0')
    ax.bar(X + width / 2, b, width, label='day 5')

    ax.set_ylim(ylim)
    ax.legend()
    ax.set_ylabel('fraction of cells')
    ax.set_xlabel('num. stimuli')
    plt.show()

    path = plotdir / f'selectivity_num_stimuli_visual={visual}.png'
    fig.savefig(path)


def plot_stimulus_selectivity():
    plotdir = Path.home() / "plots/seq_learn_3"
    visual = None

    sessions_day_0 = get_sessions(day=0, fs=0)
    sessions_day_5 = get_sessions(day=5, fs=0)

    day_0 = get_summary_stats(sessions_day_0, visual=visual)
    day_5 = get_summary_stats(sessions_day_5, visual=visual)

    stim_names = ['gray', 'A', 'B', 'C', 'D']

    counts_0 = day_0['stim_counts']
    n_rois_0 = day_0['n_rois']
    frac_0 = [counts_0[key] / n_rois_0 for key in stim_names]

    counts_5 = day_5['stim_counts']
    n_rois_5 = day_5['n_rois']
    frac_5 = [counts_5[key] / n_rois_5 for key in stim_names]

    figsize = (6.5, 4)
    fig, ax = plt.subplots(figsize=figsize)
    X = np.arange(1, 6)
    width = 0.35
    ylim = [0, 1]

    ax.bar(X - width / 2, frac_0, width, label='day 0')
    ax.bar(X + width / 2, frac_5, width, label='day 5')
    ax.set_xticks(X)
    ax.set_xticklabels(stim_names)
    # ax.set_title('all ROIs')
    ax.set_ylim(ylim)
    ax.legend()
    ax.set_ylabel('fraction of cells')
    ax.set_xlabel('stimulus')
    plt.show()

    path = plotdir / f'selectivity_stimuli={visual}.png'
    fig.savefig(path)


"""
"""

sessions_day_0 = get_sessions(day=0, fs=0)
sessions_day_5 = get_sessions(day=5, fs=0)

sessions = sessions_day_0
