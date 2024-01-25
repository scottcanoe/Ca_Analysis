
from typing import (
    Optional,
    Union,
)
from cycler import cycler
import matplotlib as mpl
from matplotlib.axes import Axes
import numpy as np

from ..common import *

__all__ = [
    "CB_color_cycler",
    "annotate_onsets",
    "remove_ticks",
]


CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
CB_color_cycler = cycler(color=CB_color_cycle)


def annotate_onsets(
    ax: Axes,
    events: ArrayLike,
    symbol: str = r'$\Delta$',
    skip_first: bool = False,
    last: Optional[str] = None,
    vline: bool = True,
    shift: Number = 0,
    **kw,
) -> None:
    """
    Put onset indicators where labels change. Optionally, add a vertical
    line, usually for heatmaps or traces.

    Parameters
    ----------
    ax

    Returns
    -------

    """

    events = np.asarray(events)

    # Add onset indicators.
    xticks = []
    if not skip_first:
        xticks.append(0 + shift)
    for i in range(1, len(events)):
        if events[i] != events[i - 1]:
            xticks.append(i + shift)
    xticklabels = [symbol] * len(xticks)
    if last:
        if last is True:
            xticklabels[-1] = symbol
        else:
            xticklabels[-1] = last

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    # add vertical lines
    if vline:
        kw = dict(kw)
        kw['color'] = kw.get('color', kw.get('c', 'gray'))
        kw['linestyle'] = kw.get('linestyle', kw.get('ls', '--'))
        kw['linewidth'] = kw.get('linewidth', kw.get('lw', 1))
        kw['alpha'] = kw.get('alpha', 1)
        for x in xticks:
            ax.axvline(x, **kw)



def remove_ticks(ax: "Axes") -> "Axes":
    """
    Remove all ticks and tick labels from axes. Usually for images.

    Parameters
    ----------
    ax

    Returns
    -------

    """
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    return ax
