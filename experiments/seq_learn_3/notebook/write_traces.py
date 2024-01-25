import h5py
from matplotlib.figure import Figure
import numpy as np

from ca_analysis.plot import *
from experiments.seq_learn_3.main import *
from experiments.seq_learn_3.utils import *



# drop_gray = False
#
# # set up splits
# sessions = get_sessions(day=0, fs=0)
# ABCD_0 = flex_split(sessions, 'ABCD', drop_gray=drop_gray)
# ABBD_0 = flex_split(sessions, 'ABBD', drop_gray=drop_gray)
# ACBD_0 = flex_split(sessions, 'ACBD', drop_gray=drop_gray)
#
# # set up splits
# sessions = get_sessions(day=5, fs=0)
# ABCD_5 = flex_split(sessions, 'ABCD', drop_gray=drop_gray)
# ABBD_5 = flex_split(sessions, 'ABBD', drop_gray=drop_gray)
# ACBD_5 = flex_split(sessions, 'ACBD', drop_gray=drop_gray)
#
# with h5py.File('traces.h5', 'w') as f:
#     group = f.create_group('day_0')
#     group.create_dataset('ABCD', data=ABCD_0.data)
#     group.create_dataset('ABBD', data=ABBD_0.data)
#     group.create_dataset('ACBD', data=ACBD_0.data)
#     group = f.create_group('day_5')
#     group.create_dataset('ABCD', data=ABCD_5.data)
#     group.create_dataset('ABBD', data=ABBD_5.data)
#     group.create_dataset('ACBD', data=ACBD_5.data)
#
#

