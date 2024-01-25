import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
from ca_analysis.plot import *

from seq_learn_3.main import *
from seq_learn_3.utils import *


roi_filter = 'all'

sessions_0 = get_sessions(day=0)
sessions_5 = get_sessions(day=5)
apply_roi_filter(sessions_0 + sessions_5, roi_filter)

ABCD_0 = flex_split(sessions_0, 'ABCD')
# ABBD_0 = flex_split(sessions_0, 'ABBD')
# ACBD_0 = flex_split(sessions_0, 'ACBD')
#
# ABCD_5 = flex_split(sessions_5, 'ABCD')
# ABBD_5 = flex_split(sessions_5, 'ABBD')
# ACBD_5 = flex_split(sessions_5, 'ACBD')

front = ABCD_0.isel(trial=slice(200, 300))
back = ABCD_0.isel(trial=slice(300, 400))
a = front.mean('time').mean('trial')
b = back.mean('time').mean('trial')
diff = b - a
bins = np.linspace(-5.5, 5.5, 20)
# plt.hist(diff, bins=bins)
# plt.show()
