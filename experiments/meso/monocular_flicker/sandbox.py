# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from experiments.meso.roi_processing import *
from experiments.meso.seq_learn_3.utils import *


s = open_session('M110', '2023-04-21', '2')
s.pull(exclude_dirs=['raw'])

s = open_session('M115', '2023-04-21', '2')
s.pull(exclude_dirs=['raw'])
