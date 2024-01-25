from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from ca_analysis.plot import *
from experiments.seq_learn_3.main import *
from experiments.seq_learn_3.utils import *

df0 = pd.read_csv('selectivity/cells_day0.tsv', sep='\t', index_col=0)

df5 = pd.read_csv('selectivity/cells_day5.tsv', sep='\t', index_col=0)
