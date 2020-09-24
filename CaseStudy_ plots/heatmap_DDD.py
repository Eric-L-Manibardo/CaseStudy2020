# =============================================================================
# PLOTTER CD PLOT 
# =============================================================================

# Libraries

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()
from Orange.evaluation import compute_CD, graph_ranks
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

np.random.seed = 0

# Input

N = 16 # Models
M = 40 # Datasets

# nameModels = ["model-"+str(n) for n in range(N)]
# nameModels = ['LV', 'KNN', 'SVR','RFR', 'ETR', 'GBR','XGBR','FNN', 'CNN', 'LSTM', 'CNNLSTM', 'ATT']
nameModels = ['LV','LR', 'KNN', 'DTR', 'ELM', 'SVR','ADA', 'RFR', 'ETR', 'GBR','XGBR','FNN', 'CNN', 'LSTM', 'CLSTM', 'ATT']

# results1 = np.random.rand(M,N) #idealmente, leer de un fichero
df = pd.read_csv('Overall.csv') # leo fichero
df = df.iloc[:,1:] # selecciono solo datos


#plot heatmap
y_axis_labels = ['Madrid H1','Madrid H2','Madrid H3','Madrid H4',
                 'California H1','California H2','California H3','California H4',
                 'New York H1','New York H2','New York H3','New York H4',
                 'Seattle H1','Seattle H2','Seattle H3','Seattle H4'] 

ax = sns.heatmap(df,vmin=0.6, vmax=1, annot=True,cmap='YlGnBu',
                 yticklabels=y_axis_labels,cbar=False,
                 square=True)

ax.set_xticklabels(ax.get_xticklabels(), rotation = 0)

# split axes of heatmap to put colorbar
ax_divider = make_axes_locatable(ax)
# define size and padding of axes for colorbar
cax = ax_divider.append_axes('top', size = '5%', pad = '2%')
# make colorbar for heatmap. 
# Heatmap returns an axes obj but you need to get a mappable obj (get_children)
colorbar(ax.get_children()[0], cax = cax, orientation = 'horizontal')
# locate colorbar ticks
cax.xaxis.set_ticks_position('top')



bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)




plt.show()