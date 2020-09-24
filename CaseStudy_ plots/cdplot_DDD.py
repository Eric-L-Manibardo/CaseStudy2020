# =============================================================================
# PLOTTER CD PLOT 
# =============================================================================

# Libraries

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from Orange.evaluation import compute_CD, graph_ranks

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

np.random.seed = 0

# Input

N = 16 # Models
M = 40 # Datasets

# nameModels = ["model-"+str(n) for n in range(N)]
# nameModels = ['LV', 'KNN', 'SVR','RFR', 'ETR', 'GBR','XGBR','FNN', 'CNN', 'LSTM', 'CNNLSTM', 'ATT']
nameModels = ['LV','LR', 'KNN', 'DTR', 'ELM', 'SVR','ADA', 'RFR', 'ETR', 'GBR','XGBR','FNN', 'CNN', 'LSTM', 'CNNLSTM', 'ATT']

# results1 = np.random.rand(M,N) #idealmente, leer de un fichero
df = pd.read_csv('CD_data - t+4.csv') # leo fichero
df = df.iloc[:,1:] # selecciono solo datos
df = df.stack().str.replace(',','.').unstack() # cambio delimitador coma por punto
results = df.to_numpy() # paso a array de objetos
results = np.vstack(results).astype(np.float) # paso a array de float
# Processing

order = np.argsort(-results,axis=1)
ranks = np.argsort(order,axis=1)+1
avgranks = np.mean(ranks,axis=0)

CD = compute_CD(avgranks, M)
graph_ranks(avgranks, nameModels, cd=CD, width=6, textspace=1.5)
plt.show()







