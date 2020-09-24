#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 13:47:56 2020

@author: eric
"""

import plotly.express as px
from plotly.offline import plot
import pandas as pd

# data = dict(
#     character=["Eve", "Cain", "Seth", "Enos", "Noam", "Abel", "Awan", "Enoch", "Azura"],
#     parent=["", "Eve", "Eve", "Seth", "Seth", "Eve", "Eve", "Awan", "Eve" ],
#     value=[10, 14, 12, 10, 2, 6, 6, 4, 4])

# fig =px.sunburst(
#     data,
#     names='character',
#     parents='parent',
#     values='value',
# )
# plot(fig) # to print an interactive version at browser
# fig.write_image("prueba.pdf")

df = pd.read_csv('sunburst.csv')
fig = px.sunburst(df, path=['Relationships', 'Input', 'Coverage', 'Horizon'], values='unit',
                   color='Input'
                  )
fig.write_image("sunburst92.pdf")
# l=px.data.gapminder().query("year == 2007")
plot(fig)