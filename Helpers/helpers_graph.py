#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 09:17:51 2023

@author: cloclo
"""
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

def map_data(data, year, variable) : 
    '''
    Plot the map of France
    '''
    plt.figure(figsize=(16,16))
    m1 = Basemap(llcrnrlon= -5, llcrnrlat=41,
                 urcrnrlon=13, urcrnrlat=52, 
                 lat_0=46.2374, lon_0=2.375,
                 resolution = 'h', projection = 'lcc')

    m1.drawcoastlines(color = 'black')
    m1.drawcountries(linewidth = 3)

    x, y = m1(list(df_processed_total[df_processed_total['Year'] == year]['longitude']), 
              list(df_processed_total[df_processed_total['Year'] == year]['latitude']))
    m1.scatter(x, y ,
              c = df_processed_total[df_processed_total['Year'] == year][variable],
              s = 8,
              cmap = 'viridis')

    plt.title('Price of meter_square for', str(year), 'in France' , fontsize = 24)
    plt.colorbar()
    plt.clim(0,10000)
    plt.show()