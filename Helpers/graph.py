#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 09:17:51 2023

@author: cloclo
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def visualise(df, vmin, vmax, ville, type_local, quantile_high, quantile_low):
    
    max_prix = np.quantile(df.Prix_m2, quantile_high)
    min_prix = np.quantile(df.Prix_m2, quantile_low)
    
    df_to_visualise = df[(df['bv2012_name'] == "['" + ville + "']")
                        & (df.Prix_m2 < max_prix)
                        & (df.Prix_m2 > min_prix)
                        & (df.code_type_local == type_local)]
    
    df_sorted = df_to_visualise.sort_values(by='Prix_m2')
    x = df_sorted['longitude']
    y = df_sorted['latitude']
    c = df_sorted['Prix_m2'] 

    plt.rcParams['figure.figsize'] = [7, 5]
    plt.rcParams['figure.dpi'] = 100 

    plt.scatter(x, y, s=0.01, c=c, cmap='plasma_r', alpha=0.8)
    plt.colorbar()
    plt.show()
