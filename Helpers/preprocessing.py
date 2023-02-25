#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 09:11:01 2023

@author: cloclo
"""
import pandas as pd
import numpy as np
import datetime, warnings, scipy 
import seaborn as sns

import matplotlib.pyplot as plt

# Mettre les variables de la liste_var en log dans le dataframe data
def log_var(data, liste_var) : 
    for variable in liste_var :
        new_variable = variable + '_log'
        data[new_variable] = np.log(data[variable])
        
# Créer une variable de la forme YEAR-Qi
def annee_trimestre(data) : 
        # Mettre le format date
    data['date_mutation'] = pd.to_datetime(data['date_mutation'], format = '%Y-%m-%d')
        # num_trimestre indique le trimestre de la date mutation ie 1, 2, 3 ou 4
    data['num_trimestre'] = data['date_mutation'].dt.quarter
        # Concaténation sous la forme YEAR-Qi
    data['quarter'] = data['Year'].astype(str) + '_Q' + data['trimestre'].astype(str)
