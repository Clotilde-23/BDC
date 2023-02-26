#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 09:11:01 2023
"""


import pandas as pd
import numpy as np
import datetime, warnings, scipy 
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV

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
    data['quarter'] = data['Year'].astype(str) + '_Q' + data['num_trimestre'].astype(str)


# Filtre pour garder les données pertinentes pour le modèle
def filtre_data_pour_model(data, ville, type_local, quantile_low = None, quantile_high = None) :
    '''
    data : DataFrame (celui a nettoyer)
    ville : str ('Paris', 'Lyon', ...)
    type_local : int (1 : Maison / 2 : Appartement)
    quantile_low : int (\in [0, 1])
    quantile_high : int (\in [0, 1])

    output : DataFrame (nettoyé)
    '''
    data_model = data.copy()
    
    max_prix = np.quantile(data.Prix_m2, quantile_high)
    min_prix = np.quantile(data.Prix_m2, quantile_low)
    
    if quantile_low : 
        data_model = data_model[data_model.Prix_m2 > min_prix]
    if quantile_high : 
        data_model = data_model[data_model.Prix_m2 < max_prix]
        
    data_model = data_model[(data_model['bv2012_name'] == "['" + ville + "']")
                     & (data_model.code_type_local == type_local)]
    
    return(data_model)

# Premier split temporel avec toutes la bases de données,
# Les 20 derniers pourcent de la base sont dans l'éch de test
# Le reste sert d'apprentissage
def split_temporel_V1(data, liste_features, output) :
    '''
    data : DataFrame (sur lequel on veut faire le split
    liste_features : list (des variables explicatives)
    output : str

    output : 4 DataFrame (X_train, X_test, y_train, y_test)
    '''
    # Trier la base par date
    data = data.sort_values('date_mutation')

    # Choix les variables du modèles
    X = data[liste_features]
    y = data[output]
    
    # Shuffle = False pour garder les dernières obs en test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)
    return(X_train, X_test, y_train, y_test)
