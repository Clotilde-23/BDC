#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 09:11:01 2023
Modified on Mon Feb 27 @LouiseBonhomme -> @split_appart_maison(), @ztransform(), @clean_iris_codes()
"""


import pandas as pd
import numpy as np
import datetime, warnings, scipy 
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV

import matplotlib.pyplot as plt

# Mettre les variables de la liste_var en log dans le dataframe data
def log_var(data, liste_vars) : 
    for variable in liste_vars :
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
    
    
    if quantile_low :
        min_prix = np.quantile(data.Prix_m2, quantile_low)
        data_model = data_model[data_model.Prix_m2 > min_prix]
    if quantile_high :
        max_prix = np.quantile(data.Prix_m2, quantile_high)
        data_model = data_model[data_model.Prix_m2 < max_prix]
        
    data_model = data_model[(data_model['bv2012_name'] == "['" + ville + "']")
                     & (data_model.code_type_local == type_local)]
    
    return(data_model)

def filter_quantile(data, var, quantile_low, quantile_high) :
    '''
    data : DataFrame (celui a nettoyer)
    ville : str ('Paris', 'Lyon', ...)
    type_local : int (1 : Maison / 2 : Appartement)
    quantile_low : int (\in [0, 1])
    quantile_high : int (\in [0, 1])
    output : DataFrame (nettoyé)
    '''
    data_model = data.copy()
    
    if quantile_low :
        min_prix = np.quantile(data[var], quantile_low)
        data_model = data_model[data_model[var] > min_prix]
    if quantile_high :
        max_prix = np.quantile(data[var], quantile_high)
        data_model = data_model[data_model[var] < max_prix]
    
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

    
#Split le dataset entre appartements et maisons
def split_appart_maison(data): 
    df_maison = data[data['code_type_local']==1]
    df_appart = data[data['code_type_local']==2]
    print(f"Nombre de maisons : {df_maison.shape[0]}")
    print(f"Nombre d'apparts : {df_appart.shape[0]}")
    return(df_maison, df_appart)

def zTransform (data, liste_vars, prix) : 
    scaler = StandardScaler()
    liste_vars.append(prix)
    scaler.fit(data[liste_vars])
    print(scaler.mean_)
    data[liste_vars] = scaler.transform(data[liste_vars])
    
def clean_iris_codes(data) : 
    data.drop(['iris_name_u', 'iris_name_l', 'iris_area_c',
       'iris_type', 'iris_grd_qu', 'iris_in_ctu', 'reg_name', 'dep_name','arrdep_code',
       'bv2012_code', 'bv2012_name', 'epci_code', 'epci_name', 'com_code', 'com_name', 'com_arm_cod', 
         'com_arm_nam', 'index_right', 'year', 'reg_code', 'dep_code'], axis = 1, inplace = True)

    data['iris_code'] = data['iris_code'].str[2:11]
    data['iris_name'] = data['iris_name'].str[2:-3]
    data.dropna(inplace = True)

def nb_iris (data, path, dep_code):
    IRIS = pd.read_excel(path, header = 5)[['CODE_IRIS', 'DEP']]
    IRIS_ville = IRIS[IRIS["DEP"] == dep_code]
    IRIS_ville = IRIS_ville['CODE_IRIS'].tolist()
    iris_df = data['iris_code'].unique().tolist()

    list_dif = [i for i in IRIS_ville + iris_df if i in IRIS_ville and i not in iris_df]
    print(f"Length of all Paris IRIS : {len(IRIS_ville)}")
    print(f"Length of IRIS in DF : {len(iris_df)}")
    print(f"Number of IRIS not in DF : {len(list_dif)}")

# Ajouter des indicatrices si la variable continue dépasse un seuil
def dummies_pr_var_continues(data, var, seuil) :
    '''
    data : DataFrame
    var : str
    seuil : int / float
    output : None
    '''
    var_dummy = var + '_dummy'
    data[var_dummy] = 0
    data.loc[data[var] > seuil, var_dummy] = 1
