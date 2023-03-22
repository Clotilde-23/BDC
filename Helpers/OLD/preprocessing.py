#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 09:11:01 2023
Modified on Mon Feb 27 @LouiseBonhomme -> @split_appart_maison(), @ztransform(), @clean_iris_codes()

@author: cloclo
"""
import pandas as pd
import numpy as np
import datetime, warnings, scipy 
import seaborn as sns
from sklearn.preprocessing import StandardScaler

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
    data['quarter'] = data['Year'].astype(str) + '_Q' + data['trimestre'].astype(str)

    
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
