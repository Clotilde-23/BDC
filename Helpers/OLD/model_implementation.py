#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 09:11:01 2023

@author: cloclo
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from preprocessing import split_temporel_V1
from score import compute_scores

def model_OLS(data, feature_geo, feature_temp, feature_autre, output) : 
    
        # Filtrer les variables pour le modèle
    feature_geo = feature_geo
    feature_temp = feature_temp
    features_OLS_hors_geotemp = feature_autre
    label = 'Prix_m2'
    
        # Split des données en TRAIN et TEST
    features_OLS = features_OLS_hors_geotemp + feature_geo + feature_temp
    X_train, X_test, y_train, y_test = split_temporel_V1(data, features_OLS, label)
    
        # Ajout des constantes au modèle
    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)
    
    model_OLS = sm.OLS(y_train, X_train).fit()
    
    print(model_OLS.summary())
    print('\n Score :')
    
    compute_scores(model_OLS, X_test, y_test)

