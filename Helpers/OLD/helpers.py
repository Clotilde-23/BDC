#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 09:17:51 2023

@author: cloclo
"""

def filling_factor(data) : 
    '''
    Output : Table des pourcentages de remplissage des variables
    '''
    missing_df = data.isnull().sum(axis=0).reset_index()
    missing_df.columns = ['variable', 'missing values']
    missing_df['filling factor (%)']=(data.shape[0]-missing_df['missing values'])/data.shape[0]*100
    missing_df.sort_values('filling factor (%)').reset_index(drop = True)
    return(missing_df)

def calc_vif(X):
    '''
    Output : Table des VIF des variables
    '''
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) 
    for i in range(X.shape[1])]
    return(vif)