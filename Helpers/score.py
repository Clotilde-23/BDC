#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 09:17:51 2023

@author: cloclo
"""
import numpy as np

def MAPE(Y_actual, Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape

def MDAPE(Y_actual, Y_Predicted):
    mdape = np.median(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mdape

def compute_scores(model, X_test, y_test) :
    y_pred = model.predict(X_test)

    mape = MAPE(y_test, y_pred)
    mdape = MDAPE(y_test, y_pred)
    
    print("MAPE: ", mape)
    print("MDAPE: ", mdape)
