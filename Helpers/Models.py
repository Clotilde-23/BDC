import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

    # OLS
import statsmodels.api as sm
    # ML Models
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

    # Pipeline & Preprocess
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV

    # Scores
from sklearn.metrics import balanced_accuracy_score, make_scorer, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error , r2_score, mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error



# ---------------   OLS    --------------------

def model_OLS_prix(data, outcome, features, summary = True) : 
    X = data[features]
    Y = data[outcome]

    X = sm.add_constant(X)

    model = sm.OLS(Y, X).fit()
    
    if summary : 
        print(model.summary())
    
    return(model)

def indice_prix_quarter(model_ols_prix, plot = True) : 
    indices = model_ols_prix.params[:15]
    indices = indices[0] + indices
    indices[0] = indices[0]/2
    index = indices.index[1:].insert(0, 'quarter_2017_Q3')
    indices.index = index
    indices

    indices_plot = indices.copy()
    indices_plot.index = indices_plot.index.str[8:12]+'-'+indices_plot.index.str[13:]
    indices_plot.index = pd.to_datetime(indices_plot.index)
    indices.index = indices.index.str[8:12]+'_'+indices.index.str[13:]
    
    if plot : 
        indices_plot.plot()
    
    return(indices)

def add_indice_prix(data_train, indices) :
    # Growth
    growth = indices.pct_change()
    growth[0] = 0
    
    growth_cum = growth.copy()
    growth_cum[1] = 1+growth_cum[1]
    for i in range(1,len(growth)-1) : 
        growth_cum[i+1] = (growth_cum[i])*(1+growth[i+1])
    growth_cum[0] = 1
    
    df_growth = pd.DataFrame(growth_cum).reset_index()
    df_growth.columns  = ['quarter', 'pct_change']
    
    df_growth['pct_change_toQ1_2021'] = df_growth.iloc[14,1]-df_growth['pct_change']
    
    # Merge with data
    data = data_train.copy()
    data = data.merge(df_growth, on = 'quarter', how = 'left')
    data['Prix_m2_actualise_Q1_2021'] = data['Prix_m2']*(1+data['pct_change_toQ1_2021'])
    
    return(data)

# -------------    KNN   ------------------ #

def model_KNN_coordinates(df_train, df_test, features,
                          label_train, label_test,
                          standardization, type_weights, max_nn, nb_cv) :
        # X datasets
    X_train_knn = df_train[features]
    X_test_knn = df_test[features]
        # outcomes
    y_train = df_train[label_train]
    y_test = df_test[label_test]

        # pipeline for the model
    pipe = Pipeline([('scaler', standardization), ('Knn', KNeighborsRegressor(weights=type_weights))])
    parameters = {'Knn__n_neighbors': range(1, max_nn, 2)} 
    knn_pipe = GridSearchCV(pipe, parameters, cv=nb_cv)
    knn_pipe.fit(X_train_knn, y_train)

    print('Returned hyperparameter: {}'.format(knn_pipe.best_params_))

    return(knn_pipe)

def model_RF_post_KNN(df_test, df_train, features_RF, features_KNN, label_train, label_test,
                      model_knn, standardisation, range_depth, nb_cv) :
    X_train_rf = df_train[features_RF]
    X_train_rf['y_pred_knn'] = model_knn.predict(df_train[features_KNN])
    X_test_rf = df_test[features_RF]
    X_test_rf['y_pred_knn'] = model_knn.predict(df_test[features_KNN])
        # outcomes
    y_train = df_train[label_train]
    y_test = df_test[label_test]

        # Model
    pipe = Pipeline([('scaler', standardisation), ('RForest', RandomForestRegressor())])
    parameters = {'RForest__max_depth': range_depth} # defining parameter space
    rforest_pipe = GridSearchCV(pipe, parameters, cv=nb_cv)
    rforest_pipe.fit(X_train_rf, y_train)

    return(rforest_pipe)


def features_importances(model_importances, X_train_model) :
    #X_train_model = df_train[features_model]
    indices = np.argsort(model_importances)[::-1]
    columns = X_train_model.columns
    # plot
    nb_features = X_train_model.shape[1]
    plt.figure(figsize=(15, 8))
    plt.title("Feature importances")
    plt.barh(
        range(nb_features),
        model_importances[indices],
        color='b')
    plt.yticks(range(nb_features), columns[indices], rotation='horizontal', size=10)
    plt.show()
    
   

# -------------   SCORES   ---------------- #

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
