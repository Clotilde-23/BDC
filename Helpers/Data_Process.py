import pandas as pd
import numpy as np
import datetime, warnings, scipy 
from sklearn.preprocessing import StandardScaler

def Process_data(data, quantile_low, quantile_high):    
    
    #we remove the extreme 5 % values of price per meter_square
    Threshold_up = data.groupby(['iris_code','quarter'])['Prix_m2'].quantile(quantile_high).reset_index()
    Threshold_down = data.groupby(['iris_code','quarter'])['Prix_m2'].quantile(quantile_low).reset_index()

    df_comune ={'iris_code':Threshold_up['iris_code'], 'Threshold_up_iris': Threshold_up['Prix_m2'] , 
            'Threshold_down_iris':Threshold_down['Prix_m2'] , 'quarter': Threshold_down['quarter']}
    df_comune = pd.DataFrame(df_comune)

    df_process = pd.merge(data, df_comune ,how='inner', on=['iris_code','quarter'])

    df_final = df_process[(df_process['Prix_m2'] < df_process['Threshold_up_iris']) 
                      & (df_process['Prix_m2'] > df_process['Threshold_down_iris'])]
    
    df_final=df_final.drop(['Threshold_up_iris','Threshold_down_iris'],axis = 1)
    
    return df_final

def Process_data_2(data):    
    
    #we remove the extreme 5 % values of price per meter_square
    data['Threshold_up'] = data['Pris_m2_moy_iris'] + data['std']
    data['Threshold_down'] = data['Pris_m2_moy_iris'] - data['std']


    df_final = data[(data['Prix_m2'] < data['Threshold_up']) 
                      & (data['Prix_m2'] > data['Threshold_down'])]
    
    df_final=df_final.drop(['Threshold_up','Threshold_down'],axis = 1)
    
    return df_final


# Mettre les variables de la liste_var en log dans le dataframe data
def log_var(data, liste_vars) : 
    for variable in liste_vars :
        new_variable = variable + '_log'
        data[new_variable] = np.log(data[variable])
        
    

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
