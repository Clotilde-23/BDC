import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport

import datetime, warnings, scipy 
import seaborn as sns

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch
from collections import OrderedDict
from matplotlib.gridspec import GridSpec
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mpl_toolkits.basemap import Basemap

from math import dist

import warnings
warnings.filterwarnings(action='once')
from geopandas import points_from_xy
import geopandas as gdp

#########################
### Cleaning Datasets ###
#########################


def Cleaning(data): 
    
    #removing empty columns
    Variables_to_remove = ['lot1_numero', 'lot1_surface_carrez', 
                       'lot2_numero', 'lot2_surface_carrez',
                       'lot3_numero', 'lot3_surface_carrez', 
                       'lot4_numero', 'lot4_surface_carrez', 
                       'lot5_numero', 'lot5_surface_carrez',
                       'code_nature_culture_speciale',  
                       'nature_culture_speciale',
                       'ancien_id_parcelle',
                       'adresse_suffixe', 'ancien_code_commune','ancien_nom_commune', 
                       'id_parcelle',
                       'numero_volume', 
                       'type_local', 'adresse_code_voie',   #redundant variables
                       'code_nature_culture',
                       'code_postal'] # on va utiliser le code commune comme il est replis a 100%

    data = data.drop(Variables_to_remove, axis=1)
    
    #removing empty rows
    data = data[(data.valeur_fonciere.isna() == False)   
               & (data.latitude.isna() == False) 
               & (data.longitude.isna() == False)
               & (data['nature_mutation'] == "Vente") 
               ]
    
    #to keep the row if nature_culture is empty but code-type_local is filled
    #data = data[(data.nature_culture.isna()) | (data.code_type_local.isna() == False)]
    
    #we limited our self for the appartments, houses and dependences. so we exclude 'loal commerciales'
    data = data[(data['code_type_local'] == 1.0)        #Maison
               | (data['code_type_local'] == 2.0)       #appartement
               | (data['code_type_local'] == 3.0)]      # dependance
    
    
    data = data[(data['nature_culture'] != "terrains a bâtir")]
    
    # Creation of tow columns, 1 for the number of 'depandance' 
    #                       and the second for the number of Houses or appartments per mutation
    Nombre_dependance = data.groupby(['id_mutation'])['code_type_local'].count().reset_index()
    Nombre_dependance.rename(columns={'code_type_local': 'Nombre_dependance'}, inplace=True)
    data = data.merge(Nombre_dependance, on=['id_mutation'])
    
    Nombre_house = data[(data['code_type_local'] == 1) | 
            (data['code_type_local'] == 2)].groupby(['id_mutation'])['code_type_local'].count().reset_index()
    Nombre_house.rename(columns={'code_type_local': 'Nombre_house'}, inplace=True)
    data = data.merge(Nombre_house, on=['id_mutation'])
    data['Nombre_dependance'] = data['Nombre_dependance'] - data['Nombre_house']
    

    # fixing the problem of multiple Id_mutation for the same sale. 
    #we have for the same sale a dependence and a house/appartment so we aggregated them in one row with on id_mutation.
    
    #in This fucntion we used only the first proposition(waiting to the meeting)
    data = data[(data['nature_culture'] == "sols") 
                | (data.nature_culture.isna())
                | (data['nature_culture'] == "jardins")].groupby(['id_mutation','numero_disposition','date_mutation'],
                                                    as_index=False).agg(
                                                       {'date_mutation':'first',
                                                        'code_type_local': 'min', 
                                                        'code_commune':'max',
                                                        'surface_terrain': 'max', 
                                                        'surface_reelle_bati':'sum',
                                                        'nombre_pieces_principales':'sum',
                                                        'nature_culture': 'sum',
                                                        'valeur_fonciere':'max', 
                                                        'latitude':'max' , 'longitude':'max',
                                                        'nombre_lots':'max',
                                                        'numero_disposition': 'max',
                                                        'code_departement':'first',
                                                        'Nombre_house':'max',
                                                        'Nombre_dependance':'max'
                                                        })
    
    #setting the data for appartments and houses
    data = data[(data.code_type_local.isna() == False)]
    
    data = data[(data['code_type_local'] == 1.0)        #Maison ' '
               | (data['code_type_local'] == 2.0) ]     #appartement   
    
    data['surface_reelle_bati'].fillna(0, inplace=True)
    data['nombre_pieces_principales'].fillna(0, inplace=True)
    data['surface_terrain'].fillna(0, inplace = True)
    
    data['date_mutation'] = pd.to_datetime(data['date_mutation'], format="%Y-%m-%d")
    

    # NOMBRE DE PIECES
    # On exclut les biens dont le nombre de pièces est inférieure à 1 et supérieur à 20 :
    
    data.drop(data[data['nombre_pieces_principales']<1].index, inplace=True)
    data.drop(data[data['nombre_pieces_principales']>20].index, inplace=True)

    # SURFACE DU BIEN 
    # On exclut les biens dont la surface est inférieure au seuil légal de 10 mètres carrés pour la location et les prix 
    # moins que 2000 euros:
    data.drop(data[data['surface_reelle_bati']<10].index, inplace=True)
    data.drop(data[data['valeur_fonciere'] < 2000].index, inplace=True)
    
    #creating the variable Quarter
    data['Year'] = pd.DatetimeIndex(data['date_mutation']).year
    data['num_trimestre'] = data['date_mutation'].dt.quarter
    # Concaténation sous la forme YEAR-Qi
    data['quarter'] = data['Year'].astype(str) + '_Q' + data['num_trimestre'].astype(str)
    
    return data


def Cleaning_iris(df_iris):
    Variables_to_remove = ['ze2010_code', 'ze2010_name', 'ept_name',
                           'ept_code', 'ze2020_name', 'ze2020_code',
                           'arrdep_name', 'iris_name_u', 'iris_area_c','iris_type', 
                           'iris_grd_qu', 'iris_in_ctu', 'reg_name', 'dep_name','arrdep_code',
                           'bv2012_code', 'bv2012_name', 'epci_code', 'com_code', 'com_name', 
                           'com_arm_cod', 'com_arm_nam', 'year', 'reg_code', 'dep_code']

    df_iris = df_iris.drop(Variables_to_remove, axis=1)
    
    df_iris['iris_code']  = df_iris['iris_code'].str[2:11]
    df_iris['iris_name'] = df_iris['iris_name'].str[2:-3]
    df_iris.dropna(inplace = True)
    
    #removing empty rows
    df_iris = df_iris[(df_iris.geometry.isna() == False) | (df_iris.epci_name.isna() == False)]
    
    return df_iris


########################
### Adding Variables ###
########################

def Add_IRIS(data, df_iris):    
    # turn df_test into geodataframe
    df_spatial = gdp.GeoDataFrame(data, crs="EPSG:4326", 
                                  geometry=points_from_xy(
                                     data["longitude"], data["latitude"]),)
    data_spatial = gdp.sjoin(df_spatial, df_iris,how='left', 
                               predicate="within",)
    
    return data_spatial

#adding p/m2 and standard deviation/iris
def Prix_m2(data):
    data['Prix_m2'] = data['valeur_fonciere'] / data['surface_reelle_bati']
    data['Month'] = pd.DatetimeIndex(data['date_mutation']).month
    
    Prix_moy = data.groupby(['iris_code','quarter'])['Prix_m2'].mean().reset_index()
    Prix_moy.rename(columns={'Prix_m2': 'Pris_m2_moy_iris'}, inplace=True)
    data = data.merge(Prix_moy, on=['iris_code','quarter'])
    
    std = data.groupby(['iris_code','quarter'])['Prix_m2'].std().reset_index()
    std.rename(columns={'Prix_m2': 'std'}, inplace=True)
    data = data.merge(std, on=['iris_code','quarter'])
    
    data.replace([np.inf, -np.inf], 0, inplace=True)
    
    return data

def Vente_iris_tri(data):
    #Vente_par_iris 
    Vente_par_iris = data.groupby(['iris_code'])['id_mutation'].count().reset_index()
    Vente_par_iris.rename(columns={'id_mutation' : 'Vente_par_iris'}, inplace=True)
    data = data.merge(Vente_par_iris, on=['iris_code'])
    
    Vente_par_iris_tri = data.groupby(['iris_code', 'quarter'])['id_mutation'].count().reset_index()
    Vente_par_iris_tri.rename(columns={'id_mutation' : 'Vente_par_iris_tri'}, inplace=True)
    data = data.merge(Vente_par_iris_tri, on=['iris_code','quarter'])
    
    return data


def Add_IPL(data, ipl):
    
    ipl.columns = ['date','IPL']
    ipl["IPL"].astype('float64')
    ipl['quarter'] = ipl['date'].apply(lambda x : x[:4]+'_Q'+x[-1])
    ipl['IPL_{Q-1}'] = ipl['IPL'].shift(-1)
    ipl.drop(['date'], axis = 1, inplace = True)
    data = data.merge(ipl, on = 'quarter')
    
    return data


def Add_logement(data, logt_iris):
    
    logt_iris.columns
    logt_iris = logt_iris[["IRIS", "P18_LOG", "P18_RP", "P18_RSECOCC", "P18_LOGVAC", "P18_MAISON", "P18_APPART",
                       "P18_RP_PROP", "P18_RP_LOC"]]
    
    logt_iris.columns = ["iris_code", "N_logements", "N_res_ppale", "N_res_second", "N_vacant", 
                     "N_maisons", 'N_apparts', "N_proprietaire", "N_locataire"]

    logt_iris[['N_logements', 'N_res_ppale', 
           "N_res_second", "N_vacant", "N_maisons", 
           'N_apparts', "N_proprietaire", "N_locataire"]] = round(logt_iris[['N_logements', 
                                                                             'N_res_ppale', "N_res_second",
                                                                             "N_vacant", "N_maisons", 'N_apparts',
                                                                             "N_proprietaire", "N_locataire"]])
    
    data = data.merge(logt_iris, on = 'iris_code')
    
    return data

##Add revenue 
def Add_revenue(data, revenues):
    revenues.rename(columns={'IRIS' : 'iris_code'}, inplace=True)
    revenues = revenues[['iris_code' , 'DISP_MED19']]
    data = data.merge(revenues, on = 'iris_code')
    
    return data
    
## Adding metros
def num_dist_metro(test) : 
    metro_iris = metros[metros['iris_code'] == test.iris_code]
    metro_arr = metros[metros['Arrondissement'] == test.Arrondissement]
    #print(metro_iris.shape[0])
    N_metros_iris = metro_iris.shape[0]
    N_metros_arr = metro_arr.shape[0]

    distances = []
    
    if N_metros_iris > 0: 
        for i in range(metro_iris.shape[0]) : 
            latitude_metro = metro_iris['Latitude'].iloc[i]
            longitude_metro = metro_iris['Longitude'].iloc[i]
            point_metro = [latitude_metro, longitude_metro]
            point_bien = [test.latitude, test.longitude]
            distances.append(dist(point_bien, point_metro))
    else :
        for i in range(metro_arr.shape[0]) : 
        
            latitude_metro = metro_arr['Latitude'].iloc[i]
            longitude_metro = metro_arr['Longitude'].iloc[i]
            point_metro = [latitude_metro, longitude_metro]
            point_bien = [test.latitude, test.longitude]
            distances.append(dist(point_bien, point_metro))
    dist_metro = np.min(distances)
    
    return distances