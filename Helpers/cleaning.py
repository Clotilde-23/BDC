#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 09:11:01 2023

@author: cloclo
"""

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
                       'adresse_numero', 'adresse_nom_voie', 'id_parcelle',
                       'numero_volume', 
                       'nom_commune', 'type_local', 'adresse_code_voie', 'nom_commune',    #redundant variables
                       'code_departement', 'code_nature_culture', 
                       'code_postal'] # on va utiliser le code commune comme il est replis a 100%

    data = data.drop(Variables_to_remove, axis=1)
    
    #removing empty rows
    data = data[(data.valeur_fonciere.isna() == False)   
               & (data.latitude.isna() == False) 
               & (data.longitude.isna() == False)
               & (data['nature_mutation'] == "Vente") 
               ] 
    
    #we limited our self for the appartments, houses and dependences. so we exclude 'loal commerciales'
    data = data[(data['code_type_local'] == 1.0)        #Maison
               | (data['code_type_local'] == 2.0)       #appartement
               | (data['code_type_local'] == 3.0)]      # dependance
    
    
    data = data[(data['nature_culture'] != "terrains a bâtir")]
    

    #fixing the problem of multiple Id_mutation for the same sale. 
    #we have for the same sale a dependence and a house/appartment so we aggregated them in one row with on id_mutation.
    
    #in This fucntion we used only the first proposition(waiting to the meeting)
    data = data[(data['nature_culture'] == "sols") 
                | (data.nature_culture.isna()) ].groupby('id_mutation', as_index=False).agg(
                                                       {'date_mutation':'first',
                                                        'code_type_local': 'min', 
                                                        'code_commune':'max',
                                                        'surface_terrain': 'max', 
                                                        'surface_reelle_bati':'sum',
                                                        'nombre_pieces_principales':'sum',
                                                        'nature_culture': 'first',
                                                        'valeur_fonciere':'max', 
                                                        'latitude':'max' , 'longitude':'max',
                                                        'nombre_lots':'first'
                                                        })
    
    #setting the data for appartments and houses
    data = data[(data.code_type_local.isna() == False)]
    
    data = data[(data['code_type_local'] == 1.0)        #Maison
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
    # On exclut les biens dont la surface est inférieure au seuil légal de 9 mètres carrés pour la location :
    data.drop(data[data['surface_reelle_bati']<9].index, inplace=True)

    data.drop('nature_culture' , axis =1)
    
    return data


def Prix_m2(data):
    data['Prix_m2'] = data['valeur_fonciere'] / data['surface_reelle_bati']
    data['Year'] = pd.DatetimeIndex(data['date_mutation']).year
    data['Month'] = pd.DatetimeIndex(data['date_mutation']).month
    
    #Somme du nombre de mètre carré de surface habitable vendue par commune par an
    Sum_m2_bati_par_commune_par_an = data.groupby(['code_commune','Year'])['surface_reelle_bati'].sum().reset_index()
    Sum_m2_bati_par_commune_par_an.rename(columns={'surface_reelle_bati': 'Sum_surface_bati_com'}, inplace=True)
    data = data.merge(Sum_m2_bati_par_commune_par_an, on=['code_commune','Year'])
    
    #Somme du nombre de mètre carré de surface terrain vendue par commune par an
    Sum_m2_terr_par_commune_par_an = data.groupby(['code_commune','Year'])['surface_terrain'].sum().reset_index()
    Sum_m2_terr_par_commune_par_an.rename(columns={'surface_terrain': 'Sum_surface_terr_com'}, inplace=True)
    data = data.merge(Sum_m2_terr_par_commune_par_an, on=['code_commune','Year'])

    #Somme des ventes par commune par
    Sum_ventes_par_commune_par_an = data.groupby(['code_commune','Year'])['valeur_fonciere'].sum().reset_index()
    Sum_ventes_par_commune_par_an.rename(columns={'valeur_fonciere': 'Sum_valeur_fonc'}, inplace=True)
    data = data.merge(Sum_ventes_par_commune_par_an, on=['code_commune','Year'])

    #Prix moyen par commune du mètre carré de surface habitable
    data = data.assign(prix_m2_moy_surf_habit=(data["Sum_valeur_fonc"]
                                                    /data["Sum_surface_bati_com"]))
    
    data = data.assign(prix_m2_moy_terrain=(data["Sum_valeur_fonc"]
                                                    /data["Sum_surface_terr_com"]))

    #Suppression des variables temporaires 
    data=data.drop(['Sum_surface_bati_com', 'Sum_surface_terr_com' , 'Sum_valeur_fonc'],axis=1)
    data.replace([np.inf, -np.inf], 0, inplace=True)
    return data

def Process_data(data, quantile_low, quantile_high):    
    
    #we remove the extreme 5 % values of price per meter_square
    Threshold_up = data.groupby(['code_commune','Year'])['Prix_m2'].quantile(quantile_high).reset_index()
    Threshold_down = data.groupby(['code_commune','Year'])['Prix_m2'].quantile(quantile_low).reset_index()

    df_comune ={'code_commune':Threshold_up['code_commune'], 'Threshold_up_commune': Threshold_up['Prix_m2'] , 
            'Threshold_down_commune':Threshold_down['Prix_m2'] , 'Year': Threshold_down['Year']}
    df_comune = pd.DataFrame(df_comune)

    df_process = pd.merge(data, df_comune ,how='inner', on=['code_commune','Year'])

    df_final = df_process[(df_process['Prix_m2'] < df_process['Threshold_up_commune']) 
                      & (df_process['Prix_m2'] > df_process['Threshold_down_commune'])]
    
    df_final=df_final.drop(['Threshold_up_commune','Threshold_down_commune'],axis = 1)
    
    return df_final


