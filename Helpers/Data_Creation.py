import pandas as pd
import numpy as np


def Base_Creation(df17, df18, df19, df20, df21, df22):
    
    #Cleaning raw data
    df_cleaned17 = Cleaning(df17)
    df_cleaned18 = Cleaning(df18)
    df_cleaned19 = Cleaning(df19)
    df_cleaned20 = Cleaning(df20)
    df_cleaned21 = Cleaning(df21)
    df_cleaned22 = Cleaning(df22)

    # Concat all data
    df_cleaned_tot = pd.concat([df_cleaned17, df_cleaned18, 
                            df_cleaned19, df_cleaned20, 
                            df_cleaned21, df_cleaned22] , axis=0 , ignore_index=True)
    
    return df_cleaned_tot


def Base_Ville(df_cleaned_tot, df_iris_2022, logt_iris,revenues, metros , Name):
    
    df_iris = Cleaning_iris(df_iris_2022)
    
    if Name == 'Paris': 
        df_data = df_cleaned_tot[df_cleaned_tot['code_departement'] == '75']
        
    elif Name == 'Marseille':
        df_data = df_cleaned_tot[(df_cleaned_tot['code_commune'] == '13201') 
                              | (df_cleaned_tot['code_commune'] == '13202')
                              | (df_cleaned_tot['code_commune'] == '13203')
                              | (df_cleaned_tot['code_commune'] == '13204')
                              | (df_cleaned_tot['code_commune'] == '13205')
                              | (df_cleaned_tot['code_commune'] == '13206')
                              | (df_cleaned_tot['code_commune'] == '13207')
                              | (df_cleaned_tot['code_commune'] == '13208')
                              | (df_cleaned_tot['code_commune'] == '13209')
                              | (df_cleaned_tot['code_commune'] == '13210')
                              | (df_cleaned_tot['code_commune'] == '13211')
                              | (df_cleaned_tot['code_commune'] == '13212')
                              | (df_cleaned_tot['code_commune'] == '13213')
                              | (df_cleaned_tot['code_commune'] == '13214')
                              | (df_cleaned_tot['code_commune'] == '13215')
                              | (df_cleaned_tot['code_commune'] == '13216')]
            
            
    elif Name == 'Lyon':
            df_data = df_cleaned_tot[(df_cleaned_tot['code_commune'] == '69381') 
                              | (df_cleaned_tot['code_commune'] == '69382')
                              | (df_cleaned_tot['code_commune'] == '69383')
                              | (df_cleaned_tot['code_commune'] == '69384')
                              | (df_cleaned_tot['code_commune'] == '69385')
                              | (df_cleaned_tot['code_commune'] == '69386')
                              | (df_cleaned_tot['code_commune'] == '69387')
                              | (df_cleaned_tot['code_commune'] == '69388')
                              | (df_cleaned_tot['code_commune'] == '69389')]
                
    elif Name == 'Toulouse':
                    df_data = df_cleaned_tot[(df_cleaned_tot['code_commune'] == '31555')]
                
            
    
    df_data_spatial      = Add_IRIS(df_data, df_iris)
    df_data_price        = Prix_m2(df_data_spatial)
    
    df_data_Processed1   = Process_data(df_data_price, 0.1, 0.9)
    df_data_Processed2   = Process_data_2(df_data_Processed1)
    
    df_data_vf           = Vente_iris_tri(df_data_Processed2)
    df_data_logement     = Add_logement(df_data_vf, logt_iris)
    df_data_revenues     = Add_revenue(df_data_logement , revenues)
    
    #Filtrer
    df_data_appartement_VF = data_spatial_data[(data_spatial_data['Nombre_house']==1) 
                                                 & (data_spatial_data['code_type_local']==2)]
    df_data_Maison_VF = data_spatial_data[(data_spatial_data['Nombre_house']==1) 
                                            & (data_spatial_data['code_type_local']==1)]
    
    return df_data_appartement_VF, df_data_Maison_VF
    
    