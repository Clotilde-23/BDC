import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def map_data(data, year, variable) : 
    '''
    Plot the map of France
    '''
    plt.figure(figsize=(16,16))
    m1 = Basemap(llcrnrlon= -5, llcrnrlat=41,
                 urcrnrlon=13, urcrnrlat=52, 
                 lat_0=46.2374, lon_0=2.375,
                 resolution = 'h', projection = 'lcc')

    m1.drawcoastlines(color = 'black')
    m1.drawcountries(linewidth = 3)

    x, y = m1(list(df_processed_total[df_processed_total['Year'] == year]['longitude']), 
              list(df_processed_total[df_processed_total['Year'] == year]['latitude']))
    m1.scatter(x, y ,
              c = df_processed_total[df_processed_total['Year'] == year][variable],
              s = 8,
              cmap = 'viridis')

    plt.title('Price of meter_square for', str(year), 'in France' , fontsize = 24)
    plt.colorbar()
    plt.clim(0,10000)
    plt.show()
    
    
    
def plot_map(data_spatial):
    
    #to create the polynom variable
    data_spatial_map = pd.merge(data_spatial, df_iris,how='left', on= 'iris_code')
    data_spatial_map.rename(columns={'geometry_y': 'geometry'}, inplace=True)
    data_spatial_VF = data_spatial_map.drop(['epci_name_y' 
                                                  ,'iris_name_y',
                                                  'iris_name_l_y'], axis = 1)
    
    return data_spatial_VF[data_spatial_VF['Year'] == 2017].drop(['date_mutation'],axis = 1).explore(
        tiles="CartoDB positron", cmap="tab20b",vmin = 6000, vmax =20000, 
        style_kwds= {"opacity":0.1 , 'fillOpacity':0.1},  column='Pris_m2_moy_iris')



def visualise(df, var_prix):
    df_sorted = df.sort_values(by=var_prix)
    x = df_sorted['longitude']
    y = df_sorted['latitude']
    c = df_sorted[var_prix] 

    plt.rcParams['figure.figsize'] = [7, 5]
    plt.rcParams['figure.dpi'] = 100 

    plt.scatter(x, y, s=0.01, c=c, cmap='plasma_r', alpha=0.8)
    plt.colorbar()
    plt.show()
