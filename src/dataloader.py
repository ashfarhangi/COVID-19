# Requirement
# pip install us
# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
import random
import os
import us
plt.rcParams['figure.figsize'] = 12, 8
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error, mean_absolute_error,mean_squared_error
### tensorflow 2 ###
import tensorflow as tf
from matplotlib import ticker 
import pycountry_convert as pc
# import folium
# import branca
from datetime import datetime, timedelta,date
from scipy.interpolate import make_interp_spline, BSpline
import plotly.express as px
import json, requests
import random
# import calmap



df_vaccine = pd.read_csv('https://raw.githubusercontent.com/govex/COVID-19/master/data_tables/vaccine_data/us_data/time_series/people_vaccinated_us_timeline.csv',parse_dates=['Date'])
df_vaccine = df_vaccine[['FIPS','Date','People_Fully_Vaccinated'	,'People_Partially_Vaccinated']].fillna(0)
df_vaccine.columns = ['fips', 'date' ,'fully_vaccinated','partially_vaccinated']
dfusa = pd.read_csv('https://raw.githubusercontent.com/nytimes/covid-19-data/master/us.csv')
dfc = pd.read_csv('https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv',parse_dates=['date'])
dfc= dfc.sort_values(['state','date']).reset_index(drop=True)
dfc = pd.merge(dfc,df_vaccine1,how='left',on=['fips','date'])
dfc = dfc.fillna(0)
print(dfc)

for j in range(len(dfc)):
    dfc.loc[j,'state'] = us.states.lookup(dfc.loc[j,'state']).abbr
dfc['dailynewcases']=0
dfc['dailynewdeaths']=0
dfc['lndailynewcases'] = 0
dfc['lndailynewdeaths'] = 0
dfc['new_full_vaccination']= 0
# dfc['new_partial_vaccination']= 0
dfc['ln_new_full_vaccination'] = 0
# dfc['ln_new_partial_vaccination'] = 0
for j in range(1,len(dfc)):
        dfc.loc[j,'dailynewcases'] =dfc.loc[j,'cases'] - dfc.loc[j-1,'cases']
        dfc.loc[j,'dailynewdeaths'] =dfc.loc[j,'deaths'] - dfc.loc[j-1,'deaths']
        dfc.loc[j,'new_full_vaccination'] =dfc.loc[j,'fully_vaccinated'] - dfc.loc[j-1,'fully_vaccinated']
        # dfc.loc[j,'new_partial_vaccination'] =dfc.loc[j,'partially_vaccinated'] - dfc.loc[j-1,'partially_vaccinated']
        if(dfc.loc[j,'state'] != dfc.loc[j-1,'state']):
            dfc.loc[j,'dailynewcases'] =0
            dfc.loc[j,'dailynewdeaths'] = 0
            dfc.loc[j,'new_full_vaccination'] =0
            # dfc.loc[j,'new_partial_vaccination'] = 0            
            #Condition to set negative daily cases into zero (False positive)
        if(dfc.loc[j,'dailynewcases'] <0):
            dfc.loc[j,'dailynewcases'] = 0 
        if(dfc.loc[j,'dailynewdeaths'] <0): 
            dfc.loc[j,'dailynewdeaths'] = 0
        if(dfc.loc[j,'new_full_vaccination'] <0):
            dfc.loc[j,'new_full_vaccination'] = 0 
        # if(dfc.loc[j,'new_partial_vaccination'] <0): 
            # dfc.loc[j,'new_partial_vaccination'] = 0            
        if (dfc.loc[j,'dailynewcases'] != 0):
            dfc.loc[j,'lndailynewcases']=np.log(dfc['dailynewcases'][j])
        if (dfc.loc[j,'dailynewdeaths'] != 0):
            dfc.loc[j,'lndailynewdeaths']=np.log(dfc['dailynewdeaths'][j])
        if (dfc.loc[j,'new_full_vaccination'] != 0):
            dfc.loc[j,'ln_new_full_vaccination']=np.log(dfc['new_full_vaccination'][j])
        # if (dfc.loc[j,'new_partial_vaccination'] != 0):
            # dfc.loc[j,'ln_new_partial_vaccination']=np.log(dfc['new_partial_vaccination'][j])            
dfc.describe()
plt.plot(dfc[dfc.state =='FL'].dailynewcases)
plt.show()