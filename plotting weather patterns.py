# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 06:45:40 2022

@author: sueco
"""

# assignment - plotting weather patterns

# data format
    #ID: station identification code 
    # date: date in YYYY-MM-DD format
    # element: indicator of element type
        # TMAX : maximum temperature (10ths of deg cel)
        # TMIN: Min temperature (10th deg C)
    #Value: data value for element (10ths of deg C)
    



#%% Step 1

#Load the dataset and transform the data into Celcius (refer to documentation) then extract all of the rows which have minimum or maximum temperatures.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib 


# Then, "ALWAYS use sans-serif fonts"
matplotlib.rcParams['font.family'] = "Times New Roman"


df = pd.read_csv('assets/fb441e62df2d58994928907a91895ec62c2c42e6cd075c2700843b89.csv')
print(df.head())

# convert dates to datetime
df['Date'] = pd.to_datetime(df['Date'])
df['Day'] = df['Date'].apply(lambda x: x.strftime('%m-%d'))

#Drop all leap year feb 29ths
df.drop(df[df['Day'] == '02-29'].index,inplace=True)


# put temp values in degree C
df['degC'] = df.Data_Value/10


#df_14= df[df['Date']<pd.to_datetime('2015')]
dfmaxAll = df[df['Element'] == 'TMAX']
dfminAll = df[df['Element'] == 'TMIN']


# take max min values by date
#df_range= df[.groupby('Date')['degC'].agg(['max','min'])

#%% Step 2
#In order to visualize the data we would plot the min and max data for each day of the year between the years 2005 and 2014 across all weather stations. But we also need to find out when the min or max temperature in 2015 falls below the min or rises above the max for the previous decade.


dfmax= dfmaxAll.groupby('Date')['degC'].agg(['max'])
dfmin= dfminAll.groupby('Date')['degC'].agg(['min'])

#%%  step 3

# find the max and min of the temperature data for each day of the year for the 2005-2014 data.


dfmax_14 = dfmaxAll[dfmaxAll['Date'] < pd.to_datetime('2015')].groupby('Day')['degC'].agg(['max'])
dfmin_14 = dfminAll[dfminAll['Date'] < pd.to_datetime('2015')].groupby('Day')['degC'].agg(['min'])


dfmax_15 = dfmaxAll[dfmaxAll['Date']>= pd.to_datetime('2015')].groupby('Day')['degC'].agg(['max'])
dfmin_15 = dfminAll[dfminAll['Date']>= pd.to_datetime('2015')].groupby('Day')['degC'].agg(['min'])

#%% Step 4  Plotting



plt.figure(figsize=(8,6))
plt.plot(dfmax_14, color = 'indianred', label='_nolegend_')
plt.plot(dfmin_14, color = 'steelblue', label='_nolegend_')

#Now we need to get hte current axes ofjbect and call fill-between.  We didn't spefiy any x values in our call to plot, so we'll just use the same range of data points it's already using.  Then we'll put in our lower bounds and our upper bounds along wit hthe color we want paninted and for fun I'll include a transparency value.  

ca = plt.gca()

ca.fill_between(range(len(dfmax_14)), 
                      dfmax_14['max'],dfmin_14['min'], 
                      facecolor='grey',
                      alpha=0.25, label='_nolegend_')

ca.spines['right'].set_visible(False)
ca.spines['top'].set_visible(False)


ca.set_xticks(np.linspace(15,355,num=12))
ca.set_xticklabels(('Jan','Feb','Mar','Apr','May','Jun','Jly','Aug','Sep','Oct','Nov','Dec'))

# Now we can itterate though each one a rotate the tick labels
#for item in x.get_ticklabels():
#    item.set_rotation(45)

#pull out the days for which 2015 data was outside the range of 2005-2014
#make a new dataframe with 2015 data added

df_15gt14max = dfmax_15[(dfmax_15['max']-dfmax_14['max'])>0]
df_14gt15min = dfmin_15[(dfmin_14['min']-dfmin_15['min'])>0]

plt.plot(df_15gt14max,'k.',df_14gt15min,'k.')

plt.xlabel('Date')
plt.ylabel('Temperature deg C')
plt.title('Days in 2015 for Ann Arbor, Michigan\nwhen temperatures broke\n2005-2014 record')
# Add a legend with legend entries (because we didn't have labels when we plotted the data series)
plt.legend(['2015 data outside\n2005-2014 range'],loc='best')
#,bbox_to_anchor=(0.9, 0.9)

plt.text(290,-22,'Minimum daily\ntemperatures\n2005-2014',color = 'steelblue',ha='center')
plt.text(40,25,'Maximum daily\ntemperatures\n2005-2014',color = 'indianred',ha='center')

plt.savefig('PlottingWeatherPatterns.png')










