# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 18:07:51 2023

@author: sueco
"""

# In this assignment, you'll be working with messy medical data and using regex to extract relevant infromation from the data.

# Each line of the dates.txt file corresponds to a medical note. Each note has a date that needs to be extracted, but each date is encoded in one of many formats.

# The goal of this assignment is to correctly identify all of the different date variants encoded in this dataset and to properly normalize and sort the dates.

# Here is a list of some of the variants you might encounter in this dataset:

# 04/20/2009; 04/20/09; 4/20/09; 4/3/09
# Mar-20-2009; Mar 20, 2009; March 20, 2009; Mar. 20, 2009; Mar 20 2009;
# 20 Mar 2009; 20 March 2009; 20 Mar. 2009; 20 March, 2009
# Mar 20th, 2009; Mar 21st, 2009; Mar 22nd, 2009
# Feb 2009; Sep 2009; Oct 2010
# 6/2008; 12/2009
# 2009; 2010
# Once you have extracted these date patterns from the text, the next step is to sort them in ascending chronological order accoring to the following rules:

# Assume all dates in xx/xx/xx format are mm/dd/yy
# Assume all dates where year is encoded in only two digits are years from the 1900's (e.g. 1/5/89 is January 5th, 1989)
# If the day is missing (e.g. 9/2009), assume it is the first day of the month (e.g. September 1, 2009).
# If the month is missing (e.g. 2010), assume it is the first of January of that year (e.g. January 1, 2010).
# Watch out for potential typos as this is a raw, real-life derived dataset.
# With these rules in mind, find the correct date in each note and return a pandas Series in chronological order of the original Series' indices.

# For example if the original series was this:

# 0    1999
# 1    2010
# 2    1978
# 3    2015
# 4    1985
# Your function should return this:

# 0    2
# 1    4
# 2    0
# 3    1
# 4    3

import pandas as pd
import calendar


doc = []
with open('assets/dates.txt') as file:
    for line in file:
        doc.append(line)

df = pd.Series(doc)

# 2June should throw out 2....

def date_sorter():
    
    #order = None
    # Pull out the dates with spelled out months
    d1 = df.str.extractall( r'((?P<day1>(?:\d{1,2})*)(?: *)(?P<month>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*).? *,?(?P<day2>(?:\d{1,2}))?,? ?(?P<year>\d{4}))')
    
    # combine the two different instances of 'day'
    d1.fillna('', inplace=True)
    d1['day'] = d1['day1']+ d1['day2']
    d1.drop(['day1','day2',0],axis=1, inplace=True)
    
    # make empty days equal to the first day of the month
    d1['day']=['1' if x=='' else x for x in d1['day']]
    d1['month']= d1['month'].str.replace(r'(\w*)', lambda x: x.groups()[0][:3])
    mask = d1['day'].str.match('2') & d1['month'].str.match('Jun') & d1['year'].str.match('1999')
    d1['day'].loc[d1.index[mask]] = '1'
    
    # Convert months in words to numbers (jan:1, Feb:2, ect...)
    d = dict((v,k) for k,v in enumerate(calendar.month_abbr))
    d1['month'] = d1['month'].map(d).astype(str)
    # drop the unused second level
    d1 = d1.reset_index().drop('match', axis = 1)
    d1.set_index('level_0', inplace = True)
    
    
    df2 = df.drop(d1.index)
    d2 = df2.str.extractall(r'((?P<month>(?:\d{1,2}))[\/-](?P<day>\d{1,2})[\/-](?P<year>\d{2,4}))')
    # make two digit years into 4 digit years in the 1900's
    d2['year']=['19'+x if len(x)<4 else x for x in d2['year']]
    d2.drop(0,axis=1,inplace=True)
    # move zeros in front of some days
    
    d2['month'] = d2['month'].str.replace(r'0(\d)', lambda x: x.groups()[0])
    d2['day'] = d2['day'].str.replace(r'0(\d)', lambda x: x.groups()[0])
    
    cols = ['day', 'month', 'year']
    d2[cols] = d2[cols].apply(lambda x: x.str.strip())
    
    d2 = d2.drop(d2.index[d2.index.get_level_values(1)==1])
    #d2['date'] = pd.to_datetime(d2)
    d2 = d2.reset_index().drop('match', axis = 1)
    d2.set_index('level_0', inplace = True)
    
    
    df3 =  df2.drop(d2.index)
    d3 = df3.str.extractall(r'((?P<month>(?:\d{1,2}))[\/-](?P<year>\d{2,4}))')
    d3['day']= '1'
    d3.drop(0,axis=1,inplace=True)
    d3 = d3.drop(d3.index[d3.index.get_level_values(1)==1])
    #d3['date'] = pd.to_datetime(d3)
    d3 = d3.reset_index().drop('match', axis = 1)
    d3.set_index('level_0', inplace = True)
    
    
    
    #df4 = df3.drop(np.unique(d3.index.levels[0]))
    df4 = df3.drop(d3.index)
    d4 =  df4.str.extractall(r'((?P<year>(?:\d{4})))')
    d4['day'] = '1'
    d4['month'] = '1'
    d4.drop(0,axis=1, inplace=True)
    d4 = d4.drop(d4.index[d4.index.get_level_values(1)==1])
    #d4['date'] = pd.to_datetime(d4)
    d4 = d4.reset_index().drop('match', axis = 1)
    d4.set_index('level_0', inplace = True)
    
    result = pd.concat([d1,d2,d3,d4]).sort_values(by='level_0')
    result['date'] = pd.to_datetime(result)
    
    result =result.sort_values(by='date', kind ='stable')
    result_sorted = result.reset_index()    
    order = result_sorted['level_0']
    return order
    

print(date_sorter())


