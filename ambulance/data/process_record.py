'''
process EMS records
'''

import re
import numpy as np
from util import zip_map
import pandas as pd


columns_to_keep = ['INCIDENT_DATETIME','INITIAL_CALL_TYPE','INITIAL_SEVERITY_LEVEL_CODE','ZIPCODE']
df = pd.read_csv("./EMS_Incident_Dispatch_Data_20250520.csv", usecols=columns_to_keep)
df = df.dropna()

df['INCIDENT_DATETIME'] = pd.to_datetime(df['INCIDENT_DATETIME'])
df['year'] = df['INCIDENT_DATETIME'].dt.year
df['month'] = df['INCIDENT_DATETIME'].dt.month
df['day'] = df['INCIDENT_DATETIME'].dt.day
df['hour'] = df['INCIDENT_DATETIME'].dt.hour
df['minute'] = df['INCIDENT_DATETIME'].dt.minute
df['second'] = df['INCIDENT_DATETIME'].dt.second
df['year'] = df['year'].astype(int)
df['month'] = df['month'].astype(int)
df['day'] = df['day'].astype(int)
df['hour'] = df['hour'].astype(int)
df['minute'] = df['minute'].astype(int)
df['second'] = df['second'].astype(int)

df_map = zip_map()
df = df.merge(df_map, left_on='ZIPCODE', right_on='zip', how='left')
df = df.drop(columns=['zip'])
df = df.dropna()

df.to_excel('Record.xlsx', index=False)