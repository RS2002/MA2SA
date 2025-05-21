'''
get valid EMS positions
'''

import re
import numpy as np
from util import zip_map
import pandas as pd


columns_to_keep = ['Mailing Address']
df = pd.read_excel("./EMS_location.xlsx", usecols=columns_to_keep)
zipcode_list = df['Mailing Address'].str.extract(r'(\d{5})')[0].astype(int).tolist()
zipcode_list = np.array(zipcode_list)
zipcode_list = zipcode_list[zipcode_list>=10000]
zipcode_list = zipcode_list[zipcode_list<=11697]

df = zip_map()
result = df[df['zip'].isin(zipcode_list)][['zip', 'lat', 'lng']]
df = df.dropna()

result.to_excel('EMS.xlsx', index=False)