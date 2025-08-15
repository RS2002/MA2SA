'''
get valid hospital positions
'''

import pdfplumber
import re
import numpy as np
from util import zip_map

zipcode_list = []
with pdfplumber.open('./Directory of 220 Hospitals.pdf') as pdf:
    num_pages = len(pdf.pages)
    for page in range(num_pages):
        text = pdf.pages[page].extract_text()
        zipcodes = re.findall(r'NY (\d{5})', text)
        zipcode_list += zipcodes
zipcode_list = [int(zipcode) for zipcode in zipcode_list]
zipcode_list = np.array(zipcode_list)
zipcode_list = zipcode_list[zipcode_list>=10000]
zipcode_list = zipcode_list[zipcode_list<=11697]

df = zip_map()
result = df[df['zip'].isin(zipcode_list)][['zip', 'lat', 'lng']]
df = df.dropna()

result.to_excel('hospital.xlsx', index=False)

