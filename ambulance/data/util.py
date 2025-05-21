import pandas as pd
def zip_map(file_path="./uszips.xlsx"):

    columns_to_keep = ['zip', 'lat', 'lng']
    df = pd.read_excel(file_path, usecols=columns_to_keep)

    return df
