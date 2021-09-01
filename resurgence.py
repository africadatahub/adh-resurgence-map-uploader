import pandas as pd
import numpy as np
from functools import partial

def inc(arr):
    return np.sum(arr)

def change(arr):
    #added check for inf/NaN here
    change = (((arr.values[6] - arr.values[0]) / np.absolute(arr.values[0])) * 100)
    if(change == np.inf):
        change = int()
    return change

def data_gaps(threshold, col):
    return col.groupby((col != col.shift()).cumsum()).transform('count').gt(threshold).any() == 1.0


print("Fetching OWID Data...")

covid19_df = pd.read_csv('https://covid.ourworldindata.org/data/owid-covid-data.csv')

print("Got the OWID Data...")

df = covid19_df.loc[covid19_df.continent == 'Africa']

print("Exporting Africa Data...")

df.to_csv('africa.csv',index=False)

print("Starting Resurgence Calculations...")

iso_codes = df['iso_code'].unique()

africa_df = pd.DataFrame()

for iso_code in iso_codes:
    print(iso_code + '- % Change')
    country_df = df.loc[df.iso_code == iso_code]
    country_df = country_df[['date','iso_code','location','new_cases_smoothed_per_million']]
    country_df.insert(4,'summed', country_df.new_cases_smoothed_per_million.rolling(7).apply(inc))
    country_df.insert(5,'change', country_df.summed.rolling(7).apply(change))
    
    print(iso_code + '- 14-Day Case History')
    case_history = []
    for i in range(len(country_df)):
        case_history.append('|'.join([str(x) for x in country_df.shift(i).tail(14).new_cases_smoothed_per_million]))
    country_df.insert(6,'case_history', case_history[::-1])

    print(iso_code + '- Data Gaps')
    gap_size = 3
    data_gaps_actual = partial(data_gaps, gap_size)
    country_df.insert(7, 'data_gaps', country_df.new_cases_smoothed_per_million.rolling(7).apply(data_gaps_actual))
    
    africa_df = africa_df.append(country_df, ignore_index=True)

print("Exporting Resurgence Data...")

africa_df.to_csv('resurgence.csv',index=False)

print("Done")