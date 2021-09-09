import requests
import json
import pandas as pd
import numpy as np
import argparse
from functools import partial

"""
COMMAND LINE ARGUMENTS
Ex. python resurgence_uploader.py --site prod --key 123abc

"""

parser = argparse.ArgumentParser(description='Provide API key')
parser.add_argument("--key", help="CKAN API key")
parser.add_argument("--site", help="dev/prod")
args = parser.parse_args()

"""
CKAN SPECIFIC SETTINGS

"""

opts = {
    "dev": {
        "api_url": "https://adhtest.opencitieslab.org",
        "resources": {
            "resurgence": "7e58603e-0b06-47cf-8e77-54b0d567d6eb",
            "countries": "fc2a18a1-0c76-4afe-8934-2b9a9dacfef4"
        }
    },
    "prod": {
        "api_url": "https://ckan.africadatahub.org",
        "resources": {
            "resurgence": "e6489086-6e9a-4e3b-94c5-5236809db053",
            "countries": "8bf9f7fe-ec0d-468d-bc7e-be9a1130dd3a"
        }
    }
}

ckan_url = opts[args.site]['api_url']
api_key = opts[args.key]
resurgence_id = opts[args.site]['resources']['resurgence']
countries_id = opts[args.site]['resources']['countries']

"""
CKAN FUNCTIONS

"""

def update_resource(df, filename, ckan_url, resource_id, ckan_api_key):
    
    df.to_csv(filename, index=False)

    u = ckan_url + "/api/action/resource_patch"
    r = requests.post(u, headers={
      "X-CKAN-API-Key": ckan_api_key
    }, data={
        "id": resource_id
    }, files=[('upload', open('./' + filename, 'r'))])
    data_output = json.loads(r.content)
    print(data_output)


def datastore_create(records, fields, ckan_url, resource_id, ckan_api_key):
    print('Creating new datastore for ' + resource_id)

    u = ckan_url + "/api/action/datastore_create"
    r = requests.post(u, headers={
      "X-CKAN-API-Key": ckan_api_key,
      "Accept": "application/json",
      'Content-Type': 'application/json',
    }, data=json.dumps({
        "resource_id": resource_id,
        "records": records,
        "fields": fields,
        "force": "true"
    }))

    return json.loads(r.text)

def datastore_delete(ckan_url, resource_id, ckan_api_key):
    print('Deleting datastore for ' + resource_id)

    u = ckan_url + "/api/action/datastore_delete"
    r = requests.post(u, headers={
      "X-CKAN-API-Key": ckan_api_key,
      "Accept": "application/json",
      'Content-Type': 'application/json',
    }, data=json.dumps({
        "resource_id": resource_id,
        "force": "true"
    }))

    return json.loads(r.text)


"""
GET OWID DATASET
Fetched the latest Covid19 data from OWID and filters for Africa

"""

print('Fetching OWID Data')

covid19_df = pd.read_csv('https://covid.ourworldindata.org/data/owid-covid-data.csv')

df = covid19_df.loc[covid19_df.continent == 'Africa']

"""
CALCULATE RESURGENCE
And update CKAN Resource datastore

"""

def inc(arr):
    return np.sum(arr)

def change(arr):
    change = (((arr.values[6] - arr.values[0]) / np.absolute(arr.values[0])) * 100)
    if(change == np.inf):
        change = int()
    return change

def data_gaps(threshold, col):
    return col.groupby((col != col.shift()).cumsum()).transform('count').gt(threshold).any() == 1.0

def calculate_resurgence():

    print('Calculating Resurgence')

    iso_codes = df['iso_code'].unique()

    countries_df = pd.DataFrame()

    for iso_code in iso_codes:
        country_df = df.loc[df.iso_code == iso_code]
        country_df = country_df[['date','iso_code','location','new_cases_smoothed_per_million']]
        country_df.insert(4,'summed', country_df.new_cases_smoothed_per_million.rolling(7).apply(inc))
        country_df.insert(5,'change', country_df.summed.rolling(7).apply(change))
        
        case_history = []
        for i in range(len(country_df)):
            case_history.append('|'.join([str(x) for x in country_df.shift(i).tail(14).new_cases_smoothed_per_million]))
        country_df.insert(6,'case_history', case_history[::-1])

        gap_size = 3
        data_gaps_actual = partial(data_gaps, gap_size)
        country_df.insert(7, 'data_gaps', country_df.new_cases_smoothed_per_million.rolling(7).apply(data_gaps_actual))
        
        countries_df = countries_df.append(country_df, ignore_index=True)

    # update_resource(countries_df, 'resurgence.csv', ckan_url, resurgence_id, api_key)     

    countries_df = countries_df.to_numpy()
    
    records = []

    for row in countries_df:
        records.append({
            "date": row[0],
            "iso_code": row[1],
            "location": row[2],
            "new_cases_smoothed_per_million": row[3],
            "summed": row[4],
            "change": row[5],
            "case_history": row[6],
            "data_gaps": row[7],
        })
        
    fields = [
        {
            "id": "date",
            "type": "timestamp"
        },
        {
            "id": "iso_code",
            "type": "text"
        },
        {
            "id": "location",
            "type": "text"
        },
        {
            "id": "new_cases_smoothed_per_million",
            "type": "numeric"
        },
        {
            "id": "summed",
            "type": "numeric"
        },
        {
            "id": "change",
            "type": "numeric"
        },
        {
            "id": "case_history",
            "type": "text"
        },
        {
            "id": "data_gaps",
            "type": "numeric"
        }
    ]

    response = datastore_create(records, fields, ckan_url, resurgence_id, api_key)

    print(response)

datastore_delete(ckan_url, resurgence_id, api_key)
calculate_resurgence()


"""
UPLOAD COUNTRIES DATASET

"""

def countries_data():

    print('Collecting Countries Data')

    records = []

    # update_resource(df, 'africa.csv', ckan_url, countries_id, api_key)

    countries_df = df.to_numpy()

    for row in countries_df:

        records.append({
            "iso_code": row[0],
            "continent": row[1],
            "location": row[2],
            "date": row[3],
            "total_cases": row[4],
            "new_cases": row[5],
            "new_cases_smoothed": row[6],
            "total_deaths": row[7],
            "new_deaths": row[8],
            "new_deaths_smoothed": row[9],
            "total_cases_per_million": row[10],
            "new_cases_per_million": row[11],
            "new_cases_smoothed_per_million": row[12],
            "total_deaths_per_million": row[13],
            "new_deaths_per_million": row[14],
            "new_deaths_smoothed_per_million": row[15],
            "reproduction_rate": row[16],
            "icu_patients": row[17],
            "icu_patients_per_million": row[18],
            "hosp_patients": row[19],
            "hosp_patients_per_million": row[20],
            "weekly_icu_admissions": row[21],
            "weekly_icu_admissions_per_million": row[22],
            "weekly_hosp_admissions": row[23],
            "weekly_hosp_admissions_per_million": row[24],
            "new_tests": row[25],
            "total_tests": row[26],
            "total_tests_per_thousand": row[27],
            "new_tests_per_thousand": row[28],
            "new_tests_smoothed": row[29],
            "new_tests_smoothed_per_thousand": row[30],
            "positive_rate": row[31],
            "tests_per_case": row[32],
            "tests_units": 'nan',
            "total_vaccinations": row[34],
            "people_vaccinated": row[35],
            "people_fully_vaccinated": row[36],
            "total_boosters": row[37],
            "new_vaccinations": row[38],
            "new_vaccinations_smoothed": row[39],
            "total_vaccinations_per_hundred": row[40],
            "people_vaccinated_per_hundred": row[41],
            "people_fully_vaccinated_per_hundred": row[42],
            "total_boosters_per_hundred": row[43],
            "new_vaccinations_smoothed_per_million": row[44],
            "stringency_index": row[45],
            "population": row[46],
            "population_density": row[47],
            "median_age": row[48],
            "aged_65_older": row[49],
            "aged_70_older": row[50],
            "gdp_per_capita": row[51],
            "extreme_poverty": row[52],
            "cardiovasc_death_rate": row[53],
            "diabetes_prevalence": row[54],
            "female_smokers": row[55],
            "male_smokers": row[56],
            "handwashing_facilities": row[57],
            "hospital_beds_per_thousand": row[58],
            "life_expectancy": row[59],
            "human_development_index": row[60],
            "excess_mortality": row[61]
        })

        fields = [
            { 
                "id": "iso_code",
                "type": "text"
            },
            { 
                "id": "continent",
                "type": "text"
            },
            { 
                "id": "location",
                "type": "text"
            },
            { 
                "id": "date",
                "type": "timestamp"
            },
            { 
                "id": "total_cases",
                "type": "numeric"
            },
            { 
                "id": "new_cases",
                "type": "numeric"
            },
            { 
                "id": "new_cases_smoothed",
                "type": "numeric"
            },
            { 
                "id": "total_deaths",
                "type": "numeric"
            },
            { 
                "id": "new_deaths",
                "type": "numeric"
            },
            { 
                "id": "new_deaths_smoothed",
                "type": "numeric"
            },
            { 
                "id": "total_cases_per_million",
                "type": "numeric"
            },
            { 
                "id": "new_cases_per_million",
                "type": "numeric"
            },
            { 
                "id": "new_cases_smoothed_per_million",
                "type": "numeric"
            },
            { 
                "id": "total_deaths_per_million",
                "type": "numeric"
            },
            { 
                "id": "new_deaths_per_million",
                "type": "numeric"
            },
            { 
                "id": "new_deaths_smoothed_per_million",
                "type": "numeric"
            },
            { 
                "id": "reproduction_rate",
                "type": "numeric"
            },
            { 
                "id": "icu_patients",
                "type": "numeric"
            },
            { 
                "id": "icu_patients_per_million",
                "type": "numeric"
            },
            { 
                "id": "hosp_patients",
                "type": "numeric"
            },
            { 
                "id": "hosp_patients_per_million",
                "type": "numeric"
            },
            { 
                "id": "weekly_icu_admissions",
                "type": "numeric"
            },
            { 
                "id": "weekly_icu_admissions_per_million",
                "type": "numeric"
            },
            { 
                "id": "weekly_hosp_admissions",
                "type": "numeric"
            },
            { 
                "id": "weekly_hosp_admissions_per_million",
                "type": "numeric"
            },
            { 
                "id": "new_tests",
                "type": "numeric"
            },
            { 
                "id": "total_tests",
                "type": "numeric"
            },
            { 
                "id": "total_tests_per_thousand",
                "type": "numeric"
            },
            { 
                "id": "new_tests_per_thousand",
                "type": "numeric"
            },
            { 
                "id": "new_tests_smoothed",
                "type": "numeric"
            },
            { 
                "id": "new_tests_smoothed_per_thousand",
                "type": "numeric"
            },
            { 
                "id": "positive_rate",
                "type": "numeric"
            },
            { 
                "id": "tests_per_case",
                "type": "numeric"
            },
            { 
                "id": "tests_units",
                "type": "numeric"
            },
            { 
                "id": "total_vaccinations",
                "type": "numeric"
            },
            { 
                "id": "people_vaccinated",
                "type": "numeric"
            },
            { 
                "id": "people_fully_vaccinated",
                "type": "numeric"
            },
            { 
                "id": "total_boosters",
                "type": "numeric"
            },
            { 
                "id": "new_vaccinations",
                "type": "numeric"
            },
            { 
                "id": "new_vaccinations_smoothed",
                "type": "numeric"
            },
            { 
                "id": "total_vaccinations_per_hundred",
                "type": "numeric"
            },
            { 
                "id": "people_vaccinated_per_hundred",
                "type": "numeric"
            },
            { 
                "id": "people_fully_vaccinated_per_hundred",
                "type": "numeric"
            },
            { 
                "id": "total_boosters_per_hundred",
                "type": "numeric"
            },
            { 
                "id": "new_vaccinations_smoothed_per_million",
                "type": "numeric"
            },
            { 
                "id": "stringency_index",
                "type": "numeric"
            },
            { 
                "id": "population",
                "type": "numeric"
            },
            { 
                "id": "population_density",
                "type": "numeric"
            },
            { 
                "id": "median_age",
                "type": "numeric"
            },
            { 
                "id": "aged_65_older",
                "type": "numeric"
            },
            { 
                "id": "aged_70_older",
                "type": "numeric"
            },
            { 
                "id": "gdp_per_capita",
                "type": "numeric"
            },
            { 
                "id": "extreme_poverty",
                "type": "numeric"
            },
            { 
                "id": "cardiovasc_death_rate",
                "type": "numeric"
            },
            { 
                "id": "diabetes_prevalence",
                "type": "numeric"
            },
            { 
                "id": "female_smokers",
                "type": "numeric"
            },
            { 
                "id": "male_smokers",
                "type": "numeric"
            },
            { 
                "id": "handwashing_facilities",
                "type": "numeric"
            },
            { 
                "id": "hospital_beds_per_thousand",
                "type": "numeric"
            },
            { 
                "id": "life_expectancy",
                "type": "numeric"
            },
            { 
                "id": "human_development_index",
                "type": "numeric"
            },
            { 
                "id": "excess_mortality",
                "type": "numeric"
            }
        ]

    response = datastore_create(records, fields, ckan_url, countries_id, api_key)
    print(response)

datastore_delete(ckan_url, countries_id, api_key)
countries_data()