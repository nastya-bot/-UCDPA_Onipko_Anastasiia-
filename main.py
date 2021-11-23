from typing import Dict, Any

import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import geopandas as gpd

# Importing national charging registry CSV file with API
url = 'http://chargepoints.dft.gov.uk/api/retrieve/registry/format/csv/'
filename_chargers = '/tmp/national-charge-point-registry.csv'
r = requests.get(url)
open(filename_chargers, 'wb').write(r.content)

# Read national charging registry CSV file
data_ev_chargers = pd.read_csv(filename_chargers, lineterminator='\n', low_memory=False,
                               usecols=['chargeDeviceID', 'latitude', 'longitude', 'chargeDeviceStatus', 'dateCreated'])

# Importing ods file with ULEVs registered for the first time by fuel type in the UK data
filename_cars = 'veh0171.ods'
data_ev_cars = pd.read_excel(filename_cars, engine='odf')

# visualize first 5 rows
print(data_ev_chargers.head())
print(data_ev_cars.head())

# setting the values for displayed width, number columns and rows to avoid rows / columns truncation
desired_width = 640
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', None)

# printing info about DataFrames
print(data_ev_chargers.info())
print(data_ev_cars.info())

# checking for missing values in charges and cars data frames
print(data_ev_chargers.isna().any())
print(data_ev_cars.isna().any())

# chargers: display dateCreated rows in ascending order to review and clean the data
data_ev_chargers_sorted = data_ev_chargers.sort_values(by='dateCreated', ascending=True)
print(data_ev_chargers_sorted.head(100))

# chargers: dropping rows that have dateCreated == 0000-00-00 00:00:0
index_list = data_ev_chargers[(data_ev_chargers['dateCreated'] == '0000-00-00 00:00:00')].index
data_ev_chargers.drop(index_list, inplace=True)

# checking for duplicated values
print(data_ev_chargers.duplicated().sum())
print(data_ev_cars.duplicated().sum())

# chargers: add new column yearCreated extracting year from dateCreated
data_ev_chargers['yearCreated'] = pd.DatetimeIndex(data_ev_chargers['dateCreated']).year

# chargers: create a new dataframe with number chargers installed each year and rolling summary
chargers_installed = data_ev_chargers.groupby('yearCreated')['chargeDeviceID'].count().reset_index(name='num_charges')
chargers_installed['rollingSUM'] = chargers_installed['num_charges'].cumsum()
print(chargers_installed)

# chargers: displaying modified data_ev_chargers table info
print(data_ev_chargers.info())

# cars: add new column that combines other Ultra low emission vehicles (ULEVs)
data_ev_cars['otherULEVs'] = data_ev_cars.iloc[:, 3:7].sum(axis=1)

# cars: adding cum_sum column of BEV + PHEV Cars
data_ev_cars['plug_rollingSUM'] = data_ev_cars.iloc[:, 1:3].sum(axis=1).cumsum()

# ---- visualisations ---- #

# filter latitude and longitude columns to be in a UK range
map_ev_chargers = data_ev_chargers[(data_ev_chargers['latitude'].between(49, 61)) & (
    data_ev_chargers['longitude'].between(-8.1, 2))]

# create scattered plot of chargers on UK map
fig, ax = plt.subplots(figsize=(8, 6))

# plot map from gpd on axis
countries = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
countries[countries['name'] == 'United Kingdom'].plot(color='lightgrey', ax=ax)
# plot ev_chargers points
map_ev_chargers.plot(x='longitude', y='latitude', kind='scatter', title='EV Chargers location distribution in the UK',
                     ax=ax)
# add grid
ax.grid(b=True, alpha=0.5)

# create a bar chart with number of EV Chargers installed each year
fig, ax = plt.subplots(figsize=(12, 8))

# set axis
x = chargers_installed['yearCreated']
y = chargers_installed['num_charges']

# parse dates for plot's title
first_year = chargers_installed['yearCreated'].min()
last_year = chargers_installed['yearCreated'].max()

# adding values to each bar
for index, value in zip(x, y):
    plt.text(index, value, str(value))

# make the bar plot
ax.bar(x, y, color='#24536c')

# chart settings
ax.set_xticks(x)
ax.set_xlabel('Year')
ax.set_ylabel('Number chargers installed')
ax.set_title(f'Number EV charges installed between {first_year} and {last_year}')

# create stacked bar chart of ULEVs cars first registered by year

fig, ax = plt.subplots(figsize=(12, 8))

# set axis
x = data_ev_cars['Date']
y1 = data_ev_cars['Battery Electric']
y2 = data_ev_cars['Plug-in Hybrid Electric 2']
y3 = data_ev_cars['otherULEVs']

# make the bar plot
ax.bar(x, y1, bottom=y2 + y3, label='BEV')
ax.bar(x, y2, bottom=y3, label='PHEV')
ax.bar(x, y3, label='other ULEVs')

# chart settings
ax.set_xticks(x)
ax.set_xlabel('Year', fontsize=14)
ax.set_ylabel('Number cars', fontsize=14)
ax.set_title('ULEVs registered for the first time in the UK')
ax.legend()

# line chart BEV + PHEV registered to EV-Chargers installed per year

# merge cars and chargers data frames for the plot
ev_merged = chargers_installed.merge(data_ev_cars, how='inner', left_on='yearCreated', right_on='Date')

# create plot
fig, ax = plt.subplots(figsize=(12, 8))

# set axis
x = ev_merged['Date']
y1 = ev_merged['num_charges']
y2 = ev_merged['Battery Electric'] + ev_merged['Plug-in Hybrid Electric 2']

# parse dates for plot's title
first_year = ev_merged['Date'].min()
last_year = ev_merged['Date'].max()

ax.plot(x, y1, label="charges_installed")
ax.plot(x, y2, label="plug_cars_registered")

# chart settings
ax.set_xlabel('Year')
ax.set_ylabel('Number')
ax.set_title(f'Charges installed vs Plug-In Cars registered between {first_year} and {last_year}')
ax.legend()

plt.show()

# analysis of the cars to chargers ratio

# NumPy array of cars rolling summary
plug_array = np.array(ev_merged['plug_rollingSUM'])

# create new df filtered to chargers In Service and installed before 2021
ev_chargers_in_service = data_ev_chargers[
    (data_ev_chargers['chargeDeviceStatus'] == 'In service') & (data_ev_chargers['yearCreated'] != 2021)]

# adding columns with number of chargers in service and rolling summary
ev_chargers_in_service = ev_chargers_in_service.groupby('yearCreated')['chargeDeviceID'].count().reset_index(
    name='num_charges_in_service')

ev_chargers_in_service['rollingSUM_in_service'] = ev_chargers_in_service['num_charges_in_service'].cumsum()
chargers_avail = np.array(ev_chargers_in_service['rollingSUM_in_service'])

# calculating the ratio of cars to chargers
ratio = plug_array / chargers_avail

# creating a dictionary for ratio in 2020
ratio_2020_dict: Dict[str, Any] = {'2020': ratio[8]}

print(ratio_2020_dict)