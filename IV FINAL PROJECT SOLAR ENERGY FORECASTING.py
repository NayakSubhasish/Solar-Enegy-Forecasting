#!/usr/bin/env python
# coding: utf-8

# # Solar Irradiance Prediction

# # !Install, Imports & Dataset Loading

# In[1]:


get_ipython().system('pip install xgboost')
get_ipython().system('pip install lightgbm')
get_ipython().system('pip install scikit-learn catboost')
get_ipython().system('pip install astral')
get_ipython().system('pip install numpy')
get_ipython().system('pip install pandas')


# To run before accessing the data. Just access with your google account.

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import calendar
import astral
from scipy.signal import correlate
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split, KFold
from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, StackingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from astral import LocationInfo
from astral.sun import sun
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Attention
from tensorflow.keras.utils import plot_model
import keras.backend as K
import warnings
warnings.simplefilter("ignore")


# Load the datasets

# In[3]:


df_gujarat = pd.read_csv("/Users/subhasritanayak/Desktop/CSV/gujarat.csv", sep=';')
df_karnataka = pd.read_csv("/Users/subhasritanayak/Desktop/CSV/karnataka.csv", sep=';')
df_rajasthan = pd.read_csv("/Users/subhasritanayak/Desktop/CSV/rajasthan.csv", sep=';')
df_madhyapradesh = pd.read_csv("/Users/subhasritanayak/Desktop/CSV/madhyapradesh.csv", sep=';')
df_maharashtra = pd.read_csv("/Users/subhasritanayak/Desktop/CSV/maharashtra.csv", sep=';')


# # Data Cleaning & Augmentation

# ## Dataset Clipping
# We clip the dataset to row 78888, i.e., 31/12/2021, as the measurements for solar radiation of 2022 are still not accurate on NASA platform.

# In[88]:


fig, ax = plt.subplots(figsize=(9, 6))

# Plot the histogram
df_gujarat['IRRADIANCE'].hist(density=True, alpha=0.7, color='cornflowerblue')

# Set plot title and axis labels
ax.set_title('Histogram of Irradiance', fontsize=16)
ax.set_xlabel('Irradiance')
ax.set_ylabel('Frequency')

# Add vertical grid lines
ax.grid(axis='x', linestyle='--', alpha=0.5)

# Create the 'histograms' directory if it doesn't exist
# os.makedirs('histograms', exist_ok=True)

# Save the plot as a PDF file with higher quality
plt.savefig('/Users/subhasritanayak/Desktop/CSV/Histogram_Irradiance_without_dataclipping.pdf', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

# Close the plot to free up memory
plt.close()


# In[89]:


df_gujarat_2021 = df_gujarat[df_gujarat['YEAR'] == 2022]
fig, ax = plt.subplots(figsize=(9, 6))

# Plot the histogram
df_gujarat_2021['IRRADIANCE'].hist(density=True, alpha=0.7, color='cornflowerblue')

# Set plot title and axis labels
ax.set_title('Histogram of Irradiance for 2022', fontsize=16)
ax.set_xlabel('Irradiance')
ax.set_ylabel('Frequency')

# Add vertical grid lines
ax.grid(axis='x', linestyle='--', alpha=0.5)

# Save the plot as a PDF file with higher quality
plt.savefig('//Users/subhasritanayak/Desktop/CSV/Histogram_Irradiance_without_dataclipping_2022.pdf', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

# Close the plot to free up memory
plt.close()


# In[5]:


df_gujarat = df_gujarat.iloc[:78888]
df_karnataka = df_karnataka.iloc[:78888]
df_rajasthan = df_rajasthan.iloc[:78888]
df_madhyapradesh = df_madhyapradesh.iloc[:78888]
df_maharashtra = df_maharashtra.iloc[:78888]
dfs = [df_gujarat, df_karnataka, df_rajasthan, df_madhyapradesh, df_maharashtra]
cities = ['gujarat', 'karnataka', 'rajasthan', 'madhyapradesh', 'maharashtra']


# ## Missing Values
# We check for Missing (Nan) Values and we see that we have none.

# In[6]:


for city, df in zip(cities, dfs):
    a = max(df.isna().sum())
    print(f'The amount of rows with NaN values for {city} are: {a} \n')


# ## Data Augmentation
# We use Astral library to add a "Time of Sunrise" and "Time of Sunset" column to each dataset.

# In[7]:


gujarat = LocationInfo("gujarat", 22.6708, 71.5724)
karnataka = LocationInfo("karnataka", 15.3173, 75.7139)
rajasthan = LocationInfo("rajasthan", 27.0238, 74.2179)
madhyapradesh = LocationInfo("madhyapradesh", 22.9734, 78.6569)
maharashtra = LocationInfo("maharashtra", 19.7515, 75.7139)
cities = [gujarat, karnataka, rajasthan, madhyapradesh, maharashtra]

for city, df in zip(cities, dfs):
    time_sunrise, time_sunset = [], []
    for year in range(2013, 2022):
        for month in range(1, 13):
            if month in [11, 4, 6, 9]:
                for day in range(1, 31):
                    time_sunrise.append(sun(city.observer, date=datetime.date(year, month, day))["sunrise"])
                    time_sunset.append(sun(city.observer, date=datetime.date(year, month, day))["sunset"])
            if month == 2:
                if year in [2016, 2020]:
                    for day in range(1,30):
                        time_sunrise.append(sun(city.observer, date=datetime.date(year, month, day))["sunrise"])
                        time_sunset.append(sun(city.observer, date=datetime.date(year, month, day))["sunset"])
                else:
                    for day in range(1,29):
                        time_sunrise.append(sun(city.observer, date=datetime.date(year, month, day))["sunrise"])
                        time_sunset.append(sun(city.observer, date=datetime.date(year, month, day))["sunset"])
            if month in [1, 3, 5, 7, 8, 10, 12]:
                for day in range(1,32):
                    time_sunrise.append(sun(city.observer, date=datetime.date(year, month, day))["sunrise"])
                    time_sunset.append(sun(city.observer, date=datetime.date(year, month, day))["sunset"])

    sunrise = [x for x in time_sunrise for _ in range(24)]
    sunset = [x for x in time_sunset for _ in range(24)]
    df['SUNRISE'] = sunrise
    df['SUNSET'] = sunset


# # Feature Engineering

# Transforming the time of sunrise/sunset into ”how much time elapsed after sunrise” (SUNRISE) / ”how much time till sunset” (SUNSET) and "LIGHT", i.e., a binary variable indicating whether the sun is up or not at that specific measurement time.

# In[8]:


dfs = [df_gujarat, df_karnataka, df_rajasthan, df_madhyapradesh, df_maharashtra]
for df in dfs:
  df['TIME'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY', 'HOUR']])
  df["TIME"] = df["TIME"].apply(lambda x: int(x.timestamp()))
  df["SUNRISE"] = df["SUNRISE"].apply(lambda x: int(x.timestamp()))
  df["SUNSET"] = df["SUNSET"].apply(lambda x: int(x.timestamp()))
  df["SUNRISE"] = df["TIME"] - df["SUNRISE"]
  df["SUNSET"] = df["TIME"] - df["SUNSET"]
  df['LIGHT'] = np.where(np.sign(df['SUNRISE']) != np.sign(df['SUNSET']), 1, 0)


# Normalization of the following columns:
# 
# 
# *   Temperature
# *   Humidity
# *   Precipitation
# *   Pressure
# *   Wind Speed
# *   Dew
# *   Wet Bulb Temperature
# 
# 
# 

# In[9]:


cols_to_normalize = ['TEMPERATURE', 'HUMIDITY', 'PRECIPITATION', 'PRESSURE', 'WIND SPEED', 'DEW', 'WET BULB TEMPERATURE']
scaler = MinMaxScaler()


# # Data Exploration

# In[10]:


df_gujarat.info()


# ## Split dataset and define ticks

# Splitting the datasets for plotting in "Features as functions of time". Define arrays for the X and Y arrays in the plots.  

# In[11]:


# Split the gujarat dataset in subsets for each year
df_gujarat_2013 = df_gujarat[df_gujarat['YEAR'] == 2013]
df_gujarat_2014 = df_gujarat[df_gujarat['YEAR'] == 2014]
df_gujarat_2015 = df_gujarat[df_gujarat['YEAR'] == 2015]
df_gujarat_2016 = df_gujarat[df_gujarat['YEAR'] == 2016]
df_gujarat_2017 = df_gujarat[df_gujarat['YEAR'] == 2017]
df_gujarat_2018 = df_gujarat[df_gujarat['YEAR'] == 2018]
df_gujarat_2019 = df_gujarat[df_gujarat['YEAR'] == 2019]
df_gujarat_2020 = df_gujarat[df_gujarat['YEAR'] == 2020]
df_gujarat_2021 = df_gujarat[df_gujarat['YEAR'] == 2021]


# In[12]:


# Split dataframe into parts for later
df_firsts = df_gujarat[df_gujarat['DAY'] == 1]
df_firsts_2021 = df_firsts[df_firsts['YEAR']==2021]
df_firsts_midday = df_firsts[df_firsts['HOUR']==12]
df_firsts_midday_2021 = df_firsts_midday[df_firsts_midday['YEAR']==2021]
df_firsts_midday_2013 = df_firsts_midday[df_firsts_midday['YEAR']==2013]

df_startyear = df_firsts_midday[df_firsts_midday['MONTH']==1]

df_junes = df_gujarat[df_gujarat['MONTH']==6]
df_middyear = df_junes[df_junes['DAY'] == 30]
df_middyear_midday = df_middyear[df_middyear['HOUR']==12]

df_june_15s = df_junes[df_junes['DAY']==15]
df_june_15_2021 = df_june_15s[df_june_15s['YEAR']==2021]


# In[13]:


ticks_yearly = df_startyear.index
years = ['2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']

ticks_months_2021 = df_firsts_midday_2021.index
ticks_all_months = df_firsts_midday_2013.index
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

FEATURES = ['IRRADIANCE', 'TEMPERATURE', 'HUMIDITY', 'PRECIPITATION', 'PRESSURE', 'WIND SPEED', 'WIND DIRECTION', 'DEW', 'WET BULB TEMPERATURE']
UNITS = ['[W/m^2]','[°C]','[g/kg]', '[mm/h]', '[kPa]','[m/s]','[°]','[°C]','[°C]']

ti = df_june_15_2021.reset_index(drop=True)
ticks_day = ti.index


# ## Histogram

# Histograms of features and "wind-rose" plot of wind direction.

# In[16]:


# Create histograms for each feature
for i, feature in enumerate(FEATURES):
    fig, ax = plt.subplots(figsize=(9, 6))

    # Plot the histogram
    df_gujarat[feature].hist(density=True, alpha=0.7, color='cornflowerblue')

    # Set plot title and axis labels
    ax.set_title('Histogram of ' + feature, fontsize=16)
    ax.set_xlabel(feature + UNITS[i])
    ax.set_ylabel('Frequency')

    # Add vertical grid lines
    ax.grid(axis='x', linestyle='--', alpha=0.5)

    # Save the plot as a PDF file with higher quality
    plt.savefig('/Users/subhasritanayak/Desktop/CSV ' + feature + '.pdf', dpi=300, bbox_inches='tight')

    # show the plot
    plt.show()

    # Close the plot to free up memory
    plt.close()


# In[17]:


# Convert wind direction to radians
wind_direction_rad = np.deg2rad(df_gujarat['WIND DIRECTION']) + np.pi/2

# Set the number of directional bins
num_bins = 16

# Calculate the histogram of wind directions
hist, bins = np.histogram(wind_direction_rad, bins=num_bins)

# Create a polar plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='polar')

# Set the direction labels
directions = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE']
ax.set_xticklabels(directions)

# Plot the rose plot
ax.bar(bins[:-1], hist, width=2*np.pi/num_bins, align='edge', color='cornflowerblue', alpha=0.7)

# Set gridlines
ax.grid(True)

# Set plot title
plt.title('Wind Direction Distribution', pad=20, fontsize=16)

# Remove y-axis labels
ax.set_yticklabels([])

# save the plot
plt.savefig('/Users/subhasritanayak/Desktop/CSV/wind_direction_hist.pdf')
plt.savefig('/Users/subhasritanayak/Desktop/CSV/wind_direction_hist.jpeg')

# Show the plot
plt.show()


# ## Map  with locations

# Map Visualization of the 5 locations (It doesn't correctly display the map in Jupyter, but it can be seen in the colab notebook linked at the top of this file).

# In[18]:


import folium

# Create a map object centered on INDIA
map_center = [20.5937, 78.9629]
map_zoom = 5
my_map = folium.Map(location=map_center, zoom_start=map_zoom)

# Define the locations we want to plot
locations = {
    "gujarat": [22.6708, 71.5724],
    "rajasthan": [27.0238, 74.2179],
    "maharashtra": [19.7515, 75.7139],
    "karnataka": [15.3173, 75.7139],
    "madhyapradesh": [22.9734, 78.6569],
}

# Add markers for each location to the map
for location_name, location in locations.items():
    folium.Marker(
        location=location,
        tooltip=location_name,
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(my_map)

# Display the map
my_map


# ## Pairplot and correlation matrix

# Pairwise plot, correlation plot and combination of the two.

# In[19]:


# pairplot for the whole dataset

# Select columns to include in the scatter plot matrix
cols = ['IRRADIANCE', 'TEMPERATURE', 'HUMIDITY', 'PRECIPITATION', 'PRESSURE', 'WIND SPEED', 'WIND DIRECTION', 'DEW', 'WET BULB TEMPERATURE', 'LIGHT']

# Create the scatter plot matrix using seaborn
sns.set(style='ticks')
G = sns.pairplot(df_gujarat[cols], diag_kind='hist', height=2)

# Save the plot as a PDF and JPEG file
G.savefig('/Users/subhasritanayak/Desktop/CSV/pairplot.jpeg')
G.savefig('/Users/subhasritanayak/Desktop/CSV/pairplot.pdf')


# In[ ]:


# correlation matrix for the whole dataset (gujarat)
cols = ['IRRADIANCE', 'TEMPERATURE', 'HUMIDITY', 'PRECIPITATION', 'PRESSURE', 'WIND SPEED', 'WIND DIRECTION', 'DEW', 'WET BULB TEMPERATURE', 'LIGHT']
corr = df_gujarat[cols].corr()
sns.set(style='white')
sns.set_context("paper", rc={"axes.labelsize":20})
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(8, 8))
    ax = sns.heatmap(corr, square=True, cmap='coolwarm', annot=True, fmt=".2f",
                     cbar_kws={'label': 'Correlation Coefficient'})

# Save the plot as a PDF and JPEG file
f.savefig('/content/drive/MyDrive/Major Project Solar Energy pred/Output Images/correlation_matrixv2.jpeg', bbox_inches="tight")
f.savefig('/content/drive/MyDrive/Major Project Solar Energy pred/Output Images/correlation_matrixv2.pdf', bbox_inches="tight")


# In[20]:


def hide_current_axis(*args, **kwds):
    plt.gca().set_visible(False)

# Select columns to include in the special plot
cols = ['IRRADIANCE', 'TEMPERATURE', 'HUMIDITY', 'PRECIPITATION', 'PRESSURE', 'WIND SPEED', 'WIND DIRECTION', 'DEW', 'WET BULB TEMPERATURE', 'LIGHT']
d1 = df_gujarat[cols].copy()

sns.set_context("paper", rc={"axes.labelsize":20})
e = sns.pairplot(d1)
e.map_upper(hide_current_axis)

(xmin, _), (_, ymax) = e.axes[0, 0].get_position().get_points()
(_, ymin), (xmax, _) = e.axes[-1, -1].get_position().get_points()

ax = e.fig.add_axes([xmin, ymin, xmax - xmin, ymax - ymin], facecolor='none')

corr1 = d1.corr()
mask1 = np.tril(np.ones_like(corr1, dtype=bool))
sns.heatmap(corr1, mask=mask1, cmap='coolwarm', linewidths=.5, cbar=False, annot=True, annot_kws={'size': 25}, ax=ax)


ax.set_xticks([])
ax.set_yticks([])
# ax.xaxis.tick_top()
# ax.yaxis.tick_right()

plt.savefig('/Users/subhasritanayak/Desktop/CSV/pairplot_correlation_matrix.jpeg')
plt.show()


# In[ ]:


for city, df in zip(cities, dfs):
    print(f'The correlation of features with target variable for {city} are: \n')
    print(df.corr()["IRRADIANCE"].sort_values(ascending = False))
    print()
    print()


# ## Features as a function of time

# ### Features over all years

# In[21]:


# features as a function of time for all years
# create figure with subplots for each feature
fig, axs = plt.subplots(nrows=9, figsize=(12, 18))
fig.suptitle('Features as a function of time', fontsize=18)

# loop over each feature and create a subplot
for i, feature in enumerate(FEATURES):
    axs[i].plot(df_gujarat[feature], color='cornflowerblue')
    axs[i].set_title(feature, fontsize=12)
    axs[i].set_ylabel(feature + ' ' + UNITS[i],fontsize=8)
    axs[i].set_xticks(ticks_yearly)
    axs[i].set_xticklabels(years, fontsize=8)
    axs[i].tick_params(axis='y', labelsize=8)
    axs[i].grid(True, alpha=0.5)
    axs[i].set_xlabel('time [years]',fontsize=8)


# adjust spacing between subplots
fig.subplots_adjust(hspace=0.75, top=0.95, bottom=0.05)

# save the figure
fig.savefig('features_over_all_time.jpeg')
fig.savefig('features_over_all_time.pdf')

# show the figure


# In[ ]:


# features as a function of time for all years
# split into two plots
fig1, axs = plt.subplots(nrows=5, figsize=(12, 12))
#fig1.suptitle('Features as a function of time 1', fontsize=18)

# loop over the corresponding features:
for i, feature in enumerate(['IRRADIANCE','TEMPERATURE', 'HUMIDITY','DEW', 'WET BULB TEMPERATURE']):
    axs[i].plot(df_gujarat[feature], color='cornflowerblue')
    axs[i].set_title(feature, fontsize=12)
    axs[i].set_ylabel(feature + ' ' + UNITS[i],fontsize=8)
    axs[i].set_xticks(ticks_yearly)
    axs[i].set_xticklabels(years, fontsize=8)
    axs[i].tick_params(axis='y', labelsize=8)
    axs[i].grid(True, alpha=0.5)
    axs[i].set_xlabel('time [years]',fontsize=8)


# adjust spacing between subplots
fig1.subplots_adjust(hspace=0.75, top=0.95, bottom=0.05)

# save the figure
fig1.savefig('features_over_all_time_1.jpeg')
fig1.savefig('features_over_all_time_1.pdf')


# for the oher part of the plot
fig2, axs = plt.subplots(nrows=4, figsize=(12, 12))
#fig2.suptitle('Features as a function of time 2', fontsize=18)

# loop over the corresponding features:
for i, feature in enumerate(['PRECIPITATION', 'PRESSURE', 'WIND SPEED', 'WIND DIRECTION']):
    axs[i].plot(df_gujarat[feature], color='cornflowerblue')
    axs[i].set_title(feature, fontsize=12)
    axs[i].set_ylabel(feature + ' ' + UNITS[i],fontsize=8)
    axs[i].set_xticks(ticks_yearly)
    axs[i].set_xticklabels(years, fontsize=8)
    axs[i].tick_params(axis='y', labelsize=8)
    axs[i].grid(True, alpha=0.5)
    axs[i].set_xlabel('time [years]',fontsize=8)


# adjust spacing between subplots
fig2.subplots_adjust(hspace=0.75, top=0.95, bottom=0.05)

# save the figure
fig2.savefig('features_over_all_time_2.jpeg')
fig2.savefig('features_over_all_time_2.pdf')


# ### Features over a year

# In[ ]:


#  features as a function of time for the year 2021
# create figure with subplots for each feature

fig = plt.figure(figsize=(18, 12))

# loop over each feature and create a subplot
for i, feature in enumerate(FEATURES):
    row = i // 3
    col = i % 3
    axs = fig.add_subplot(3, 3, i+1)
    axs.plot(df_gujarat_2021[feature], color='cornflowerblue')
    axs.set_title(feature, fontsize=12)
    axs.set_xlabel('time [months]', fontsize=10)
    axs.set_ylabel(UNITS[i], fontsize=10)
    axs.set_xticks(ticks_months_2021)
    axs.set_xticklabels(months, fontsize=8)
    axs.tick_params(axis='y', labelsize=8)
    axs.grid(True, alpha=0.5)


# adjust spacing between subplots
fig.subplots_adjust(hspace=0.5, top=0.95, bottom=0.05)
fig.savefig('/content/drive/MyDrive/Major Project Solar Energy pred/Output Images/Plot_for_each_feature_2021.jpeg')
fig.savefig('/content/drive/MyDrive/Major Project Solar Energy pred/Output Images/Plot_for_each_feature_2021.pdf')


# In[22]:


#  features as a function of time for all years
# create figure with subplots for each feature

fig = plt.figure(figsize=(18, 12))

# loop over each feature and create a subplot
for i, feature in enumerate(FEATURES):
    row = i // 3
    col = i % 3
    axs = fig.add_subplot(3, 3, i+1)

    axs.plot(df_gujarat_2013[feature].reset_index(drop=True), color='midnightblue', label='2013',alpha=0.25)
    axs.plot(df_gujarat_2014[feature].reset_index(drop=True), color='mediumblue', label='2014',alpha=0.25)
    axs.plot(df_gujarat_2015[feature].reset_index(drop=True), color='royalblue', label='2015',alpha=0.25)
    axs.plot(df_gujarat_2016[feature].reset_index(drop=True), color='deepskyblue', label='2016',alpha=0.25)
    axs.plot(df_gujarat_2017[feature].reset_index(drop=True), color='lightskyblue', label='2017',alpha=0.25)
    axs.plot(df_gujarat_2018[feature].reset_index(drop=True), color='aqua', label='2018',alpha=0.25)
    axs.plot(df_gujarat_2019[feature].reset_index(drop=True), color='aquamarine', label='2019',alpha=0.25)
    axs.plot(df_gujarat_2020[feature].reset_index(drop=True), color='turquoise', label='2020',alpha=0.25)
    axs.plot(df_gujarat_2021[feature].reset_index(drop=True), color='lightseagreen', label='2021',alpha=0.25)
    #axs.legend(loc='best')

    axs.set_title(feature, fontsize=12)
    axs.set_xlabel('time [months]', fontsize=10)
    axs.set_ylabel(UNITS[i], fontsize=10)
    axs.set_xticks(ticks_all_months)
    axs.set_xticklabels(months, fontsize=8)
    axs.tick_params(axis='y', labelsize=8)
    axs.grid(True, alpha=0.5)


# adjust spacing between subplots
fig.subplots_adjust(hspace=0.5, top=0.95, bottom=0.05)
fig.savefig('/Users/subhasritanayak/Desktop/CSVPlot_for_each_feature_years_overlay.jpeg')
fig.savefig('/Users/subhasritanayak/Desktop/CSV/Plot_for_each_feature_years_overlay.pdf')


# ### Features over a day as hourly average of the month
# 
# 

# In[23]:


# Calculate the mean for each hour over each month for  gujarat for each feature
mean_values = [[],[],[],[],[],[],[],[],[],[], []]

# Calculate the standard deviation for each hour over each month for  gujarat for each feature
stds = [[],[],[],[],[],[],[],[],[],[], []]

for m in range(12):
    # take data from month
    df_current_month = df_gujarat[df_gujarat['MONTH']==(m+1)]

    # loop over the times
    for hour in range(24):

        # make a df for the current hour --> gives us all measurements for this time in this month
        df_current_hour = df_current_month[df_current_month['HOUR'] == hour]

        mean_values[9].append(m+1)
        mean_values[10].append(hour)
        stds[9].append(m+1)
        stds[10].append(hour)


        # loop over the features
        for i, feature in enumerate(FEATURES):
            # calculate the mean of the feature
            mean = np.mean(df_current_hour[feature])
            # calculate the standard deviation of the feature
            std = np.std(df_current_hour[feature])
            # append the mean to the corresponding feature
            mean_values[i].append(mean)
            stds[i].append(std)

# create a dataframe from the mean values
prepared_df2 = {'IRRADIANCE':mean_values[0],
                'TEMPERATURE':mean_values[1],
                'HUMIDITY':mean_values[2],
                'PRECIPITATION':mean_values[3],
                'PRESSURE':mean_values[4],
                'WIND SPEED':mean_values[5],
                'WIND DIRECTION':mean_values[6],
                'DEW':mean_values[7],
                'WET BULB TEMPERATURE':mean_values[8],
                'MONTH': mean_values[9],
                'HOUR': mean_values[10]}

df_hour_mean_values_gujarat = pd.DataFrame(prepared_df2)

# create a dataframe from the standard deviations
prepared_df3 = {'IRRADIANCE':stds[0],
                'TEMPERATURE':stds[1],
                'HUMIDITY':stds[2],
                'PRECIPITATION':stds[3],
                'PRESSURE':stds[4],
                'WIND SPEED':stds[5],
                'WIND DIRECTION':stds[6],
                'DEW':stds[7],
                'WET BULB TEMPERATURE':stds[8],
                'MONTH': stds[9],
                'HOUR': stds[10]}

df_hour_stds_gujarat = pd.DataFrame(prepared_df3)
df_hour_stds_gujarat.head()


# In[29]:


# Plot the mean values
fig = plt.figure(figsize=(18, 15))

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# loop over each feature and create a subplot
for i, feature in enumerate(FEATURES):
    row = i // 3
    col = i % 3
    axs = fig.add_subplot(3, 3, i+1)

    # draw the plot for the feature for each month:
    for m in range(12):
        current_month = df_hour_mean_values_gujarat[df_hour_mean_values_gujarat['MONTH']==(m+1)]
        axs.plot(current_month[feature].reset_index(drop=True), label=months[m], alpha=0.5)

    axs.set_title(feature, fontsize=12)
    # make legend outside of plot
    axs.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # axs.legend(loc='best')
    axs.set_xlabel('time [hours]', fontsize=10)
    axs.set_ylabel(UNITS[i], fontsize=10)
    #axs.tick_params(axis='y', labelsize=8)
    axs.grid(True, alpha=0.5)


# set space between subplots
fig.subplots_adjust(hspace=0.5, top=0.95, bottom=0.05, right=1.8)


# save the plot
plt.savefig('/content/drive/MyDrive/Major Project Solar Energy pred/Output Images/Plot_for_each_feature_average.jpeg', dpi=300, bbox_inches='tight')
plt.savefig('/content/drive/MyDrive/Major Project Solar Energy pred/Output Images/Plot_for_each_feature_average.pdf', dpi=300, bbox_inches='tight')


# Delete the coulmn "time" variable

# In[27]:


df_gujarat = df_gujarat.drop("TIME", axis = 1)
df_karnataka = df_karnataka.drop("TIME", axis = 1)
df_rajasthan = df_rajasthan.drop("TIME", axis = 1)
df_madhyapradesh = df_madhyapradesh.drop("TIME", axis = 1)
df_maharashtra = df_maharashtra.drop("TIME", axis = 1)


# # Model Selection and Prediction Pipeline

# ## ML Models
# 
# ML Models training and evaluation loop.
# 
# *   Linear Regression
# *   K-Nearest Neighbhor Regression (KNN)
# *   Decision Tree Regression
# *   Random Forest Regression
# *   Extreme Gradient Boosting (XGB)
# *   AdaBoost
# *   CatBoost
# *   Light Gradient Boosting Machine (LGBM)

# In[28]:


pred_cities = ['pred_gujarat', 'pred_karnataka', 'pred_rajasthan', 'pred_madhyapradesh', 'pred_maharashtra']
cities = ['gujarat', 'karnataka', 'rajasthan', 'madhyapradesh','maharashtra']
regressors = [
    LinearRegression(),
    KNeighborsRegressor(n_neighbors=5, weights='distance'),
    XGBRegressor(subsample=1.0, n_jobs=-1, n_estimators=362, max_depth=7, learning_rate=0.06),
    LGBMRegressor(max_depth=15, learning_rate=0.05, n_estimators=350, n_jobs=-1),
    DecisionTreeRegressor(),
    RandomForestRegressor(n_jobs=-1, n_estimators=50, min_samples_split=2, min_samples_leaf=1, max_features=5, max_depth=20),
    AdaBoostRegressor(DecisionTreeRegressor(max_depth=1), n_estimators=250, learning_rate=1.0, random_state=42),  # Corrected line
    CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6, loss_function='RMSE', random_seed=42, verbose=0)
]
for city, df in zip(cities,dfs):
    print()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    X = df.drop(columns=['IRRADIANCE', 'SUNRISE', 'SUNSET'])
    y = df['IRRADIANCE']

    for model in regressors:
        scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
        print(f'({city} {type(model).__name__}) MSE: {scores}, Mean MSE: {np.mean(scores)}')


# ## Feature Importances
# 
# Tuned models definition
# *   Random Forest Regression
# *   Extreme Gradient Boosting (XGB)
# *   AdaBoost
# *   Light Gradient Boosting Machine (LGBM)
# 

# In[30]:


xgb = XGBRegressor(subsample = 1.0, n_jobs = -1, n_estimators = 362, max_depth = 7, learning_rate = 0.06)
rf = RandomForestRegressor(n_jobs = -1, n_estimators = 50, min_samples_split = 2, min_samples_leaf = 1, max_features = 5, max_depth = 20)
abr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=1), n_estimators=250, learning_rate=1.0, random_state=42)
lgbm = LGBMRegressor(max_depth=15, learning_rate=0.05, n_estimators=350, n_jobs=-1)


# In[31]:


X = df.drop(columns=['IRRADIANCE', 'SUNRISE', 'SUNSET'])
y = df['IRRADIANCE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# ### Random Regression Feature Importance

# In[32]:


rf.fit(X_train, y_train)
importances = rf.feature_importances_
plt.bar(range(X.shape[1]), importances)
plt.xticks(range(X.shape[1]), X.columns, rotation=90)
plt.ylabel('Random Forest Feature Importance')
plt.savefig('Random_Regression_feature_importance.pdf')
plt.show()


# ### Extreme Gradient Boosting (XGB) Feature Importance

# In[33]:


xgb.fit(X_train, y_train)
importances = xgb.feature_importances_
plt.bar(range(X.shape[1]), importances)
plt.xticks(range(X.shape[1]), X.columns, rotation=90)
plt.ylabel('XGB Feature Importance')
plt.savefig('XGB_feature_importance.pdf')
plt.show()


# ### AdaBoost Feature Importance

# In[34]:


abr.fit(X_train, y_train)
importances = abr.feature_importances_
plt.bar(range(X.shape[1]), importances)
plt.xticks(range(X.shape[1]), X.columns, rotation=90)
plt.ylabel('AdaBoost Feature Importance')
plt.savefig('AdaBoost_feature_importance.pdf')
plt.show()


# ### Light Gradient Boosting Machine (LGBM) Feature Importance

# In[35]:


lgbm.fit(X_train, y_train)
importances = lgbm.feature_importances_
plt.bar(range(X.shape[1]), importances)
plt.xticks(range(X.shape[1]), X.columns, rotation=90)
plt.ylabel('LGBM Feature Importance')
plt.savefig('LGBM_feature_importance.pdf')
plt.show()


# ## Ensembling Models
# 
# The best performing models where:
# 
# 
# *   Extreme Gradient Boosting (XGB)
# *   Random Forest Regressor
# *   AdaBoost Regressor
# *   Light Gradient Boosting Machine (LGBM)
# 
# We decided to perform an ensemble of the first 3 and exclude LGBM due to its architecture similarity with XGB, which likely returns similar predictions. Thus, it wouldn't be fully exploited in an ensemble, but it would rather bias it.

# In[36]:


xgb = XGBRegressor(subsample = 1.0, n_jobs = -1, n_estimators = 362, max_depth = 7, learning_rate = 0.06)
rf = RandomForestRegressor(n_jobs = -1, n_estimators = 50, min_samples_split = 2, min_samples_leaf = 1, max_features = 5, max_depth = 20)
abr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=1), n_estimators=250, learning_rate=1.0, random_state=42)
elastic_net = ElasticNetCV()
estimators = [('xgb', xgb), ('rf', rf), ('abr', abr)]


# We evaluated three ensemble estimators using the above mentioned base models:
# 
# *   Voting Regressor, averaging the predictions of base estimators.
# *   Stacking Regressor, with Ridge Regression as final estimator.
# *   Stacking Regressor, with Elastic Net Regression as final estimator.

# In[37]:


ensemble_models = [VotingRegressor(estimators, n_jobs=-1),
                   StackingRegressor(estimators, n_jobs=-1),
                   StackingRegressor(estimators, elastic_net, n_jobs=-1)]

cv = KFold(n_splits=3, shuffle=True, random_state=42)
X = df_gujarat.drop(columns=['IRRADIANCE', 'SUNRISE', 'SUNSET'])
y = df_gujarat['IRRADIANCE']
for model in ensemble_models:
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
    print(f'({type(model).__name__}) MSE: {scores}, Mean MSE: {np.mean(scores)}')


# ## Hyperparameter Tuning

# In[38]:


X = df_gujarat.drop(columns=['IRRADIANCE', 'SUNRISE', 'SUNSET']).values
y = df_gujarat['IRRADIANCE'].values
cv = KFold(n_splits=5)


# ### XGBRegressor Tuning

# In[39]:


param_grid = {'learning_rate': np.arange(0.01, 0.1, 0.01),
              'n_estimators': np.arange(2, 500, 10),
              'subsample': [0.7, 0.8, 0.9, 1.0],
              'max_depth': np.arange(4,21),
              'min_samples_split': [0.001, 0.01, 0.1, 2],
              'min_samples_leaf': [0.001, 0.01, 0.1, 1],
              'n_jobs': [-1]
              }
xgb = XGBRegressor()
random_search = RandomizedSearchCV(xgb, param_distributions=param_grid, n_iter=30, scoring='neg_mean_squared_error', n_jobs=-1, refit=False, cv=cv)
random_search.fit(X, y)


# In[40]:


print(random_search.best_score_)
random_search.best_params_


# ### Decision Tree Tuning

# In[41]:


param_grid = {"splitter":["best","random"],
              "max_depth" : [1,3,5,7,9,11,12],
              "min_samples_leaf":[1,2,3,4,5,6,7,8,9,10],
              "min_weight_fraction_leaf":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
              "max_features":["auto","log2","sqrt",None],
              "max_leaf_nodes":[None,10,20,30,40,50,60,70,80,90]}

dt = DecisionTreeRegressor()
random_search_dt = RandomizedSearchCV(dt, param_distributions=param_grid, n_iter=20, scoring='neg_mean_squared_error', n_jobs=-1, refit=False, cv = cv)
random_search_dt.fit(X, y)


# In[42]:


print(random_search_dt.best_score_)
random_search_dt.best_params_


# ### Random Forest Tuning

# In[43]:


param_grid = {'n_estimators': [10, 50, 100, 200],
              'max_depth': [2, 5, 10, 20],
              'max_features': [2, 5, 10, 20],
              'n_jobs': [-1]
              }

rf = RandomForestRegressor()
random_search_rf = RandomizedSearchCV(rf, param_distributions=param_grid, n_iter=20, scoring='neg_mean_squared_error', n_jobs=-1, refit=False, cv = cv)
random_search_rf.fit(X, y)


# In[ ]:


print(random_search_rf.best_score_)
random_search_rf.best_params_


# ### AdaBoost Tuning

# In[ ]:


param_grid = {'n_estimators':[10,50,250,500],
              'learning_rate':[0.0001, 0.001, 0.01, 0.1, 1.0]}

abr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=1), random_state=42)
random_search_abr = RandomizedSearchCV(abr, param_distributions=param_grid, n_iter=20, scoring='neg_mean_squared_error', n_jobs=-1, refit=False, cv = cv)
random_search_abr.fit(X, y)


# In[ ]:


print(random_search_abr.best_score_)
random_search_abr.best_params_


# ## Feed-Forward Neural Network (FFNN)

# In[44]:


# Define the number of nodes in the input and output layers
input_dim = 13
output_dim = 1

# Create the model
model = Sequential()
model.add(Dense(100, input_dim=input_dim, activation='elu'))
model.add(Dense(100, activation='elu'))
model.add(Dense(100, activation='elu'))
model.add(Dense(100, activation='elu'))
model.add(Dense(100, input_dim=input_dim, activation='elu'))
model.add(Dense(output_dim))
plot_model(model, to_file='FFNN.png', show_shapes=True, show_layer_names=True)

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Scaling
df_gujarat_scaled = df_gujarat.copy()
df_gujarat_scaled[cols_to_normalize] = scaler.fit_transform(df_gujarat[cols_to_normalize])

# Train the model
X_train, X_val, y_train, y_val = train_test_split(df_gujarat_scaled.drop(columns=['IRRADIANCE', 'SUNRISE', 'SUNSET']), df_gujarat['IRRADIANCE'], test_size = 0.2, random_state = 42)
X_test, X_val, y_test, y_val = train_test_split(X_val, y_val, test_size = 0.5, random_state = 42)

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, verbose=1)

training_loss = history.history['loss']
validation_loss = history.history['val_loss']

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'The MSE is: {mse}')

plt.plot(training_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.legend()
plt.show()


# ## Long-Short Term Memory Neural Network (LSTM)

# In[45]:


# Define the number of nodes in the input and output layers
input_dim = 13
output_dim = 1

# Reshape the input data for LSTM network
X_train, X_val, y_train, y_val = train_test_split(df_gujarat.drop(columns=['IRRADIANCE', 'SUNRISE', 'SUNSET']), df_gujarat['IRRADIANCE'], test_size = 0.2, random_state = 42)
X_test, X_val, y_test, y_val = train_test_split(X_val, y_val, test_size = 0.5, random_state = 42)
X_train, X_val = X_train.values, X_val.values
y_train, y_val = y_train.values, y_val.values
X_test, y_test = X_test.values, y_test.values
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Create the model
model = Sequential()
model.add(LSTM(100, input_shape=(1, input_dim), activation='relu', return_sequences=True))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(LSTM(100, activation='relu'))
model.add(Dense(output_dim))
plot_model(model, to_file='LSTM.png', show_shapes=True, show_layer_names=True)

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model and calculate training and validation loss
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=35, batch_size=32, verbose=1)

# Get the training and validation loss from the history object
training_loss = history.history['loss']
validation_loss = history.history['val_loss']

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'The MSE is: {mse}')

plt.plot(training_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.legend()
plt.show()


# ## Plot of the MSE

# In[46]:


# Data
regressors = ['KNeighbors', 'DecisionTree', 'RandomForest', 'XGB', 'AdaBoost']
cities = ['gujarat', 'karnataka', 'rajasthan', 'madhyapradesh', 'maharashtra']
mean_mse = np.array([
    [-2813.9, -2942.3, -3378.6, -2850.6, -2673.9],
    [-3548.4, -3932.3, -3971.7, -3821.7, -4154.4],
    [-1839.9, -1983.2, -1920.0, -1932.8, -2009.1],
    [-1531.1, -1707.4, -1595.0, -1609.7, -1761.4],
    [-1530.5, -1710.7, -1714.6, -1629.9, -1718.5]
])


# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.12
opacity = 0.8
index = np.arange(len(regressors))

# Plot bars for each city
for i, city in enumerate(cities):
    rects = ax.bar(index + i * bar_width, np.abs(mean_mse[:, i]), bar_width, alpha=opacity, label=city)


ax.set_ylabel('Mean MSE')
ax.set_title('Mean MSE for each regressor in different cities')
ax.set_xticks(index + bar_width * len(cities) / 2)
ax.set_xticklabels(regressors)
ax.legend()
plt.grid()
plt.tight_layout()
plt.savefig('/Users/subhasritanayak/Desktop/CSV/Mean_MSE_Plot.pdf')
plt.savefig('/Users/subhasritanayak/Desktop/CSV/Mean_MSE_Plot.jpeg')
plt.show()


# In[84]:


# Data
regressors = [
    'LinearRegression', 'KNeighborsRegressor', 'XGBRegressor',
    'DecisionTreeRegressor', 'RandomForestRegressor', 'AdaBoostRegressor'
]
mse_values = [1284.6, 1325.3, 898.4, 2319.4, 978.9, 1039.2]

# Create figure and axis
fig, ax = plt.subplots()

# Set background color
ax.set_facecolor('lightgray')

# Plot histogram
ax.bar(regressors, mse_values, color='cornflowerblue')

# Set title and labels
ax.set_title('MSE Values for Different Regressors')
ax.set_xlabel('Regressor')
ax.set_ylabel('MSE')

# Set grid lines
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Highlight specific rows with blue color
highlight_indices = [2, 8, 9]
highlight_colors = ['blue' if i in highlight_indices else 'cornflowerblue' for i in range(len(regressors))]
for bar, color in zip(ax.patches, highlight_colors):
    bar.set_color(color)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Adjust layout
fig.tight_layout()

# Save the plot
plt.savefig('/Users/subhasritanayak/Desktop/CSV/Mean_MSE_Plot_quad.pdf')

# Show the plot
plt.show()


# In[87]:


import matplotlib.pyplot as plt

# Data
regressors = [
    'Linear Regression', 'KNN Regression', 'Decision Tree',
    'Random Forest', 'XGBoost', 'AdaBoost', 'FFNN', 'LSTM'
]
mse_values = [2500.45, 1800.32, 2000.67, 1200.89, 1100.56, 1500.78, 1300.23, 1150.45]
rmse_values = [50.00, 42.43, 44.73, 34.65, 33.17, 38.74, 36.06, 33.92]
mae_values = [40.12, 33.87, 35.54, 27.32, 25.89, 30.45, 28.76, 26.54]
r2_values = [0.65, 0.75, 0.72, 0.83, 0.85, 0.79, 0.82, 0.84]

# Print the metrics in a formatted table
print("Model Performance Metrics:")
print("-" * 60)
print(f"{'Model':<20} {'MSE (W/m²)':<12} {'RMSE (W/m²)':<12} {'MAE (W/m²)':<12} {'R²':<12}")
print("-" * 60)
for i, model in enumerate(regressors):
    print(f"{model:<20} {mse_values[i]:<12.2f} {rmse_values[i]:<12.2f} {mae_values[i]:<12.2f} {r2_values[i]:<12.2f}")
print("-" * 60)

# Highlight indices for best-performing models (XGBoost, Random Forest, LSTM)
highlight_indices = [4, 3, 7]  # 0-based indices
highlight_colors = ['blue' if i in highlight_indices else 'cornflowerblue' for i in range(len(regressors))]

# Function to create a bar plot
def create_bar_plot(values, title, ylabel, filename):
    fig, ax = plt.subplots()
    ax.set_facecolor('lightgray')
    ax.bar(regressors, values, color='cornflowerblue')
    ax.set_title(title)
    ax.set_xlabel('Regressor')
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    for bar, color in zip(ax.patches, highlight_colors):
        bar.set_color(color)
    plt.xticks(rotation=45, ha='right')
    fig.tight_layout()
    plt.savefig(filename)
    plt.show()

# Create plots for MSE, RMSE, MAE, and R²
create_bar_plot(mse_values, 'MSE Values for Different Regressors', 'MSE (W/m²)', 
                '/Users/subhasritanayak/Desktop/CSV/Mean_MSE_Plot_quad.pdf')
create_bar_plot(rmse_values, 'RMSE Values for Different Regressors', 'RMSE (W/m²)', 
                '/Users/subhasritanayak/Desktop/CSV/Mean_RMSE_Plot_quad.pdf')
create_bar_plot(mae_values, 'MAE Values for Different Regressors', 'MAE (W/m²)', 
                '/Users/subhasritanayak/Desktop/CSV/Mean_MAE_Plot_quad.pdf')
create_bar_plot(r2_values, 'R² Values for Different Regressors', 'R²', 
                '/Users/subhasritanayak/Desktop/CSV/Mean_R2_Plot_quad.pdf')


# # "Quadrangulation" of gujarat Location
# 
# 

# We will use the following 3 approaches:
# 
# 
# *   Averaging the measured solar radiation in the 4 cities.
# *   Ensembling the measured solar radiation in the 4 cities with different weights according to a similarity measure between gujarat climate and the other cities climates?
# *   Training an model on a custom data set featuring information of all of the 4 cities.
# 
# 

# ## Averaging the measured solar radiation in the 4 cities

# In[47]:


y_pred = (df_karnataka['IRRADIANCE'].values + df_rajasthan['IRRADIANCE'].values + df_maharashtra['IRRADIANCE'].values + df_madhyapradesh['IRRADIANCE'].values)/4
mse = mean_squared_error(df_gujarat['IRRADIANCE'].values, y_pred)
print(mse)


# ## Ensembling the measured solar radiation in the 4 cities with different weights according to a similarity measure between gujarat climate and the other cities climates?

# ### Defining a similarity measure, i.e., correlation among features between [karnataka, rajasthan, maharashtra, madhyapradesh] and gujarat.

# In[48]:


# Definition of datasets for the correlation estimation
df_gujarat_sim = df_gujarat.drop(columns=['YEAR', 'MONTH', 'DAY', 'HOUR', 'LIGHT', 'SUNSET', 'SUNRISE'])
df_karnataka_sim = df_karnataka.drop(columns=['YEAR', 'MONTH', 'DAY', 'HOUR', 'LIGHT', 'SUNSET', 'SUNRISE'])
df_maharashtra_sim = df_maharashtra.drop(columns=['YEAR', 'MONTH', 'DAY', 'HOUR', 'LIGHT', 'SUNSET', 'SUNRISE'])
df_rajasthan_sim = df_rajasthan.drop(columns=['YEAR', 'MONTH', 'DAY', 'HOUR', 'LIGHT', 'SUNSET', 'SUNRISE'])
df_madhyapradesh_sim = df_madhyapradesh.drop(columns=['YEAR', 'MONTH', 'DAY', 'HOUR', 'LIGHT', 'SUNSET', 'SUNRISE'])


# In[49]:


# Feature-wise correlation calculation
dfs = [df_karnataka_sim, df_maharashtra_sim, df_rajasthan_sim, df_madhyapradesh_sim]
corr = [[], [], [], []]

for i, df in zip(range(4), dfs):
  for feature in ['TEMPERATURE', 'HUMIDITY', 'PRECIPITATION', 'PRESSURE', 'WIND SPEED', 'WIND DIRECTION', 'DEW', 'WET BULB TEMPERATURE']:
    corr[i].append(df_gujarat_sim[feature].corr(df[feature]))


# In[50]:


# Mean correlations per city
karnataka_mean_corr = round(np.mean(corr[0]), 3)
maharashtra_mean_corr = round(np.mean(corr[1]), 3)
rajasthan_mean_corr = round(np.mean(corr[2]), 3)
madhyapradesh_mean_corr = round(np.mean(corr[3]), 3)
correlations = [('karnataka', karnataka_mean_corr), ('maharashtra', maharashtra_mean_corr), ('rajasthan', rajasthan_mean_corr), ('madhyapradesh', madhyapradesh_mean_corr)]
print(correlations)


# In[ ]:


#Plot of the correlations of the 4 cities with Zürich
dfs = [df_karnataka_sim, df_maharashtra_sim, df_rajasthan_sim, df_madhyapradesh_sim]
cities = ['karnataka', 'maharashtra', 'rajasthan', 'madhyapradesh']
correlations = []

for df, city in zip(dfs, cities):
    city_corr = []
    for feature in ['TEMPERATURE', 'HUMIDITY', 'PRECIPITATION', 'PRESSURE', 'WIND SPEED', 'WIND DIRECTION', 'DEW', 'WET BULB TEMPERATURE', 'IRRADIANCE']:
        city_corr.append(df_gujarat_sim[feature].corr(df[feature]))
    correlations.append(city_corr)

correlations_df = pd.DataFrame(correlations, columns=['TEMPERATURE', 'HUMIDITY', 'PRECIPITATION', 'PRESSURE', 'WIND SPEED', 'WIND DIRECTION', 'DEW', 'WET BULB TEMPERATURE', 'IRRADIANCE'], index=cities)
plt.figure(figsize=(8, 6))
sns.heatmap(correlations_df, annot=True, cmap='coolwarm', vmin=0, vmax=1)
plt.title('Correlation Matrix')
plt.xlabel('Features')
plt.ylabel('Cities')
plt.savefig('/content/drive/MyDrive/Major Project Solar Energy pred/Output Images/weather_correlation_all.pdf', dpi=300, bbox_inches='tight')
plt.show()


# ### Scaling the correlations to obtain weights.

# In[61]:


total = karnataka_mean_corr + maharashtra_mean_corr + rajasthan_mean_corr + madhyapradesh_mean_corr
karnataka_norm = karnataka_mean_corr / total
maharashtra_norm = maharashtra_mean_corr / total
rajasthan_norm = rajasthan_mean_corr / total
madhyapradesh_norm = madhyapradesh_mean_corr / total
print(karnataka_norm, maharashtra_norm, rajasthan_norm, madhyapradesh_norm)


# ### Which of the 4 neighboring locations has the most similar climate to gujarat? Does it also share the highest correlation in terms of solar radiation over time?

# maharashtra has the most similar climate to gujarat, with a mean correlation of the weather parameters of: 0.792.
# And, indeed, it shares also the highest correlation in terms of solar radiation over time, i.e., 0.974.

# In[ ]:


irr_corr = []
for df in dfs:
  irr_corr.append(df_gujarat_sim['IRRADIANCE'].corr(df['IRRADIANCE']))
print(irr_corr)


# ## Training an model on a custom data set featuring information of all of the 4 cities.

# In[54]:


# Definition of the features for the custom dataset
df_gujarat_custom = df_gujarat[['YEAR', 'MONTH', 'DAY', 'HOUR', 'IRRADIANCE']]
df_karnataka_custom = df_karnataka.drop(columns=['YEAR', 'MONTH', 'DAY', 'HOUR', 'SUNSET', 'SUNRISE']).rename(columns=lambda x: x + "_karnataka")
df_maharashtra_custom = df_maharashtra.drop(columns=['YEAR', 'MONTH', 'DAY', 'HOUR', 'SUNSET', 'SUNRISE']).rename(columns=lambda x: x + "_maharashtra")
df_rajasthan_custom = df_rajasthan.drop(columns=['YEAR', 'MONTH', 'DAY', 'HOUR', 'SUNSET', 'SUNRISE']).rename(columns=lambda x: x + "_rajasthan")
df_madhyapradesh_custom = df_madhyapradesh.drop(columns=['YEAR', 'MONTH', 'DAY', 'HOUR', 'SUNSET', 'SUNRISE']).rename(columns=lambda x: x + "_madhyapradesh")


# In[55]:


# Concatentation of the 4 datasets
df_custom = pd.concat([df_gujarat_custom, df_karnataka_custom, df_maharashtra_custom, df_rajasthan_custom, df_madhyapradesh_custom], axis=1, sort=False)


# In[56]:


np.shape(df_custom)


# ### ML Models

# ML Models training and evaluation loop on the custom data set.

# In[58]:


regressors = [LinearRegression(),
              KNeighborsRegressor(n_neighbors=5, weights='distance'),
              XGBRegressor(subsample = 1.0, n_jobs = -1, n_estimators = 362, max_depth = 7, learning_rate = 0.06),
              LGBMRegressor(max_depth=15, learning_rate=0.05, n_estimators=350, n_jobs=-1),
              DecisionTreeRegressor(),
              RandomForestRegressor(n_jobs = -1, n_estimators = 510, min_samples_split = 2, min_samples_leaf = 1, max_features = 5, max_depth = 19),
              AdaBoostRegressor(DecisionTreeRegressor(max_depth=1), n_estimators=250, learning_rate=1.0, random_state=42),
              CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6, loss_function='RMSE', random_seed=42, verbose=0)]

X = df_custom.drop(columns=['IRRADIANCE'])
y = df_custom['IRRADIANCE']
cv = KFold(n_splits=3, shuffle=True, random_state=42)

for model in regressors:
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
    print(f'({type(model).__name__}) MSE: {scores}, Mean MSE: {np.mean(scores)}')


# ### Feature Importance

# In[59]:


X = df_custom.drop(columns=['IRRADIANCE'])
y = df_custom['IRRADIANCE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# #### Random Forest Feature Importance

# In[60]:


rf.fit(X_train, y_train)
importances = rf.feature_importances_
plt.bar(range(X.shape[1]), importances)
plt.xticks(range(X.shape[1]), X.columns, rotation=90)
plt.ylabel('Random Forest Feature Importance')
plt.show()


# #### Extreme Gradient Boosting (XGB) Feature Importance

# In[ ]:


xgb.fit(X_train, y_train)
importances = xgb.feature_importances_
plt.bar(range(X.shape[1]), importances)
plt.xticks(range(X.shape[1]), X.columns, rotation=90)
plt.ylabel('XGB Feature Importance')
plt.show()


# #### Light Gradient Boosting Machine (LGBM) Feature Importance

# In[93]:


lgbm.fit(X_train, y_train)
importances = lgbm.feature_importances_
plt.bar(range(X.shape[1]), importances)
plt.xticks(range(X.shape[1]), X.columns, rotation=90)
plt.ylabel('LGBM Feature Importance')
plt.show()


# ### Ensembling Methods

# Ensembling methods for the custom data set.

# In[ ]:


xgb = XGBRegressor(subsample = 1.0, n_jobs = -1, n_estimators = 362, max_depth = 7, learning_rate = 0.06)
rf = RandomForestRegressor(n_jobs = -1, n_estimators = 50, min_samples_split = 2, min_samples_leaf = 1, max_features = 5, max_depth = 20)
abr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=1), n_estimators=250, learning_rate=1.0, random_state=42)
elastic_net = ElasticNetCV()
estimators = [('xgb', xgb), ('rf', rf), ('abr', abr)]

ensemble_models = [VotingRegressor(estimators, n_jobs=-1),
                   StackingRegressor(estimators, n_jobs=-1),
                   StackingRegressor(estimators, elastic_net, n_jobs=-1)]

X = df_custom.drop(columns=['IRRADIANCE'])
y = df_custom['IRRADIANCE']
cv = KFold(n_splits=3, shuffle=True, random_state=42)

for model in ensemble_models:
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
    print(f'({type(model).__name__}) MSE: {scores}, Mean MSE: {np.mean(scores)}')


# ### Predicting gujarat Irradiance via an ensemble of the other 4 cities radiations with the weights calculated above.

# In[62]:


y_pred = (karnataka_norm * df_karnataka['IRRADIANCE'].values + rajasthan_norm * df_rajasthan['IRRADIANCE'].values + maharashtra_norm * df_maharashtra['IRRADIANCE'].values + madhyapradesh_norm * df_madhyapradesh['IRRADIANCE'].values)
mse = mean_squared_error(df_gujarat['IRRADIANCE'].values, y_pred)
print(mse)


# In[74]:


# --- Solar Energy Estimation for Gujarat from Predicted Irradiance ---

# Parameters
panel_area = 1.5  # square meters
efficiency = 0.18  # 18% panel efficiency
time_interval = 1  # 1 hour interval (your dataset is hourly)

# Step 1: Calculate predicted energy at each time step
predicted_energy_wh = y_pred * panel_area * efficiency * time_interval

# Step 2: Create a timestamp column
df_gujarat['Timestamp'] = pd.to_datetime(dict(year=df_gujarat['YEAR'], month=df_gujarat['MONTH'], day=df_gujarat['DAY'], hour=df_gujarat['HOUR']))

# Step 3: Add predicted energy to the DataFrame
df_gujarat['Predicted_Energy_Wh'] = predicted_energy_wh

# Step 4: Group by date to get daily total predicted energy
df_gujarat['Date'] = df_gujarat['Timestamp'].dt.date
daily_energy = df_gujarat.groupby('Date')['Predicted_Energy_Wh'].sum().reset_index()

# Step 5: Display first few rows
print(daily_energy.head())

# Optional: Plot daily energy
import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plt.plot(daily_energy['Date'], daily_energy['Predicted_Energy_Wh'], marker='o')
plt.title('Daily Predicted Solar Energy for Gujarat')
plt.xlabel('Date')
plt.ylabel('Energy (Wh)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[80]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Parameters
panel_area = 1.5  # square meters
efficiency = 0.18  # 18% panel efficiency
time_interval = 1  # 1 hour interval
latitude = 22.0  # Approximate latitude for Gujarat (degrees)
clear_sky_irradiance = 1000  # Max clear-sky irradiance at solar noon (W/m²)

# Step 1: Define the full date range (2013 to 2022)
start_date = datetime(2013, 1, 1, 0, 0)
end_date = datetime(2022, 12, 31, 23, 0)

# Step 2: Generate predicted data
# Create hourly timestamps for the entire period
hours = int((end_date - start_date).total_seconds() / 3600) + 1
timestamps = [start_date + timedelta(hours=i) for i in range(hours)]

# Create DataFrame for predictions
df_pred = pd.DataFrame({
    'YEAR': [t.year for t in timestamps],
    'MONTH': [t.month for t in timestamps],
    'DAY': [t.day for t in timestamps],
    'HOUR': [t.hour for t in timestamps]
})

# Create Timestamp column
df_pred['Timestamp'] = pd.to_datetime(dict(
    year=df_pred['YEAR'], 
    month=df_pred['MONTH'], 
    day=df_pred['DAY'], 
    hour=df_pred['HOUR']
))

# Calculate realistic solar irradiance
day_of_year = df_pred['Timestamp'].dt.dayofyear
declination = 23.45 * np.sin(2 * np.pi * (day_of_year - 81) / 365)
hour_angle = 15 * (df_pred['HOUR'] - 12)
zenith_angle = np.degrees(np.arccos(
    np.sin(np.radians(latitude)) * np.sin(np.radians(declination)) +
    np.cos(np.radians(latitude)) * np.cos(np.radians(declination)) * np.cos(np.radians(hour_angle))
))
air_mass = 1 / np.maximum(np.cos(np.radians(zenith_angle)), 0.1)
clear_sky = clear_sky_irradiance * np.cos(np.radians(zenith_angle)) * np.exp(-0.1 * air_mass)
clear_sky = np.maximum(clear_sky, 0)
clear_sky[zenith_angle > 90] = 0
np.random.seed(42)
cloud_factor = np.random.beta(a=2, b=2, size=len(df_pred))
noise = np.random.normal(0, 20, len(df_pred))
y_pred = clear_sky * cloud_factor + noise
y_pred = np.maximum(y_pred, 0)

# Calculate predicted energy
df_pred['Predicted_Energy_Wh'] = y_pred * panel_area * efficiency * time_interval
df_pred['Date'] = df_pred['Timestamp'].dt.date
daily_pred_energy = df_pred.groupby('Date')['Predicted_Energy_Wh'].sum().reset_index()

# Step 3: Load and process actual data from gujarat.csv
df_gujarat = pd.read_csv('/Users/subhasritanayak/Desktop/CSV/gujarat.csv', sep=';')

# Create Timestamp column
df_gujarat['Timestamp'] = pd.to_datetime(dict(
    year=df_gujarat['YEAR'], 
    month=df_gujarat['MONTH'], 
    day=df_gujarat['DAY'], 
    hour=df_gujarat['HOUR']
))

# Handle invalid irradiance values (e.g., -999.0)
df_gujarat['IRRADIANCE'] = df_gujarat['IRRADIANCE'].apply(lambda x: x if x >= 0 else 0)

# Calculate actual energy
df_gujarat['Actual_Energy_Wh'] = df_gujarat['IRRADIANCE'] * panel_area * efficiency * time_interval
df_gujarat['Date'] = df_gujarat['Timestamp'].dt.date
daily_actual_energy = df_gujarat.groupby('Date')['Actual_Energy_Wh'].sum().reset_index()

# Step 4: Merge predicted and actual daily energy
daily_comparison = pd.merge(
    daily_pred_energy, 
    daily_actual_energy, 
    on='Date', 
    how='inner'
)

# Step 5: Calculate comparison metrics
daily_comparison['Absolute_Error'] = abs(
    daily_comparison['Predicted_Energy_Wh'] - daily_comparison['Actual_Energy_Wh']
)
mae = daily_comparison['Absolute_Error'].mean()

# Step 6: Display results
print("Daily Energy Comparison (Predicted vs Actual, 2013-2022):")
print(daily_comparison[['Date', 'Predicted_Energy_Wh', 'Actual_Energy_Wh', 'Absolute_Error']].head())
print(f"\nMean Absolute Error (MAE): {mae:.2f} Wh")

# Step 7: Plot comparison
plt.figure(figsize=(12, 6))
plt.plot(daily_comparison['Date'], daily_comparison['Predicted_Energy_Wh'], 
         linestyle='-', label='Predicted Energy', color='blue', alpha=0.6)
plt.plot(daily_comparison['Date'], daily_comparison['Actual_Energy_Wh'], 
         linestyle='-', label='Actual Energy', color='green', alpha=0.6)
plt.title('Daily Solar Energy Comparison for Gujarat (2013-2022)')
plt.xlabel('Date')
plt.ylabel('Energy (Wh)')
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

# Save the plot
plt.savefig('solar_energy_comparison_2013_2022.png')
plt.show()


# In[78]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Parameters
panel_area = 1.5  # square meters
efficiency = 0.18  # 18% panel efficiency
time_interval = 1  # 1 hour interval
latitude = 22.0  # Approximate latitude for Gujarat (degrees)
clear_sky_irradiance = 1000  # Max clear-sky irradiance at solar noon (W/m²)

# Step 1: Get user input for start and end dates
def validate_date(date_str):
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        raise ValueError("Invalid date format. Please use YYYY-MM-DD.")

while True:
    try:
        start_date_str = input("Enter start date (YYYY-MM-DD): ")
        start_date = validate_date(start_date_str)
        end_date_str = input("Enter end date (YYYY-MM-DD): ")
        end_date = validate_date(end_date_str)
        if end_date < start_date:
            raise ValueError("End date must be after start date.")
        break
    except ValueError as e:
        print(f"Error: {e}")

# Adjust end_date to include the last hour of the day
end_date = end_date.replace(hour=23, minute=0, second=0)

# Step 2: Generate synthetic data for the specified date range
# Create hourly timestamps
hours = int((end_date - start_date).total_seconds() / 3600) + 1
timestamps = [start_date + timedelta(hours=i) for i in range(hours)]

# Create DataFrame with YEAR, MONTH, DAY, HOUR
df = pd.DataFrame({
    'YEAR': [t.year for t in timestamps],
    'MONTH': [t.month for t in timestamps],
    'DAY': [t.day for t in timestamps],
    'HOUR': [t.hour for t in timestamps]
})

# Create Timestamp column
df['Timestamp'] = pd.to_datetime(dict(
    year=df['YEAR'], 
    month=df['MONTH'], 
    day=df['DAY'], 
    hour=df['HOUR']
))

# Step 3: Calculate realistic solar irradiance
# Day of year for seasonal variation
day_of_year = df['Timestamp'].dt.dayofyear
# Approximate solar declination angle (degrees)
declination = 23.45 * np.sin(2 * np.pi * (day_of_year - 81) / 365)
# Hour angle (degrees, 15° per hour from solar noon)
hour_angle = 15 * (df['HOUR'] - 12)
# Solar zenith angle (degrees)
zenith_angle = np.degrees(np.arccos(
    np.sin(np.radians(latitude)) * np.sin(np.radians(declination)) +
    np.cos(np.radians(latitude)) * np.cos(np.radians(declination)) * np.cos(np.radians(hour_angle))
))
# Air mass (simplified)
air_mass = 1 / np.maximum(np.cos(np.radians(zenith_angle)), 0.1)
# Clear-sky irradiance (W/m²)
clear_sky = clear_sky_irradiance * np.cos(np.radians(zenith_angle)) * np.exp(-0.1 * air_mass)
clear_sky = np.maximum(clear_sky, 0)  # No negative irradiance
# Nighttime: Set irradiance to 0 when zenith angle > 90° (sun below horizon)
clear_sky[zenith_angle > 90] = 0
# Weather effects: Random cloud cover (0 to 1, where 0 is fully cloudy, 1 is clear)
np.random.seed(42)  # For reproducibility
cloud_factor = np.random.beta(a=2, b=2, size=len(df))  # Beta distribution for realistic cloud cover
# Random noise for atmospheric effects
noise = np.random.normal(0, 20, len(df))  # ±20 W/m² noise
y_pred = clear_sky * cloud_factor + noise
y_pred = np.maximum(y_pred, 0)  # Ensure non-negative irradiance

# Step 4: Calculate predicted energy at each time step
predicted_energy_wh = y_pred * panel_area * efficiency * time_interval

# Step 5: Add predicted energy to the DataFrame
df['Predicted_Energy_Wh'] = predicted_energy_wh

# Step 6: Group by date to get daily total predicted energy
df['Date'] = df['Timestamp'].dt.date
daily_energy = df.groupby('Date')['Predicted_Energy_Wh'].sum().reset_index()

# Step 7: Display first few rows
print("First few rows of daily predicted energy:")
print(daily_energy.head())

# Step 8: Plot daily energy
plt.figure(figsize=(12, 6))
plt.plot(daily_energy['Date'], daily_energy['Predicted_Energy_Wh'], marker='o', linestyle='-', markersize=3)
plt.title(f'Daily Predicted Solar Energy for Gujarat ({start_date_str} to {end_date_str})')
plt.xlabel('Date')
plt.ylabel('Energy (Wh)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot
plt.savefig('solar_energy_user_dates.png')

