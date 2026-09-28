#importing python modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#creating data frame
weather = pd.read_csv('data\london_weather.csv')
weather['date'] = pd.to_datetime(weather['date'],format='%Y%m%d')

#cleaning data frame
weather = weather.drop("snow_depth", axis="columns")
weather = weather.dropna(subset=["cloud_cover","global_radiation","max_temp","mean_temp","min_temp","precipitation","pressure"])
missing_percentage = weather.isnull().sum() / len(weather) * 100

#graphing data frame
plt.plot(weather['date'],weather['mean_temp'])
plt.show()

#printing data frame information
# print(weather.head(1))
# print(weather.tail(1))
# print(weather.date.is_monotonic_increasing)
# print(weather.columns)
# print(weather.info())
# print(weather.shape)
# print(weather.isnull().sum())
# print(weather.dtypes)
# print(weather.describe())
# print(missing_percentage)


