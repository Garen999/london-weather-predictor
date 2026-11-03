#imports modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#creates data frame
weather = pd.read_csv("data/london_weather.csv")
weather["date"] = pd.to_datetime(weather["date"],format="%Y%m%d")

#cleans data frame
weather = weather.drop("snow_depth", axis="columns")
weather = weather.dropna(
    subset=[
    "cloud_cover",
    "global_radiation",
    "max_temp",
    "mean_temp",
    "min_temp",
    "precipitation",
    "pressure"
    ])
weather = weather.sort_values(by="date")

#correlates data with mean temp
mean_temp_correlation = weather.drop(columns=[
    "cloud_cover",
    "global_radiation",
    "max_temp",
    "min_temp",
    "precipitation",
    "pressure"]).corrwith(weather["mean_temp"])

max(mean_temp_correlation.values)

#groups data by month
weather["month"] = weather["date"].dt.month
mean_monthly_weather = weather.groupby("month").mean()

#graphs data frame
plt.plot(mean_monthly_weather.index,mean_monthly_weather["mean_temp"])
plt.xlabel("Mean Temperature (°C)")
plt.ylabel("Month")
plt.show()


