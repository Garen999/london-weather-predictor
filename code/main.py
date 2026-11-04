#imports modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

#groups data by month
weather["month"] = weather["date"].dt.month
mean_monthly_weather = weather.groupby("month").mean()

#adds next day mean temp column
weather["next_day_mean_temp"] = weather["mean_temp"].shift(-1)
weather["next_day_mean_temp"].loc[weather["date"].diff().dt.days != 1, "next_day_mean_temp"].drop

#creates graph labels
def graph_labels(axis, title, x_label, y_label):
    axes[axis].set_title(title)
    axes[axis].set_xlabel(x_label)
    axes[axis].set_ylabel(y_label)

#sets up graph grid
figures, axes = plt.subplots(1,3, figsize=(14,5))

#graphs monthly mean temperature
axes[0].plot(mean_monthly_weather.index,mean_monthly_weather["mean_temp"])
graph_labels(0, "Monthly Mean Temperature", "Mean Temperature (°C)", "Month")

#correlates data with mean temp
mean_temp_correlation = weather.corr(numeric_only=True)["mean_temp"].drop("mean_temp")
mean_temp_correlation = mean_temp_correlation.reindex(mean_temp_correlation.abs().sort_values(ascending=False).index)

#graphs mean temperature correlations
sns.heatmap(mean_temp_correlation.to_frame(), cmap="coolwarm", annot=True, center=0, ax=axes[1])
graph_labels(1, "Temperature Correlation", "Correlation with Mean Temperature", "Weather Variable")

#graphs mean tempareture distribution
mean_temp_distribution = weather["mean_temp"].hist(bins=30, ax=axes[2])
graph_labels(2, "Temperature Distribution", "Mean Temperature (°C)", "Frequency (Days)")

#calculates mean temperature distribution statisitics
def mean_temp_stats():
    print(f"mean is {weather["mean_temp"].mean()}")
    print(f"median is {weather["mean_temp"].median()}")
    print(f"std is {weather["mean_temp"].std()}")
    print(f"LQ is {weather["mean_temp"].quantile(0.25)}")
    print(f"UQ is {weather["mean_temp"].quantile(0.75)}")

mean_temp_stats()

#shows graphs
plt.tight_layout()
plt.show()


