#imports modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#creates data frame
weather = pd.read_csv("data/london_weather.csv")
weather["date"] = pd.to_datetime(weather["date"],format="%Y%m%d")
weather = weather.sort_values(by="date")

#creates lag features
def create_lag_feature(label, number):
    weather[label] = weather["mean_temp"].shift(number)
    weather.loc[weather["date"].diff(number).dt.days != number, label] = np.nan

create_lag_feature("target_mean_temp", -1)
create_lag_feature("lag1_mean_temp", 1)
create_lag_feature("lag2_mean_temp", 2)
create_lag_feature("lag3_mean_temp", 3)

#cleans data frame
weather = weather.drop("snow_depth", axis="columns")
weather = weather.dropna(
    subset=[
    "cloud_cover",
    "global_radiation",
    "max_temp",
    "mean_temp",
    "target_mean_temp",
    "lag1_mean_temp",
    "lag2_mean_temp",
    "lag3_mean_temp",
    "min_temp",
    "precipitation",
    "pressure"
    ])

#adds date columns to data frame
weather["day_of_year"] = weather["date"].dt.dayofyear

weather.loc[weather["date"].dt.is_leap_year, "year_length"] = 366
weather.loc[~weather["date"].dt.is_leap_year, "year_length"] = 365

weather["day_sin"] = np.sin(weather["day_of_year"] * 2 * np.pi / weather["year_length"])
weather["day_cos"] = np.cos(weather["day_of_year"] * 2 * np.pi / weather["year_length"])

weather["month"] = weather["date"].dt.month
mean_monthly_weather = weather.groupby("month").mean()

#creates graph labels
def graph_labels(axis, title, x_label, y_label):
    axes[axis].set_title(title)
    axes[axis].set_xlabel(x_label)
    axes[axis].set_ylabel(y_label)

#sets up graph grid
figures, axes = plt.subplots(1,3, figsize=(14,5))

#graphs monthly mean temperature
axes[0].plot(mean_monthly_weather.index,mean_monthly_weather["mean_temp"])
graph_labels(0, "Monthly Mean Temperature", "Month", "Mean Temperature (°C)")

#correlates data with mean temp
mean_temp_correlation = weather.corr(numeric_only=True)["mean_temp"].drop([
    "mean_temp",
    "target_mean_temp",
    "lag1_mean_temp",
    "lag2_mean_temp",
    "lag3_mean_temp",
    "month",
    "day_of_year",
    "year_length",
    "day_sin",
    "day_cos"
    ])

mean_temp_correlation = mean_temp_correlation.reindex(mean_temp_correlation.abs().sort_values(ascending=False).index)

#graphs mean temperature correlations
sns.heatmap(mean_temp_correlation.to_frame(), cmap="coolwarm", annot=True, center=0, ax=axes[1])
graph_labels(1, "Temperature Correlation", "Correlation with Mean Temperature", "Weather Variable")

#graphs mean tempareture distribution
mean_temp_distribution = weather["mean_temp"].hist(bins=30, ax=axes[2])
graph_labels(2, "Temperature Distribution", "Mean Temperature (°C)", "Frequency (Days)")

#calculates mean temperature distribution statisitics
def mean_temp_stats(df):
    print(f"mean is {df["mean_temp"].mean()}")
    print(f"median is {df["mean_temp"].median()}")
    print(f"std is {df["mean_temp"].std()}")
    print(f"LQ is {df["mean_temp"].quantile(0.25)}")
    print(f"UQ is {df["mean_temp"].quantile(0.75)}")

mean_temp_stats(weather)

#shows graphs
plt.tight_layout()
plt.show()


