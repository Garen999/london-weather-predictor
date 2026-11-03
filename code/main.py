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

#graphs monthly mean temperature
figures, axes = plt.subplots(1,2, figsize=(14,5))

axes[0].plot(mean_monthly_weather.index,mean_monthly_weather["mean_temp"])
axes[0].set_title("Monthly Mean Temperature")
axes[0].set_xlabel("Mean Temperature (°C)")
axes[0].set_ylabel("Month")

#correlates data with mean temp
mean_temp_correlation = weather.corr(numeric_only=True)["mean_temp"].drop("mean_temp")
mean_temp_correlation = mean_temp_correlation.reindex(mean_temp_correlation.abs().sort_values(ascending=False).index)

#graphs mean temperature correlations
sns.heatmap(mean_temp_correlation.to_frame(), cmap="coolwarm", annot=True, center=0, ax=axes[1])
axes[1].set_title("Temperature Correlations")
axes[1].set_xlabel("Correlation with Mean Temperature")
axes[1].set_ylabel("Weather Variable")

#shows graphs
plt.tight_layout()
plt.show()


