#imports modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

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
def graph_labels(row, column, title, x_label, y_label):
    axes[row][column].set_title(title)
    axes[row][column].set_xlabel(x_label)
    axes[row][column].set_ylabel(y_label)

#creates input and target features data frame
input_features = weather.drop(["target_mean_temp","date","day_of_year", "month", "year_length"], axis="columns")
training_input_features = input_features.iloc[0: int(0.8*input_features.shape[0])]
testing_input_features = input_features.iloc[int(0.8*input_features.shape[0]): input_features.shape[0]]

target_feature = weather["target_mean_temp"]
training_target_feature = target_feature.iloc[0: int(0.8*input_features.shape[0])]
testing_target_feature = target_feature.iloc[int(0.8*input_features.shape[0]): input_features.shape[0]]

#machine learning model predictions
def model_evaluation(x, y):
    print(f"MAE: {mean_absolute_error(x, y)}")
    print(f"RMS: {root_mean_squared_error(x, y)}")
    print(f"r2: {r2_score(x, y)}")
    print("\n")

linear_regression_model = LinearRegression()
linear_regression_model.fit(training_input_features, training_target_feature)
linear_regression_prediction = linear_regression_model.predict(testing_input_features)

temperature_residuals = testing_target_feature - linear_regression_prediction
monthly_temperature_residuals = temperature_residuals.abs().groupby(weather["month"].loc[temperature_residuals.index]).mean()

model_evaluation(testing_target_feature, linear_regression_prediction)

#sets up graph grid
figures, axes = plt.subplots(2,3, figsize=(14,5))

#graphs monthly mean temperature
axes[0][0].plot(mean_monthly_weather.index,mean_monthly_weather["mean_temp"])
graph_labels(0, 0, "Monthly Mean Temperature", "Month", "Mean Temperature (°C)")

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
sns.heatmap(mean_temp_correlation.to_frame(), cmap="coolwarm", annot=True, center=0, ax=axes[0][1])
graph_labels(0, 1, "Temperature Correlation", "Correlation with Mean Temperature", "Weather Variable")

#graphs mean tempareture distribution
mean_temp_distribution = weather["mean_temp"].hist(bins=30, ax=axes[0][2])
graph_labels(0, 2, "Temperature Distribution", "Mean Temperature (°C)", "Frequency (Days)")

#graphs actual vs predicted temperature
axes[1][0].scatter(testing_target_feature, linear_regression_prediction)
graph_labels(1, 0, "Actual vs Predicted Temperature", "Actual Temperature (°C)", "Predicted Temperature (°C)")

axes[1][1].plot(weather["date"].loc[testing_target_feature.index], temperature_residuals)
graph_labels(1, 1, "Prediction Errors Over Time", "Date", "Prediction Error (°C)")

axes[1,2].plot(monthly_temperature_residuals.index, monthly_temperature_residuals)
graph_labels(1, 2, "Monthly Residual MAE", "Month", "Absolute Residual")

#calculates mean temperature distribution statisitics
def mean_temp_stats():
    print(f"Mean: {weather["mean_temp"].mean()}")
    print(f"Median: {weather["mean_temp"].median()}")
    print(f"Std: {weather["mean_temp"].std()}")
    print(f"LQ: {weather["mean_temp"].quantile(0.25)}")
    print(f"UQ: {weather["mean_temp"].quantile(0.75)}\n")

    print(f"Mean of Residuals: {temperature_residuals.mean()}\n")
    print(f"Outliers: \n{abs(temperature_residuals).nlargest(5)}")

mean_temp_stats()

#shows graphs
plt.tight_layout()
plt.show()


