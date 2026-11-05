# London Weather Predictor

A Python machine learning project that analyses historical London weather data and predicts the following day's mean temperature using Linear Regression.

The project explores the process of building a regression model, from data cleaning and exploratory analysis to feature engineering, training and performance evaluation.

## Features

- **Data preprocessing:** Handles missing observations, converts dates and prepares historical weather measurements.
- **Exploratory data analysis:** Visualises monthly temperature trends, feature correlations and temperature distributions.
- **Feature engineering:** Creates historical temperature lag features and cyclical seasonal features using sine and cosine transformations.
- **Machine learning:** Trains a Linear Regression model to predict next-day mean temperature.
- **Model evaluation:** Compares predictions against a persistence baseline using MAE, RMSE and R².
- **Visualisation:** Displays actual versus predicted temperatures, prediction errors over time and monthly prediction errors.

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Dataset

The project uses historical London weather observations from approximately 1979 to 2021, stored in `data/london_weather.csv`.

The dataset includes temperature, precipitation, atmospheric pressure, sunshine, cloud cover and solar radiation measurements.

Rows with missing values in the required fields are excluded during preprocessing. Date continuity is checked when generating historical and next-day temperature features.

## Methodology

### 1. Data Exploration

Historical weather measurements are analysed to identify seasonal patterns, temperature distributions and relationships between meteorological variables.

### 2. Feature Engineering

The model uses current weather measurements alongside additional engineered features:

- Mean temperatures from the previous three days.
- Sine and cosine transformations representing the time of year.
- A target variable containing the following day's mean temperature.

### 3. Model Training

The dataset is split chronologically into approximately 80% training data and 20% testing data.

A Linear Regression model is trained using Scikit-learn. A persistence baseline, which predicts that tomorrow's mean temperature will equal today's, provides a reference for evaluating performance.

### 4. Model Evaluation

The models are evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE) and R².

## Results

| Metric | Persistence Baseline | Linear Regression |
|---|---:|---:|
| MAE | 1.589°C | 0.886°C |
| RMSE | 2.037°C | 1.156°C |
| R² | 0.870 | 0.957 |

*These are preliminary results from the project's chronological test split and should be independently verified before being treated as final.*

Linear Regression reduced MAE by approximately **44%** relative to the persistence baseline.

## Limitations

- The model predicts next-day mean temperature rather than producing a complete weather forecast.
- It uses historical observations from one location.
- Predictions depend on the availability of current-day weather measurements.
- Missing observations are removed rather than imputed.
- The project currently uses one trained regression algorithm.
- A separate validation period and more extensive out-of-sample testing have not yet been implemented.

## Future Improvements

- Compare Linear Regression with Random Forest and other regression algorithms.
- Investigate additional meteorological features.
- Introduce time-series cross-validation.
- Explore hyperparameter optimisation.
- Investigate model performance across different seasons.
- Develop a simple interface for generating predictions.

## Running the Project

1. Clone this repository.
2. Install the required Python dependencies.
3. Place the historical weather CSV file in the `data` directory.
4. Run the main Python script to train the model, display visualisations and calculate evaluation metrics.

## Project Background

This project was developed to explore practical applications of Python, data analysis and machine learning. It provided experience in preparing real-world datasets, engineering predictive features and evaluating regression models against a simple baseline.