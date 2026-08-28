# Weather Prediction Pipeline

This project contains an end-to-end machine learning pipeline for time series forecasting of weather station data (e.g., temperature). It supports multi-horizon forecasting (predicting multiple steps into the future) and integrates exogenous data like official ARSO forecasts.

## Data Flow

The typical lifecycle of data through the pipeline( `pipeline.py`) is as follows:

1. **Data Fetching (`shared/db.py`)**: Raw historical weather data is downloaded from an MSSQL database. Exogenous data (ARSO forecasts) is queried from a PostgreSQL database.
2. **Imputation (`shared/impute.py`)**: Missing data points in the time series are identified and imputed to ensure a continuous index.
3. **Feature Engineering (`shared/features.py`)**: Time-series features are generated from the raw target variable using `tsfresh` over rolling windows. Targets for each horizon are constructed here as well.
4. **Feature Ranking (`shared/rank_features.py`)**: Given the large number of generated features, a lightweight random forest is used to rank feature importances, outputting the top N features.
5. **Model Training (`shared/train.py`)**: The top N endogenous features and exogenous ARSO forecasts are fed into a custom model (`MultiHorizonForest`). The model trains separate estimators for each target horizon (e.g., next 96 steps).
6. **Evaluation (`shared/evaluate.py`) & Logging (`pipeline.py`)**: The model's performance on a test set is evaluated (MSE, MAE). Metrics and execution times are appended to `model_runs.jsonl`.
7. **Inference / Real-time Prediction (`shared/predict_realtime.py`)**: For live forecasting, the pre-trained model fetches just enough recent data to calculate rolling features, then outputs a forecast dataframe for the upcoming timeline.
## Directory and File Structure

- `pipeline.py`: The main executable script that sequences feature extraction, feature ranking, and model training/logging for a full offline training run.
- `model_runs.jsonl`: Log file appending metrics, configurations, and timings of each execution of `pipeline.py`.
- `data/`: Local cache directory storing intermediate outputs like generated features, aligned targets, and feature importance rankings (CSVs).
- `models/`: Directory where finalized model artifacts (`.pkl` files) are serialized.
- `plots/`: Directory for saved evaluation plots.

### `shared/` Modules
- `shared/data_config.py` & `model_config.py`: Configuration files centralizing paths, database credentials, window sizes, and ML parameter groupings cleanly.
- `shared/db.py`: Scripts handling connections to MSSQL and PostgreSQL to fetch datasets.
- `shared/impute.py`: Logic for interpolating/filling missing values in raw timestamps.
- `shared/features.py`: The `tsfresh` feature extraction and target alignment logic.
- `shared/rank_features.py`: Calculates feature importances and saves rankings to filter out noise.
- `shared/multi_horizon_forest.py`: A scikit-learn compatible estimator mapping Random Forests over multiple output horizons.
- `shared/train.py`: Initializes the model class, limits inputs down to top-ranked features + exogenous data, and runs the main `fit` block.
- `shared/evaluate.py`: Loads a saved model artifact mapping it against an exact test split to extract MAE / MSE.
- `shared/predict.py`: Evaluates visually by extracting plots overlaying predictions on top of actuals and baseline forecasts.
- `shared/predict_realtime.py`: A live-production script. Downloads the most recent rolling window of data, builds features, imports the model, and outputs a forecasted trajectory in real-time.

## Running the Pipeline

To run a full offline feature-generation and model-training loop:
```bash
python3 pipeline.py --start-date 2026-03-01 --end-date 2026-04-01
```

To run a real-time prediction using the trained model:
```bash
python3 shared/predict_realtime.py
```