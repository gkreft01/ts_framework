# Time Series Forecasting Framework

A configurable machine-learning framework for short-term time-series forecasting. The repository includes a heat pump forecasting example using the framework.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configure an Experiment

Experiments are defined in:

```text
shared/experiments.py
```

Create or modify a `ModelConfig` to specify the target variable, feature windows, forecasting horizon, model, test period, and optional external forecasts.

For example:

```python
HEAT_PUMP_CONFIG = ModelConfig(
    config_name="heat_pump",
    target_name='Active_Power',
    target_smooth_size=8,
    freq='15min',
    horizon=96,
    feature_params={'Active_Power': [FeatureSpec(96, 1, False, 8, 0), FeatureSpec(96*7, 96, False)]},
    top_n_features=50,
    model_type=RandomForestRegressor,
    test_size=14,
    model_kwargs={"n_estimators": 100, "max_features": 'log2', "random_state": 42},
    additional_forecasts={"Letališče Jožeta Pučnika Ljubljana": ['temperature_c', 'precip_accumulated_mm']}
)

```


## Data Sources

Adapt:

```text
shared/data_config.py
```

to connect to your own databases or other data sources.

The repository contains `example.db` for the provided example.

## Run the Pipeline

Run an experiment with:

```bash
python3 pipeline.py \
    --start-date 2026-03-01 \
    --end-date 2026-05-01 \
    --config HEAT_PUMP_CONFIG
```

Replace `HEAT_PUMP_CONFIG` with the configuration you want to run.

## Real-Time Prediction

Use a trained model for real-time forecasting with:

```bash
python3 shared/predict_realtime.py --config-name heat_pump
```

Replace `heat_pump` with the `config_name` of the experiment.

