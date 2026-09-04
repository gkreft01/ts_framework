from shared.model_config import ModelConfig, FeatureSpec
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

# --- Define Specific Experiments ---

HEAT_PUMP_CONFIG = ModelConfig(
    config_name="heat_pump",
    target_name='Active_Power',
    target_smooth_size=8,
    horizon=96,
    feature_params={'Active_Power': [FeatureSpec(96, 1, False, 8, 0), FeatureSpec(96*7, 96, False)]},
    top_n_features=50,
    model_type=RandomForestRegressor,
    test_size=14,
    model_kwargs={"n_estimators": 100, "max_features": 'log2', "random_state": 42},
    additional_forecasts={"Letališče Jožeta Pučnika Ljubljana": ['temperature_c', 'precip_accumulated_mm']}
)
