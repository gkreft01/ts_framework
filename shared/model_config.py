from typing import Dict, List, Tuple

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from shared.data_config import DATA_DIR
# --- Pipeline & Model Parameters ---
class ModelConfig:
    """
    A class to hold all model and feature engineering parameters.
    This makes it easy to pass them around and to create different experiment setups.
    """
    def __init__(self,
                 config_name: str,
                 target_name: str,
                 target_diff_lag: int = 0,
                 target_smooth_size: int = 1,
                 feature_params: Dict[str, List[Tuple[int, int, bool]]] = None,
                 horizon: int = 96,
                 top_n_features: int = 20,
                 test_size: int = 7,
                 additional_forecasts: dict = None,
                 additional_forecasts_mapping: dict = None,
                 model_type = RandomForestRegressor,
                 model_kwargs = None,
                 pvlib_transform: bool = False):
        
        self.CONFIG_NAME = config_name
        self.TARGET_NAME = target_name
        self.TARGET_DIFF_LAG = target_diff_lag
        self.TARGET_SMOOTH_SIZE = target_smooth_size
        self.FEATURE_PARAMS = feature_params
        self.HORIZON = horizon
        self.TOP_N_FEATURES = top_n_features
        self.TEST_SIZE = test_size # days for testing
        self.DATA_DIR = DATA_DIR / config_name
        
        # Model specific parameters
        self.MODEL_TYPE = model_type
        self.MODEL_KWARGS = model_kwargs
        self.PVLIB_TRANSFORM = pvlib_transform

        #Additional data sources

    #   # Available columns:
    #        'precip_accumulated_mm', 'snow_accumulated_cm', 'weather_summary_text',
    #        'cloud_cover_text', 'cloud_icon_code', 'temperature_min_c',
    #        'temperature_max_c', 'temperature_c', 'wind_icon_code',
    #        'wind_direction_text', 'wind_speed_category', 'wind_speed_avg_ms',
    #        'wind_speed_max_ms', 'weather_icon_code'
        self.ADDITIONAL_FORECASTS = additional_forecasts
        self.ADDITIONAL_FORECASTS_MAPPING = additional_forecasts_mapping

    def __str__(self):
        return f"ModelConfig(config_name={self.CONFIG_NAME}, target_name={self.TARGET_NAME}, feature_params={self.FEATURE_PARAMS}, horizon={self.HORIZON}, top_n_features={self.TOP_N_FEATURES}, n_estimators={self.MODEL_KWARGS.get('n_estimators') if self.MODEL_KWARGS else None}, max_depth={self.MODEL_KWARGS.get('max_depth') if self.MODEL_KWARGS else None}, additional_forecasts={self.ADDITIONAL_FORECASTS}, additional_forecasts_mapping={self.ADDITIONAL_FORECASTS_MAPPING})"
    
    def to_dict(self):
        return {
            "config_name": self.CONFIG_NAME,
            "target_name": self.TARGET_NAME,
            "feature_params": self.FEATURE_PARAMS,
            "horizon": self.HORIZON,
            "top_n_features": self.TOP_N_FEATURES,
            "test_size": self.TEST_SIZE,
            "model_type": self.MODEL_TYPE.__name__ if self.MODEL_TYPE else None,
            "model_kwargs": self.MODEL_KWARGS,
            "additional_forecasts": self.ADDITIONAL_FORECASTS,
            "additional_forecasts_mapping": self.ADDITIONAL_FORECASTS_MAPPING
        }
    

from typing import NamedTuple, Union

class FeatureSpec(NamedTuple):
    win_size: int
    agg_size: int
    minimal: bool
    smooth_size: int = 1
    diff_lag: int = 0  # 0 = no differencing