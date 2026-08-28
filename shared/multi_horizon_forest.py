from turtle import pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from tqdm import tqdm
import pandas as pd

class MultiHorizonForest(BaseEstimator, RegressorMixin):
    """
    Trains one RandomForestRegressor per forecast horizon.
    Predicts 192 steps ahead (2 days at 15min resolution, or 48h hourly etc.)
    """
    def __init__(self, n_horizons=192, model_type=RandomForestRegressor, **model_kwargs):
        self.n_horizons = n_horizons
        self.model_type = model_type
        self.model_kwargs = model_kwargs
        self.models_ = []
        self.columns_ = None
        self.additional_feature_names = None

    def fit(self, X, Y, X_additional=None):
        # Y shape: (n_samples, n_horizons)
        self.columns_ = X.columns.tolist()  # Store feature names

        # We must copy X to avoid modifying the original dataframe when adding features
        X_h = X.copy()

        for h in tqdm(range(self.n_horizons)):
            model = self.model_type(**self.model_kwargs)

            #self.additional_features is a list of DataFrames with additional features for each horizon, if provided
            #the column names in the additional features are '{var_name}_h1', '{var_name}_h2', etc. where h1, h2 etc. correspond to the horizon
            if X_additional is not None:
                # Need to use a fresh copy for each horizon if we are adding specific horizon columns
                # Alternatively, we can just replace the column in X_h if it already exists
                X_h = X.copy()
                for additional_df in X_additional:
                    col = additional_df.columns[0][:-3]  # remove _h1, _h2 etc. suffix

                    # append col to feature names if not already present
                    if col not in self.columns_:
                        self.columns_.append(col)

                    additional_col = f"{col}_h{h+1}"
                    if additional_col in additional_df.columns: # Add to feature names
                        X_h = X_h.merge(additional_df[[additional_col]], left_index=True, right_index=True, how='left', suffixes=('', f'_h{h+1}'))
                    else:
                        print(f"Warning: {additional_col} not found in additional features for horizon {h+1}. Skipping this feature.")
                        continue
                
            model.fit(X_h, Y.iloc[:, h])
            self.models_.append(model)
        return self

    def predict(self, X, X_additional=None):
        # returns pandas DataFrame of shape (n_samples, n_horizons) with time index and horizon columns
        predictions = np.zeros((X.shape[0], self.n_horizons))
        for h, model in enumerate(self.models_):
            if X_additional is not None:
                X_h = X.copy()
                for additional_df in X_additional:
                    col = additional_df.columns[0][:-3]  # remove _h1, _h2 etc. suffix
                    additional_col = f"{col}_h{h+1}"

                    if additional_col in additional_df.columns:
                        X_h = X_h.merge(additional_df[[additional_col]], left_index=True, right_index=True, how='left', suffixes=('', f'_h{h+1}'))
                    else:
                        print(f"Warning: {additional_col} not found in additional features for horizon {h+1}. Skipping this feature.")
                        continue

                predictions[:, h] = model.predict(X_h)
            else:
                predictions[:, h] = model.predict(X)
        return pd.DataFrame(predictions, index=X.index, columns=[f'h{i+1}' for i in range(self.n_horizons)])
    
    def feature_importances(self):
        # Average feature importances across all horizons
        # Return a DataFrame with feature names and average importance

        # ONLY WORKS FOR RANDOM FOREST
        importances = np.array([model.feature_importances_ for model in self.models_])
        avg_importances = np.mean(importances, axis=0)

        return pd.DataFrame({'feature': self.columns_, 'importance': avg_importances})