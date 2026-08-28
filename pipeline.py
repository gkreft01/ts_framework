from time import time
from shared.features import main as features
from shared.rank_features import main as rank_features
from shared.train import main as train
from shared.evaluate import main as evaluate
import argparse

from shared import model_config, experiments

import json
import time
from datetime import datetime

def log_run(config, start_date, end_date, execution_time, metrics, file_path="model_runs.jsonl"):
    """
    Appends the details of a single model run to a JSON Lines file.

    Args:
        config (object): The configuration object used for the run.
        metrics (dict): A dictionary of performance metrics (e.g., {'mse': 0.5}).
        execution_time (float): The time it took to run the pipeline in seconds.
        file_path (str): The path to the log file.
    """
    run_data = {
        "timestamp": datetime.now().isoformat(),
        "execution_time_seconds": round(execution_time, 2),
        "start_date": start_date,
        "end_date": end_date,
    }

    # Unpack the config dictionary and merge it into run_data
    run_data = {**run_data, **metrics, **config.to_dict()}

    # Append the run data as a new line in the JSONL file
    with open(file_path, 'a') as f:
        f.write(json.dumps(run_data) + '\n')

    print(f"Run logged to {file_path}")



def run_pipeline(config: model_config.ModelConfig, start_date: str = None, end_date: str = None):

    start_time = time.time()

    print("--- Step 1: computing features ---")
    X_train, X_test, Y_train, Y_test = features(config, start_date, end_date)
    
    print("--- Step 2: ranking features ---")
    importances = rank_features(config, X_train, Y_train)
    
    print("--- Step 3: training model ---")
    model = train(config, X_train, Y_train, importances)

    print("--- Step 4: evaluating model ---")
    metrics = evaluate(config, model, X_test, Y_test, importances)

    end_time = time.time()
    execution_time = end_time - start_time
    log_run(config, start_date, end_date, execution_time, metrics)
    
    
    return model

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run the full pipeline for a given variable.")
    parser.add_argument("--start-date", type=str, default="2026-02-01", help="Start date for the data to use (format: YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=datetime.now().strftime('%Y-%m-%d'), help="End date for the data to use (format: YYYY-MM-DD)")
    parser.add_argument("--config", type=str, help="The name of the configuration to use for the forecast.")

    args = parser.parse_args()
    
    if not hasattr(experiments, args.config):
        parser.error(
            f"Unknown config '{args.config}'. "
            f"Available configs: "
            f"{[name for name in dir(experiments) if not name.startswith('_')]}"
        )

    config = getattr(experiments, args.config)
    run_pipeline(config, start_date=args.start_date, end_date=args.end_date)