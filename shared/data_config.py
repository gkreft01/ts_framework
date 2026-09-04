"""
This script contains configuration settings for the project, such as database
credentials, file paths, and model parameters.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent  # goes up from shared/
DATA_DIR = ROOT / "data"
PLOTS_DIR = ROOT / "plots"

# Database connection parameters
DB_SERVER = None
DB_DATABASE = 'example.db'
DB_USER = None
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_TABLE = 'HistoricalData'
DB_TYPE = 'local'  # Options: 'local', 'mssql', 'postgres'

# Model save paths
MODELS_DIR = ROOT / "models"

FORECASTS_SERVER = None
FORECASTS_DATABASE = "example.db"
FORECASTS_USER = None
FORECASTS_PASSWORD = os.getenv('FORECASTS_PASSWORD')
FORECASTS_TABLE = "ForecastsData"
FORECASTS_TYPE = "local"  # Options: 'local', 'mssql', 'postgres'

# Output database connection parameters
OUTPUT_DB_SERVER = None
OUTPUT_DB_DATABASE = "example.db"
OUTPUT_DB_USER = None
OUTPUT_DB_PASSWORD = os.getenv('OUTPUT_DB_PASSWORD')
OUTPUT_DB_TABLE = "OutputForecasts"
OUTPUT_DB_TYPE = 'local'

COUNTRY_CODE = 'SI'
