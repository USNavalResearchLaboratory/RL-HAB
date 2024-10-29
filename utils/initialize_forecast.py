from env.forecast_processing.forecast import Forecast, Forecast_Subset
from env.config.env_config import env_params
import pandas as pd

def initialize_forecasts():
    """
    This is the common setup for many scripts that rely on ERA5 and Synth Forecasts

    Right now, we assume Synth and ERA5 forecasts have the same area dimensions.

    Time in Synth needs to be contained within ERA5 (Which are much smaller files, due to lack of levels)
    """
    FORECAST_SYNTH = Forecast(env_params['synth_netcdf'], forecast_type="SYNTH", timewarp=3)
    # Get month associated with Synth
    synth_month = pd.to_datetime(FORECAST_SYNTH.TIME_MIN).month
    # Then process ERA5 to span the same timespan as a monthly Synthwinds File
    FORECAST_ERA5 = Forecast(env_params['era_netcdf'], forecast_type="ERA5", month=synth_month, timewarp=3)

    forecast_subset_synth = Forecast_Subset(FORECAST_SYNTH)
    forecast_subset_era5 = Forecast_Subset(FORECAST_ERA5)

    return FORECAST_SYNTH, FORECAST_ERA5, forecast_subset_era5, forecast_subset_synth



