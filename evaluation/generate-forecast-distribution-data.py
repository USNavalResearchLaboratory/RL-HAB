"""
THis script shows an examples to generate a CSV file with forecast scores for the same coordinate in a Synth and ERA5 forecast.
"""

from env.forecast_processing.forecast import Forecast, Forecast_Subset
from env.forecast_processing.ForecastClassifier import ForecastClassifier
from env.config.env_config import env_params
import pandas as pd


FORECAST_SYNTH = Forecast(env_params['synth_netcdf'], forecast_type="SYNTH")
month = pd.to_datetime(FORECAST_SYNTH.TIME_MIN).month
# Then process ERA5 to span the same timespan as a monthly Synthwinds File
FORECAST_ERA5 = Forecast(env_params['era5_netcdf'], forecast_type="ERA5", month=month)
ForecastClassifier = ForecastClassifier()

#initialize forecast subsets
forecast_subset_era5 = Forecast_Subset(FORECAST_ERA5)
forecast_subset_synth = Forecast_Subset(FORECAST_SYNTH)

days = 1


#Keep track of overall evaluation variables
synth_scores = []
ERA5_scores = []


for i in range (200):
    synth_score = -1
    ERA5_score = -1
    while ERA5_score < env_params['forecast_score_threshold']:
        # Randomize ERA5 forecast coord
        forecast_subset_era5.randomize_coord()
        forecast_subset_era5.subset_forecast(days=days)

        # Then assign same coord to synth winds
        forecast_subset_synth.assign_coord(lat=forecast_subset_era5.lat_central,
                                                lon=forecast_subset_era5.lon_central,
                                                timestamp=forecast_subset_era5.start_time)
        forecast_subset_synth.subset_forecast(days=days)

        _, synth_score = ForecastClassifier.determine_OW_Rate(forecast_subset_synth)
        _, ERA5_score = ForecastClassifier.determine_OW_Rate(forecast_subset_era5)

    synth_scores.append(synth_score)
    ERA5_scores.append(ERA5_score)
    print(i, "Synth", synth_score, "ERA5", ERA5_score)


#Make Dataframe with overall scores
df = pd.DataFrame({'Synth_Forecast_Score': synth_scores,
                    'ERA5_Forecast_Score': ERA5_scores})

eval_dir = "evaluation/EVALUATION_DATA/"
df.to_csv(eval_dir + "THIS IS A TEST.csv")
