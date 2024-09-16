from era5.forecast import Forecast, Forecast_Subset
from era5.ForecastClassifier import ForecastClassifier
from env3d.config.env_config import env_params
import pandas as pd

synth_filename = "../../../../mnt/d/FORECASTS/SYNTH-Oct-2023-USA-UPDATED.nc"
era5_filename = "../../../../mnt/d/FORECASTS/ERA5-H2-2023-USA.nc"


FORECAST_SYNTH = Forecast(synth_filename, forecast_type="SYNTH")
month = pd.to_datetime(FORECAST_SYNTH.TIME_MIN).month
# Then process ERA5 to span the same timespan as a monthly Synthwinds File
FORECAST_ERA5 = Forecast(era5_filename, forecast_type="ERA5", month=month)

ForecastClassifier = ForecastClassifier()

forecast_subset_era5 = Forecast_Subset(FORECAST_ERA5)
forecast_subset_era5.randomize_coord()

days =1

#Then assign coord to synth winds
forecast_subset_synth = Forecast_Subset(FORECAST_SYNTH)
forecast_subset_synth.assign_coord(lat = forecast_subset_era5.lat_central,
                                        lon = forecast_subset_era5.lon_central,
                                        timestamp= forecast_subset_era5.start_time)
forecast_subset_synth.subset_forecast(days=days)




#Keep track of overall evaluation variables for creating heatmaps

synth_scores = []
ERA5_scores = []


for i in range (10_000):
    synth_score = -1
    ERA5_score = -1
    while synth_score < env_params['forecast_score_threshold']:
        forecast_subset_era5.randomize_coord()
        forecast_subset_era5.subset_forecast(days=days)

        # Then assign coord to synth winds
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


df.to_csv("Oct-ERA5-with-zero.csv")
