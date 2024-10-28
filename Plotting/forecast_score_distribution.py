import csv
import numpy as np
import matplotlib.pyplot as plt

"""
Before running this script run evaluation/generate-forecast-distribution-data.py with the desired ERA5 and SYnth 
forecasts (month and region should match).  Also decide if include 0 forecast scores or not. 

This script makes forecast score distribution plots for multiple months of the same type of forecast (ERA5 or Synth). 

options:
forecast_type:  Can be Synth or ERA5
raw_filenames:  Month forecast distribution csv's generated from evaluation/generate-forecast-distribution-data.py

"""


forecast_type = "ERA5" # Can be Synth or ERA5
# Forecast filenames example
raw_filenames = ["Jan-ERA5-no-zero-updated.csv", "Apr-ERA5-no-zero-updated.csv", "Jul-ERA5-no-zero-updated.csv", "Oct-ERA5-no-zero-updated.csv"]
#raw_filenames = ["Jan-ERA5-with-zero.csv", "Apr-ERA5-with-zero.csv", "Jul-ERA5-with-zero.csv", "Oct-ERA5-with-zero.csv"]

def forecastDistributionPlots(bar_enabled: bool):
    # Define consistent color scheme (darker for splines, lighter for bars)
    colors = ['#0000FF', '#008000', '#FF0000', '#FFA500']  # Darker colors for splines
    light_colors = ['#99CCFF', '#99FF99', '#FF9999', '#FFD1A1']  # Lighter for bars

    plt.figure(figsize=(10, 5))

    bins_x = np.linspace(0, 1, 29)  # Common x bins for all datasets
    print(bins_x)

    # List to hold the column sums for all datasets
    all_col_sums = []

    # Loop through each file and calculate the column sums
    for raw_filename in raw_filenames:
        filename = "evaluation/EVALUATION_DATA/" + raw_filename
        forecast_scores = []
        twr_scores = []

        count = 0
        zero_count = 0
        with open(filename, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                item = float(row[forecast_type + '_Forecast_Score']) #
                if item == 0.0:
                    zero_count += 1
                if item > -0.1:
                    count += 1
                    forecast_scores.append(item)
                    twr_scores.append(float(row[""]))  # Replace with TWR key if needed (empty column for now)

        #bins_y = np.linspace(0, 1200, 50)
        bins_y = np.linspace(0, 10000000, 50) # for when TWR column doesn't exist

        # Create the 2D histogram
        heatmap, _, _ = np.histogram2d(forecast_scores, twr_scores, bins=[bins_x, bins_y])

        col_sum = heatmap.sum(axis=1)

        # Normalize col_sum to percentages
        col_sum_percentage = (col_sum / count)
        all_col_sums.append(col_sum_percentage)

        print(f"{raw_filename.split('-')[0]} Zero %: {zero_count/count}")
        print(f"{raw_filename.split('-')[0]} Mean: {np.mean(forecast_scores)}, Std: {np.std(forecast_scores)}")

    # Plot bars (if enabled)
    if bar_enabled:
        label_added = [False] * len(raw_filenames)  # Track if a label has been added for each dataset
        for i in range(len(bins_x) - 1):  # Loop over each bin of forecast scores
            bin_values = [col_sum[i] for col_sum in all_col_sums]
            sorted_idx = np.argsort(bin_values)[::-1]  # Sort values in descending order

            # Plot each dataset in descending order (so larger bars are at the bottom)
            for idx in sorted_idx:
                if not label_added[idx]:  # Add label only if it hasn't been added
                    plt.bar(bins_x[i], all_col_sums[idx][i], width=np.diff(bins_x)[i], 
                            edgecolor='black', color=light_colors[idx], align='edge', 
                            label=f'{raw_filenames[idx].split("-")[0]}')
                    label_added[idx] = True  # Mark this label as added
                else:
                    plt.bar(bins_x[i], all_col_sums[idx][i], width=np.diff(bins_x)[i], 
                            edgecolor='black', color=light_colors[idx], align='edge')

    # Set up axes labels and limits
    plt.xlabel('Forecast Score', fontsize=20)
    plt.ylabel('Percent of Forecasts', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim(0, 1)  # Ensure the x-axis is between 0 and 1
    plt.ylim(0, 0.25)  # Adjusted ylim for percentage scale
    
    # Display the legend only for the first 4 datasets (no duplicates)
    plt.legend(loc='upper right', fontsize=16)
    plt.tight_layout()
    plt.show()
    plt.close()


def forecastDistributionStats():
    bins_x = np.linspace(0, 1, 29)  # Common x bins for all datasets
    print(bins_x)

    # Loop through each file and calculate the column sums
    for raw_filename in raw_filenames:
        filename = "evaluation/EVALUATION_DATA/" + raw_filename
        era5_forecast_scores = []
        synth_forecast_scores = []
        twr_scores = []
        diff_scores = []

        with open(filename, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                era5_score = float(row['ERA5_Forecast_Score'])
                synth_score = float(row['Synth_Forecast_Score'])
                era5_forecast_scores.append(era5_score)
                synth_forecast_scores.append(synth_score)
                diff_scores.append(synth_score - era5_score)
                twr_scores.append(float(row[""]))  # Replace with correct key if needed


        print(f"ERA5 {raw_filename.split('-')[0]} Mean: {np.mean(era5_forecast_scores)}, Std: {np.std(era5_forecast_scores)}")
        print(f"SYNTH {raw_filename.split('-')[0]} Mean: {np.mean(synth_forecast_scores)}, Std: {np.std(synth_forecast_scores)}")
        print(f"DIFFERENCE {raw_filename.split('-')[0]} Mean: {np.mean(diff_scores)}, Std: {np.std(diff_scores)}")


if __name__ == "__main__":
    forecastDistributionStats()
    forecastDistributionPlots(bar_enabled=True)