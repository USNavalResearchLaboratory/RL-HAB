import os
import pandas as pd
import matplotlib.pyplot as plt
import re

# Parameters
base_folder = "evaluation/EVALUATION_DATA/"  # Base folder containing all sub-eval directories
sub_eval_directories = [
    "DUAL_IMITATION_swept-eon_Jan",
    "DUAL_IMITATION_logical-moon-2_Mar",
    "DUAL_IMITATION_hardy-wind-2_Oct",
    "DUAL_IMITATION_hardy-wind-baseline_Oct",
    "DUAL_IMITATION_misty-grass_Sep",
    "baseline_DUAL_rogue_seed2",
    #"baseline_SINGLE_ERA5_rogue_seed2",
    "baseline_SINGLE_SYNTH_rogue_seed2",
    #"DUAL_IMITATION_silvery-jazz_Jul",
]  # List of sub-eval directories
forecast_cutoff_score = 0.01  # Threshold for filtering
variable_to_plot = "TWR_Score"  # Column name to plot (e.g., 'Forecast_Score', 'Rogue_Status')
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
colors = ["blue", "green", "orange", "darkorange", "red", "black", "purple", "purple", "pink", "cyan", "black"]

# Initialize dataframes for storing means and standard deviations
mean_df = pd.DataFrame(index=months)
std_df = pd.DataFrame(index=months)

# Process each sub-eval directory
for sub_eval in sub_eval_directories:
    sub_folder = os.path.join(base_folder, sub_eval)
    if not os.path.isdir(sub_folder):
        print(f"Sub-directory {sub_folder} not found, skipping.")
        continue

    # Initialize lists to store results for each month
    mean_values = []
    std_values = []

    # Process each file in the sub-eval directory
    for file in os.listdir(sub_folder):
        if file.endswith(".csv"):
            # Match filenames of the structure DUAL_IMITATION-Oct-on-Apr-hardy-wind-2.csv
            match = re.match(r".*-(\w+)-on-(\w+)-.*\.csv", file)
            if not match:
                continue

            # Extract target month from the filename
            target_month = match.group(2)
            if target_month not in months:
                continue

            file_path = os.path.join(sub_folder, file)

            try:
                # Load and filter data
                data = pd.read_csv(file_path)
                data = data[data['Forecast_Score'] >= forecast_cutoff_score]

                # Calculate mean and standard deviation for the selected variable
                mean_value = data[variable_to_plot].mean()
                std_value = data[variable_to_plot].std()
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                mean_value, std_value = float('nan'), float('nan')

            # Store the values in the appropriate position
            mean_df.loc[target_month, sub_eval] = mean_value
            std_df.loc[target_month, sub_eval] = std_value

# Plotting
plt.figure(figsize=(12, 6))
color_idx = 0
for sub_eval in mean_df.columns:
    plt.errorbar(
        mean_df.index, mean_df[sub_eval],
        yerr=std_df[sub_eval],
        marker='o', alpha=0.75, linestyle='-', label=sub_eval,
        capsize=5, capthick=1, color=colors[color_idx % len(colors)]
    )
    color_idx += 1

plt.title(f'Mean {variable_to_plot} by Month (Filtered FS > {forecast_cutoff_score})')
plt.xlabel('Month')
plt.ylabel(f'Mean {variable_to_plot}')
plt.xticks(rotation=45)
plt.legend(title='Sub Eval Directories')
plt.grid()
plt.tight_layout()
plt.show()
