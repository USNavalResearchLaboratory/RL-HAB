import csv
import numpy as np
import matplotlib.pyplot as plt

# Select heatmap options
verbose = False
cutoff_forecast_score = 0.01
heat_map_mask_threshold = 1.0
eval_dir = "evaluation/EVALUATION_DATA/"
filename = eval_dir + "DUAL-Jul-on-Jul-USA-hopeful-pyramid-piecewise.csv"
filename = eval_dir + "TIMEWARP-DUAL-Apr-on-Jul-USA-effortless-blaze-piecewise_TEST.csv"
filename = eval_dir + "TIMEWARP-DUAL-Jul-on-Jul-USA-genial-shadow-piecewise_TEST.csv"

filename = eval_dir + "baseline_july_DUAL.csv"
filename = eval_dir + "baseline_july_DUAL_thresholded.csv"

# Load data from CSV
forecast_scores = []
twr_scores = []
with open(filename, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        item = float(row['Forecast_Score'])
        if item > cutoff_forecast_score:
            forecast_scores.append(item)
            twr_scores.append(float(row["TWR_Score"]))

# Print Overall Statistics
print(f"Average TWR: {np.mean(twr_scores)}")
print(f"Standard Dev TWR: {np.std(twr_scores)}")

print(f"Average Forecast: {np.mean(forecast_scores)}")
print(f"Standard Dev Forecast: {np.std(forecast_scores)}")

# Define the bins for the 2D histogram, this is manually set for now,  Need to change based on X: Forecast Score Discretizations, Y: TWR Score Discretizations
bins_x = np.linspace(0, 1, 29)
bins_y = np.linspace(0, 1200, 50)

print(len(forecast_scores), len(twr_scores))

# Create the 2D histogram
heatmap, xedges, yedges = np.histogram2d(forecast_scores, twr_scores, bins=[bins_x, bins_y])

col_sum = heatmap.sum(axis=1)
print(print(f"Total # trials: {np.sum(col_sum)}"))

if verbose:

    print(f"Row sums (total counts per TWR score bin): {col_sum}")

    # Print the number of items in each bin along with the bin ranges
    print("Bin details:")
    for i in range(len(xedges) - 1):
        for j in range(len(yedges) - 1):
            x_range = (xedges[i], xedges[i + 1])
            y_range = (yedges[j], yedges[j + 1])
            bin_value = heatmap[i, j]
            print(f"Bin {i + 1},{j + 1} | X range: {x_range} | Y range: {y_range} | Count: {bin_value}")

# **************** Frequency Heatmap Masking *****************
# Mask values below 1.0 (for frequency heatmap)
heatmap_masked = np.ma.masked_less(heatmap, heat_map_mask_threshold)

# Use the viridis colormap and set masked values (NaNs) to black
cmap_freq = plt.cm.viridis
cmap_freq.set_bad(color='black')

# ***************** Percentage Heatmap Normalization ***************
# Normalize the heatmap data by the sum of each column to come up with probability percantage
col_sum[col_sum == 0] = 1  # Avoid division by zero
heatmap_normalized = (heatmap.T / col_sum).T  # Normalize by dividing by column sum
col_sum = heatmap_normalized.sum(axis=1)

if verbose:
    print(f"Row sums (total counts per TWR score bin in %): {col_sum}")

# Use normal color map for percentage
cmap_perc = plt.cm.magma

# ************Plotting heatmaps*************

# Calculate bin centers for the x-axis
x_centers = 0.5 * (xedges[:-1] + xedges[1:])

# Frequency Heatmap
plt.figure(figsize=(8, 7))

ax2 = plt.gca()

# ---- Plot Frequency HeatMap ----#
# The frequency heatmap simple plots how many occurances of ForecastScore/TWR discretizations occurs and color codes the boxes accordingly
plt.imshow(heatmap_masked.T, origin='lower', cmap=cmap_freq, extent=[0, 1, 0, 1200], aspect='auto')
plt.colorbar(label='Frequency of Forecast Column', extend="max")
plt.clim(5,40.)
plt.xlabel('Forecast Score')
plt.ylabel('TWR50 (%)')

yticks = ax2.get_yticks()
ax2.set_yticks(yticks)  # Fix the tick locations
ax2.set_yticklabels([f'{int(tick / 1200*100)}' for tick in yticks])


plt.tight_layout()
plt.title(f'Heatmap (Frequency) - ({filename})')



# ---- Plot Percentage HeatMap ----#
# The percentage heatmap converts boxes to percent based on overall column count.
plt.figure(figsize=(8, 7))
ax = plt.gca()
plt.imshow(heatmap_normalized.T, origin='lower', cmap=cmap_perc, extent=[0, 1, 0, 1200], aspect='auto')
plt.colorbar(label='Distribution')
plt.clim(0,1.)
plt.xlabel('Forecast Score')
plt.ylabel('TWR50 (%)')

yticks = ax.get_yticks()
ax.set_yticks(yticks)  # Fix the tick locations
ax.set_yticklabels([f'{int(tick / 1200*100)}' for tick in yticks])

plt.title(f'Heatmap (Percentage) - ({filename})')
plt.tight_layout()
plt.show()


