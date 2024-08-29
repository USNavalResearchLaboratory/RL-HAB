import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns



df = pd.read_csv('piecewise-20m-random-total_score-non-zero-10k.csv')

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18,5))
fig.suptitle('TWR Score for Piecewise')
ax1.scatter(df["Forecast_Score"], df["TWR_Inner_Score"])
ax2.scatter(df["Forecast_Score"], df["TWR_Score"])
ax3.scatter(df["Forecast_Score"], df["TWR_Outer_Score"])

ax1.set_xlabel("Forecast Classification Score")
ax1.set_ylabel("TWR (25km)")
ax1.set_title("Inner")

ax2.set_xlabel("Forecast Classification Score")
ax2.set_ylabel("TWR (50km)")
ax2.set_title("TWR")

ax3.set_xlabel("Forecast Classification Score")
ax3.set_ylabel("TWR (75km)")
ax3.set_title("Outer")


# Fit linear regression via least squares with numpy.polyfit
# It returns an slope (b) and intercept (a)
# deg=1 means linear fit (i.e. polynomial of degree 1)
b, a = np.polyfit(df["Forecast_Score"], df["TWR_Inner_Score"], deg=1)

# Create sequence of 100 numbers from 0 to 100
xseq = np.linspace(0.2, 1, num=100)

# Plot regression line
#ax2.plot(xseq, a + b * xseq, color="k", lw=2.5)


fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(18,5))
fig.suptitle('TWR Score for Piecewise')

mean_twr_per_score = df.groupby('Forecast_Score')['TWR_Inner_Score'].agg(lambda x: x.value_counts().index[0])
axa.bar(mean_twr_per_score.index.values,mean_twr_per_score.values, color='skyblue')
axa.set_xlabel("Forecast Classification Score")
axa.set_ylabel("TWR (25km)")
axa.set_ylim(0,300)
axa.set_title("Inner")

mean_twr_per_score = df.groupby('Forecast_Score')['TWR_Score'].agg(lambda x: x.value_counts().index[0])
axb.bar(mean_twr_per_score.index.values,mean_twr_per_score.values, color='skyblue')
axb.set_ylim(0,300)
axb.set_xlabel("Forecast Classification Score")
axb.set_ylabel("TWR (50km)")
axb.set_title("TWR")

print(mean_twr_per_score)
print(mean_twr_per_score.index.values)
print(mean_twr_per_score.values)





mean_twr_per_score = df.groupby('Forecast_Score')['TWR_Outer_Score'].agg(lambda x: x.value_counts().index[0])
axc.bar(mean_twr_per_score.index.values,mean_twr_per_score.values, color='skyblue')

axc.set_xlabel("Forecast Classification Score")
axc.set_ylabel("TWR (75km)")
axc.set_title("Outer")
axc.set_ylim(0,300)

plt.figure()

# Step 1: Define bins for twr values
twr_bins = np.linspace(0, 1200, 21)  # 20 bins between 0 and 1

# Step 2: Bin the twr values
df['twr_bin'] = pd.cut(df['TWR_Score'], bins=twr_bins, labels=False)

# Step 3: Create a 2D histogram of Score vs. twr_bin
heatmap_data = pd.crosstab(df['Forecast_Score'], df['twr_bin'])

# Step 4: Replace bin labels with the bin midpoints
bin_midpoints = (twr_bins[:-1])
midpoints = dict(enumerate(bin_midpoints))  # Mapping bin indices to midpoints
heatmap_data.columns = [midpoints[col] for col in sorted(heatmap_data.columns)]

# Step 5: Sort the columns of the heatmap data by the bin midpoints
heatmap_data = heatmap_data.reindex(sorted(heatmap_data.columns), axis=1)

# Step 6: Transpose the heatmap data to switch axes
heatmap_data = heatmap_data.T

# Step 7: Plot the heatmap with Seaborn
plt.figure(figsize=(10, 6))
ax = sns.heatmap(heatmap_data, cmap="viridis", cbar=True, annot=False)

# Invert the y-axis to make TWR values increase from bottom to top
ax.invert_yaxis()

# Set labels
plt.xlabel('Forecast Score')
plt.ylabel('TWR Values')
plt.title('Heatmap of TWR Density')

# Show the plot
plt.show()