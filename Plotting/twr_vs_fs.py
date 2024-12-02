import pandas as pd
import matplotlib.pyplot as plt

eval_dir = "evaluation/EVALUATION_DATA/"

df_model = pd.read_csv(eval_dir + "DUAL-Jul_cust-on-Oct-USA-charmed-resonance-piecewise.csv")

print(df_model)

#plt.scatter(df_model['Forecast_Score'], df_model['TWR_Score'])


# Group by 'Forecast_Score' and calculate the mean and standard deviation of 'TWR_Score' for each group
df_grouped = df_model.groupby('Forecast_Score', as_index=False).agg(
    mean_TWR=('TWR_Score', 'mean'),
    std_TWR=('TWR_Score', 'std')
)

# Sort the dataframe by Forecast_Score for a smooth line plot
df_grouped = df_grouped.sort_values('Forecast_Score')

# Plotting the mean TWR scores with error bars (standard deviation)
#plt.errorbar(df_grouped['Forecast_Score'], df_grouped['mean_TWR'],
#             yerr=df_grouped['std_TWR'], fmt='-o', color='orange',
#             ecolor='orange', elinewidth=2, capsize=4, label = "model")

plt.plot(df_grouped['Forecast_Score'], df_grouped['mean_TWR'], marker='o', label = "model")

# Adding labels and title
plt.xlabel('Forecast Score')
plt.ylabel('Mean TWR Score')
plt.title('Mean TWR Score vs Forecast Score with Standard Deviation Error Bars')

df_baseline = pd.read_csv(eval_dir + "baseline_SINGLE_SYNTH_rogue_seed2/SINGLE_SYNTH-baseline-on-Oct-USA-rogue.csv")
print(df_baseline)

df_grouped = df_baseline.groupby('Forecast_Score', as_index=False).agg(
    mean_TWR=('TWR_Score', 'mean'),
    std_TWR=('TWR_Score', 'std')
)

# Sort the dataframe by Forecast_Score for a smooth line plot
df_grouped = df_grouped.sort_values('Forecast_Score')

# Plotting the mean TWR scores with error bars (standard deviation)
#plt.errorbar(df_grouped['Forecast_Score'], df_grouped['mean_TWR'],
#             yerr=df_grouped['std_TWR'], fmt='-o', color='b',
#             ecolor='b', elinewidth=2, capsize=4, label = "baseline_SINGLE_SYNTH")

plt.plot(df_grouped['Forecast_Score'], df_grouped['mean_TWR'], marker='o', label = "baseline_DUAL")

df_baseline = pd.read_csv(eval_dir + "baseline_DUAL_rogue_seed2/DUAL-baseline-on-Oct-USA-rogue.csv")
print(df_baseline)

df_grouped = df_baseline.groupby('Forecast_Score', as_index=False).agg(
    mean_TWR=('TWR_Score', 'mean'),
    std_TWR=('TWR_Score', 'std')
)

# Sort the dataframe by Forecast_Score for a smooth line plot
df_grouped = df_grouped.sort_values('Forecast_Score')

# Plotting the mean TWR scores with error bars (standard deviation)
#plt.errorbar(df_grouped['Forecast_Score'], df_grouped['mean_TWR'],
#             yerr=df_grouped['std_TWR'], fmt='-o', color='r',
#             ecolor='r', elinewidth=2, capsize=4, label = "baseline_DUAL")

plt.plot(df_grouped['Forecast_Score'], df_grouped['mean_TWR'], marker='o', label = "baseline_SINGLE_SYNTH")

plt.legend()

#plt.scatter(df_baseline['Forecast_Score'], df_baseline['TWR_Score'])

plt.show()