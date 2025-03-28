import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from env.config.env_config import env_params

eval_dir = "evaluation/EVALUATION_DATA2/"
#sub_eval = "baseline_DUAL"
baseline_sub_eval = "baseline_DUAL/"
sub_eval = "DUAL_frosty-plasma-36_Jan/"


# Load baseline categorization for TWR percentage bin definitions
#baseline_file = eval_dir + "baseline_DUAL-categorization.csv"  # Adjust this filename as needed
#baseline_df = pd.read_csv(baseline_file)




# List to hold mean TWR_Score for the current evaluation month
mean_twr_scores = []
std_twr_scores = []

months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
#months = ["Jun" ]

# Define bins and labels
bins = [0, 240, 480, 720, 960, 1200]  # 20% intervals of 1200
labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]


e = ["baseline_DUAL"]

df_list = []
for month in months:
    #csv_name = eval_dir + sub_eval + "/DUAL-baseline-on-" + month +"-USA-new.csv"
    csv_name =eval_dir + sub_eval + env_params["eval_type"] + "-" + env_params["model_month"] + "-on-" + month + "-" + env_params["eval_model"] + "-deterministic.csv"
    print(csv_name)
    
    
    
    
    csv_baseline = eval_dir + baseline_sub_eval + "/DUAL-baseline-on-" + month +"-USA-new.csv"
    df_baseline = pd.read_csv(csv_baseline)
    
    
    df = pd.read_csv(csv_name)
    print(df)
    
    # TO compare with baseline's performance
    # df["TWR_Score"] = df_baseline["TWR_Score"]
    
    print(df)
    print(df_baseline)
    #Sdfsdf
    
    # Categorize TWR_Score into bins
    df['TWR_Percentage_Range'] = pd.cut(df["TWR_Score"], bins=bins, labels=labels, right=False)
    
    # Count occurrences in each bin
    twr_counts = df["TWR_Percentage_Range"].value_counts().sort_index()
    
    # Compute mean and standard deviation of Forecast_Score for each TWR_Percentage_Range
    forecast_stats = df.groupby("TWR_Percentage_Range")["Forecast_Score"].agg(["mean", "std"])
    
    # Merge counts with forecast statistics
    twr_summary = twr_counts.to_frame().merge(forecast_stats, left_index=True, right_index=True, how="left").reset_index()
    
    
    twr_summary["month"] = month
    
    twr_summary.reset_index()
    
    print(twr_summary)    
    
    #twr_summary.to_csv(eval_dir+sub_eval + "/baseline-on-" +  month + "-categorization.csv")
    twr_summary.to_csv(eval_dir+sub_eval + env_params["eval_type"] + "-" + env_params["model_month"] + "-on-" + month + "-" + env_params["eval_model"] + "-categorization.csv")
    
    df_list.append(twr_summary)
    

    


# Combine all months into one DataFrame
df_combined = pd.concat(df_list, ignore_index=True)

# Define colors, markers, and linestyles
palette = sns.color_palette("tab10", 12)  # Up to 12 distinct colors
markers = ['o']
linestyles = ['--']

# Plot
plt.figure(figsize=(12, 6))

# Loop through each month to apply different colors, markers, and linestyles
for i, month in enumerate(months):
    subset = df_combined[df_combined["month"] == month]
    sns.lineplot(
        data=subset,
        x="TWR_Percentage_Range",
        y="count",
        label=month,
        marker=markers[i % len(markers)],  # Cycle through markers
        linestyle=linestyles[i % len(linestyles)],  # Cycle through linestyles
        color=palette[i]  # Unique color per month
    )

plt.xlabel("TWR Percentage Range")
plt.ylabel("Count")
plt.title("Comparison of TWR Score Distribution Across Months")
plt.legend(title="Month", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
#plt.show()




# Plot
plt.figure(figsize=(12, 6))

# Loop through each month to apply different colors, markers, and linestyles
for i, month in enumerate(months):
    subset = df_combined[df_combined["month"] == month]
    sns.lineplot(
        data=subset,
        x="TWR_Percentage_Range",
        y="mean",
        label=month,
        marker=markers[i % len(markers)],  # Cycle through markers
        linestyle=linestyles[i % len(linestyles)],  # Cycle through linestyles
        color=palette[i]  # Unique color per month
    )

plt.xlabel("TWR Percentage Range")
plt.ylabel("Mean FS")
plt.title("Comparison of Mean Forecast Score for Baseline Performance")
plt.legend(title="Month", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()