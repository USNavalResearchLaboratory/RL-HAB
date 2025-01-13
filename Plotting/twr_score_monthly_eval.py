"""
Line plots of of mean twr with error bars for different models in each month.

Can update the forecast cutoff_score for filtering.

To run this script, evaluation has to have already been run on all 12 months individually and stored in CSVs.  See evaluation/EVALUATION_DATA as an example.

"""


import pandas as pd
import matplotlib.pyplot as plt

eval_months = [ ("Jan" , "dainty-water"), ("Apr" , "effortless-blaze"),("Jul" , "hopeful-pyramid"), ("Sep", "pretty-cosmos"), ("Jul_cust", "silvery-jazz")]
#eval_months = [ ("Apr" , "effortless-blaze"),("Jul" , "hopeful-pyramid"), ("Sep", "pretty-cosmos")]
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
colors = [ "blue", "green", "orange", "red", "black", "purple", "pink", "cyan"]

# Create a DataFrame to store mean TWR_Score for each evaluation month
mean_twr_df = pd.DataFrame(index=months)
std_twr_df = pd.DataFrame(index=months)

forecast_cutoff_score = 0.75

eval_dir = "evaluation/EVALUATION_DATA/"

for e in eval_months:
    # List to hold mean TWR_Score for the current evaluation month
    mean_twr_scores = []
    std_twr_scores = []

    print(e)


    for month in months:
        csv_name = eval_dir + "DUAL-" +e[0] +"-on-" + month +"-USA-" + e[1] +"-piecewise.csv"
        print(csv_name)

        try:
            data = pd.read_csv(csv_name)
            # Filter out TWR_Score values below 0.5
            data = data[data['Forecast_Score'] >= forecast_cutoff_score]

            # Calculate mean TWR_Score
            mean_twr = data['TWR_Score'].mean()
            std_twr = data['TWR_Score'].std()
            mean_twr_scores.append(mean_twr)
            std_twr_scores.append(std_twr)
        except FileNotFoundError:
            # If the file does not exist, append NaN
            mean_twr_scores.append(float('nan'))
            std_twr_scores.append(float('nan'))

    # Add the mean TWR_Scores for this evaluation month to the DataFrame
    mean_twr_df[e[0]] = mean_twr_scores
    std_twr_df[e[0]] = std_twr_scores


#ADD Baseline***********************


eval_dir = "evaluation/EVALUATION_DATA/"
sub_eval = "baseline_DUAL_rogue_seed2"

# List to hold mean TWR_Score for the current evaluation month
mean_twr_scores = []
std_twr_scores = []


e = ["baseline_DUAL"]
for month in months:
    csv_name = eval_dir + sub_eval + "/DUAL-baseline-on-" + month +"-USA-rogue.csv"
    print(csv_name)

    try:
        data = pd.read_csv(csv_name)
        # Filter out TWR_Score values below 0.5
        data = data[data['Forecast_Score'] >= forecast_cutoff_score]

        # Calculate mean TWR_Score
        mean_twr = data['TWR_Score'].mean()
        std_twr = data['TWR_Score'].std()
        mean_twr_scores.append(mean_twr)
        std_twr_scores.append(std_twr)
    except FileNotFoundError:
        # If the file does not exist, append NaN
        mean_twr_scores.append(float('nan'))
        std_twr_scores.append(float('nan'))


mean_twr_df[e[0]] = mean_twr_scores
std_twr_df[e[0]] = std_twr_scores



#ADD Baseline***********************

eval_dir = "evaluation/EVALUATION_DATA/"
sub_eval = "baseline_SINGLE_ERA5_rogue_seed2"

# List to hold mean TWR_Score for the current evaluation month
mean_twr_scores = []
std_twr_scores = []

e = ["baseline_SINGLE_ERA5"]
for month in months:
    csv_name = eval_dir + sub_eval + "/SINGLE_ERA5-baseline" +"-on-" + month +"-USA-rogue.csv"
    print(csv_name)

    try:
        data = pd.read_csv(csv_name)
        # Filter out TWR_Score values below 0.5
        data = data[data['Forecast_Score'] >= forecast_cutoff_score]

        # Calculate mean TWR_Score
        mean_twr = data['TWR_Score'].mean()
        std_twr = data['TWR_Score'].std()
        mean_twr_scores.append(mean_twr)
        std_twr_scores.append(std_twr)
    except FileNotFoundError:
        # If the file does not exist, append NaN
        mean_twr_scores.append(float('nan'))
        std_twr_scores.append(float('nan'))

# Add the mean TWR_Scores for this evaluation month to the DataFrame
#e = "baseline_SINGLE_ERA5"
mean_twr_df[e[0]] = mean_twr_scores
std_twr_df[e[0]] = std_twr_scores

#*******************************

#ADD Baseline***********************

eval_dir = "evaluation/EVALUATION_DATA/"
sub_eval = "baseline_SINGLE_SYNTH_rogue_seed2-UPDATED"

# List to hold mean TWR_Score for the current evaluation month
mean_twr_scores = []
std_twr_scores = []

e = ["baseline_SINGLE_SYNTH"]
for month in months:
    csv_name = eval_dir + sub_eval + "/SINGLE_SYNTH-baseline" +"-on-" + month +"-USA-rogue.csv"
    print(csv_name)

    try:
        data = pd.read_csv(csv_name)
        # Filter out TWR_Score values below 0.5
        data = data[data['Forecast_Score'] >= forecast_cutoff_score]

        # Calculate mean TWR_Score
        mean_twr = data['TWR_Score'].mean()
        std_twr = data['TWR_Score'].std()
        mean_twr_scores.append(mean_twr)
        std_twr_scores.append(std_twr)
    except FileNotFoundError:
        # If the file does not exist, append NaN
        mean_twr_scores.append(float('nan'))
        std_twr_scores.append(float('nan'))

# Add the mean TWR_Scores for this evaluation month to the DataFrame
#e = "baseline_SINGLE_ERA5"
mean_twr_df[e[0]] = mean_twr_scores
std_twr_df[e[0]] = std_twr_scores

#*******************************






# Plotting
plt.figure(figsize=(12, 6))
i=0
for eval_month in mean_twr_df.columns:
    plt.errorbar(mean_twr_df.index, mean_twr_df[eval_month],
                 yerr=std_twr_df[eval_month], marker='o', alpha=0.75, linestyle = '-', label=eval_month, capsize=5, capthick=1, color = colors[i]) #, color = "black")
    i+=1
plt.title('Mean TWR Score by Month (filtered FS >' + str(forecast_cutoff_score) + ")")
plt.xlabel('Month')
plt.ylabel('Mean TWR Score')
plt.xticks(rotation=45)
plt.legend(title='Evaluation Months')
plt.grid()
plt.tight_layout()
plt.show()