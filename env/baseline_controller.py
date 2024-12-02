import time
from env.config.env_config import env_params
from env.forecast_processing.forecast import Forecast, Forecast_Subset
from env.RLHAB_gym_SINGLE import FlowFieldEnv3d_SINGLE
from env.RLHAB_gym_DUAL import FlowFieldEnv3d_DUAL
from utils.initialize_forecast import initialize_forecasts
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def baseline_controller(obs):
    """
    Given the current altitude and a list of relative flow column entries ([altitude, relative angle, speed]),
    this function returns the best altitude to transition to in order to minimize the relative angle and the action
    needed to reach that altitude.

    Args:
    - obs (dict): Observation dictionary that contains the following keys:
        - 'current_altitude' (float): The current altitude.
        - 'flow_field' (list of lists): List of flow field entries [altitude, relative angle, speed].

    Returns:
    - action (int): -1 for down, 0 for stay, 1 for up.
    - best_altitude (float): The altitude to transition to that minimizes the relative angle.
    """
    # Initialize variables to track the best altitude and its corresponding relative angle
    best_altitude = None
    min_relative_angle = float('inf')

    # Loop through the flow column to find the altitude with the smallest relative angle
    for level in obs['flow_field']:
        altitude, relative_angle, speed = level

        # Update if a new minimum relative angle is found
        #print(altitude, relative_angle, min_relative_angle)
        if relative_angle < min_relative_angle:
            min_relative_angle = relative_angle
            best_altitude = altitude

    # Determine the action to take based on the current altitude and the best altitude found
    if obs['altitude'] < best_altitude:
        action = 2  # Go up
    elif obs['altitude'] > best_altitude:
        action = 0  # Go down
    else:
        action = 1  # stay

    # Return the best altitude found
    return best_altitude, action

def baseline_controller_thresholded(obs, angle_threshold=np.radians(10)):
    """
    Given the current altitude and a list of relative flow column entries ([altitude, relative angle, speed]),
    this function returns the closest altitude to transition to within a specified angular threshold,
    and the action needed to reach that altitude. If no altitude is found within the threshold,
    it will select the next best altitude outside the threshold.

    Args:
    - obs (dict): Observation dictionary that contains the following keys:
        - 'altitude' (float): The current altitude.
        - 'flow_field' (list of lists): List of flow field entries [altitude, relative angle, speed].
    - angle_threshold (float): The maximum allowed relative angle for a candidate altitude.

    Returns:
    - action (int): -1 for down, 0 for stay, 1 for up.
    - best_altitude (float): The altitude to transition to.
    """

    # Extract flow field data
    flow_field = obs['flow_field']

    # Sort the flow field by relative angle (ascending order)
    sorted_flow_field = sorted(flow_field, key=lambda x: abs(x[1]))  # Sorting by absolute value of relative angle

    # Filter altitudes within the angular threshold
    within_threshold = [level for level in sorted_flow_field if abs(level[1]) <= angle_threshold]



    # If there are altitudes within the threshold, choose the one closest to the current altitude
    if within_threshold:
        best_altitude = min(within_threshold, key=lambda x: abs(x[0] - obs['altitude']))[0]
    else:
        # If no altitudes are within the threshold, choose the altitude with the lowest relative angle
        best_altitude = min(sorted_flow_field, key=lambda x: abs(x[1]))[0]

    #print(obs['altitude'], best_altitude, within_threshold)

    # Determine the action to take based on the current altitude and the best altitude found
    if obs['altitude'] < best_altitude:
        action = 2  # Go up
    elif obs['altitude'] > best_altitude:
        action = 0  # Go down
    else:
        action = 1  # Stay

    # Return the best altitude found and the action to take
    return best_altitude, action


import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_mean_scores(csv_dir, x_values):
    # Ensure the number of x_values matches the number of CSV files
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
    if len(csv_files) != len(x_values):
        raise ValueError("The length of x_values must match the number of CSV files in the directory.")

    # List to store the mean scores for each CSV file
    mean_scores = {'TWR_Score': [], 'TWR_Inner_Score': [], 'TWR_Outer_Score': []}

    # Iterate through the CSV files in the directory
    for filename in csv_files:
        # Read the CSV file
        file_path = os.path.join(csv_dir, filename)
        df = pd.read_csv(file_path)

        # Calculate the mean of the relevant columns
        mean_scores['TWR_Score'].append(df['TWR_Score'].mean())
        mean_scores['TWR_Inner_Score'].append(df['TWR_Inner_Score'].mean())
        mean_scores['TWR_Outer_Score'].append(df['TWR_Outer_Score'].mean())

    # Create a DataFrame for the mean scores
    mean_df = pd.DataFrame(mean_scores)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, mean_df['TWR_Score'], label='Mean TWR Score', marker='o')
    plt.plot(x_values, mean_df['TWR_Inner_Score'], label='Mean TWR Inner Score', marker='o')
    plt.plot(x_values, mean_df['TWR_Outer_Score'], label='Mean TWR Outer Score', marker='o')

    plt.title("Mean TWR Scores for Varying Angle Thresholds")
    plt.xlabel("Angle Thresholds")
    plt.ylabel("Mean TWR Score")
    plt.legend()
    plt.grid(True)
    plt.show()


# Example usage:
# x_values = [1, 2, 3, 4, 5, ..., 10]  # Custom x-axis values
# plot_mean_scores("/path/to/your/csv_directory", x_values)


def main(angle, eval_dir, sub_eval):

    #make sub directory if doesnt exist
    if not os.path.exists(eval_dir+sub_eval):
        os.makedirs(eval_dir+sub_eval)

    # Keep track of overall evaluation variables for creating heatmaps
    twr_score = []
    twr_inner_score = []
    twr_outer_score = []
    reward_score = []
    forecast_score = []

    rogue = []
    rogue_percent = []

    # Import Forecasts
    FORECAST_SYNTH, FORECAST_ERA5, forecast_subset_era5, forecast_subset_synth = initialize_forecasts()

    #env = FlowFieldEnv3d_DUAL(FORECAST_ERA5=FORECAST_ERA5, FORECAST_SYNTH=FORECAST_SYNTH, render_mode=None)
    env = FlowFieldEnv3d_SINGLE(FORECAST_PRIMARY = FORECAST_ERA5, render_mode=None)


    NUM_EPS = 2_000  # Number of episodes to evaulate on

    for i in range(0, NUM_EPS):
        total_steps = 0
        rogue_status = 0
        rogue_cumulative = 0

        obs, info = env.reset()
        total_reward = 0
        for step in range( env_params["episode_length"]):

            best_altitude, action = baseline_controller_thresholded(obs, np.radians(angle))

            # Use this for keyboard input
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            total_steps += 1

            if env.render_mode == "human":
                env.render()

            if info["distance"] > env_params["rel_dist"]:
                rogue_status = 1
                rogue_cumulative += 1

            if done:
                break

        # Update scores arrays
        print()
        print("COUNT:", i)
        score = info["forecast_score"]
        forecast_score.append(score)
        print("Forecast Score", score)
        twr_score.append(info["twr"])
        twr_inner_score.append(info["twr_inner"])
        twr_outer_score.append(info["twr_outer"])
        reward_score.append(total_reward)
        twr_rounded = round(int(info["twr"] / 1200. * 100), -1)

        rogue.append(rogue_status)
        rogue_percent.append(rogue_cumulative / (total_steps * 1.))

        print("episode length", total_steps, "Total Reward", total_reward, "TWR", info["twr"], "Rogue", rogue_status,
              "Rogue Percent", rogue_cumulative / (total_steps * 1.))

    # Make Dataframe with overall scores
    df = pd.DataFrame({'Forecast_Score': forecast_score,
                       'TWR_Inner_Score': twr_inner_score,
                       'TWR_Score': twr_score,
                       'TWR_Outer_Score': twr_outer_score,
                       'Total_Reward': reward_score,
                       'rogue': rogue,
                       'rogue_status': rogue_percent},)

    df.to_csv(eval_dir + sub_eval + "/SINGLE_SYNTH-baseline-on-Dec-USA-rogue.csv")
    #df.to_csv(eval_dir + sub_eval + "/SINGLE_ERA5-baseline-on-Jan-USA-rogue.csv")
    print(df)

if __name__ == '__main__':
    angles = [0, 5, 10, 15, 20, 30, 40]
    eval_dir = "evaluation/EVALUATION_DATA/"
    sub_eval = "baseline_SINGLE_SYNTH_rogue_seed2"

    #for angle in angles:
    #    main(angle, eval_dir, sub_eval)

    main(0, eval_dir, sub_eval)

    #plot_mean_scores(eval_dir + sub_eval + "/", angles)