"""
Suite of example reward functions to use within the RLHAB environment
"""
import numpy as np
import math
from utils.common import convert_range
from env.config.env_config import env_params

def reward_google(Balloon, radius, radius_inner, radius_outer, twr_data):
    """
    Google Loon's reward function from the paper `"Autonomous navigation of stratospheric balloons using reinforcement
    learning" <https://www.nature.com/articles/s41586-020-2939-8>`_
    """
    distance_to_target = Balloon.distance
    c_cliff = 0.4 # scale factor
    tau = 100000 # m

    reward = 0
    within_target = False

    if Balloon.altitude >= env_params['alt_min'] and Balloon.altitude <= env_params['alt_max']:

        if distance_to_target <= radius:
            reward = 1
            twr_data["twr"] += 1
            within_target = True
        else:
            #reward = np.exp(-0.01 * (distance_to_target - self.radius))
            reward = c_cliff*2*np.exp((-1*(distance_to_target-radius)/tau))

        #Add more regions to track,  Not doing anything with them yet,  just for metric analysis
        if distance_to_target <= radius_inner:
            twr_data["twr_inner"] += 1

        if distance_to_target <= radius_outer:
            twr_data["twr_outer"] += 1

    else:
        reward = 0

    return reward, within_target

def reward_piecewise(Balloon, radius, radius_inner, radius_outer, twr_data):
    """
    Compute the reward based on the balloon's distance to the target.

    Returns:
        float: Reward value.
    """
    distance_to_target = Balloon.distance
    c_cliff = 0.4 # scale factor
    tau = 100000 # m

    reward = 0
    within_target = False

    if Balloon.altitude >= env_params['alt_min'] and Balloon.altitude <= env_params['alt_max']:

        if distance_to_target <= radius_inner:
            reward = 2
            within_target = True
        elif distance_to_target <= radius and distance_to_target > radius_inner:
            reward = 1
            within_target = True
        else:
            # reward = np.exp(-0.01 * (distance_to_target - self.radius))
            reward = c_cliff * 2 * np.exp((-1 * (distance_to_target - radius) / tau))


        twr_data["twr"] += 1 if within_target else 0

        # Add more regions to track,  Not doing anything with them yet,  just for metric analysis
        if distance_to_target <= radius_inner:
            twr_data["twr_inner"] += 1
        if distance_to_target <= radius_outer:
            twr_data["twr_outer"] += 1

    return reward, within_target

def reward_euclidian(Balloon, radius, radius_inner, radius_outer, twr_data):
    """
    Linear Euclidian reward within target region, google cliff function for outside of radius

    """

    distance_to_target = Balloon.distance
    c_cliff = 0.4 # scale factor
    tau = 100000 # m

    reward = 0
    within_target = False

    if Balloon.altitude >= env_params['alt_min'] and Balloon.altitude <= env_params['alt_max']:

        if distance_to_target <= radius:
            #Normalize distance within radius,  for a maximum score of 2.
            reward = convert_range(distance_to_target,0,radius, 2, 1)
            within_target = True

        else:
            reward = c_cliff * 2 * np.exp((-1 * (distance_to_target - radius) / tau))

        twr_data["twr"] += 1 if within_target else 0

        # Add more regions to track,  Not doing anything with them yet,  just for metric analysis
        if distance_to_target <= radius_inner:
            twr_data["twr_inner"] += 1
        if distance_to_target <= radius_outer:
            twr_data["twr_outer"] += 1

    return reward, within_target
    
def reward_bearing(Balloon, radius, radius_inner, radius_outer, twr_data):
    """
    Bearing based reward. similar to baseline controller
    """

    distance_to_target = Balloon.distance
    reward = convert_range(Balloon.rel_bearing, 0, np.pi, 1, 0)
    within_target = False

    if Balloon.altitude >= env_params['alt_min'] and Balloon.altitude <= env_params['alt_max']:

        if distance_to_target <= radius:
            within_target = True
            twr_data["twr"] += 1

        if distance_to_target <= radius_inner:
            twr_data["twr_inner"] += 1

        if distance_to_target <= radius_outer:
             twr_data["twr_outer"] += 1


        return reward, within_target