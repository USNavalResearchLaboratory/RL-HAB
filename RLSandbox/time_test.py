from line_profiler import LineProfiler
import math
import numpy as np


from era5.era5_gym import FlowFieldEnv3d
from era5.forecast import Forecast

def test_function():
    env = FlowFieldEnv3d()
    env.FlowField3D.create_netcdf()


if __name__ == "__main__":




    env = FlowFieldEnv3d()
    obs = env.reset()

    action = 1
    done = False
    #obs, reward, done, truncated, info = env.step(action=1)

    # lp = LineProfiler()
    # lp_wrapper = lp(env.calculate_relative_angle)
    # lp_wrapper(env.state["x"], env.state["y"], env.goal["x"], env.goal["y"], env.state["x_vel"], env.state["y_vel"])
    # lp.print_stats()

    lp = LineProfiler(Forecast(env_params['rel_dist'], env_params['pres_min'], env_params['pres_max']))
    lp_wrapper = lp(env.FlowField3D.lookup)
    lp_wrapper()
    lp.print_stats()
