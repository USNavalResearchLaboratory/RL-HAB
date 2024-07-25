from line_profiler import LineProfiler
import math
import numpy as np


from FlowEnv3D_SK_relative import FlowFieldEnv3d


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

    lp = LineProfiler()
    lp_wrapper = lp(env.FlowField3D.lookup)
    lp_wrapper()
    lp.print_stats()
