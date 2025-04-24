.. _env_directory:

=====================================
Environment Module (`env` Directory)
=====================================

The `env` directory contains the core components for simulating, processing, and visualizing high-altitude balloon dynamics in a 3D flow field environment. This module is built around a custom Gym environment (`FlowFieldEnv3d_DUAL`) and several supporting classes for forecast processing, balloon state tracking, and visualization.

Directory Structure
===================

- **`env_config.py`**
  Defines environment parameters used throughout the simulation (e.g., `env_params`).

- **`FlowFieldEnv3d_DUAL`**
  The main custom Gym environment for simulating high-altitude balloon station-keeping in dynamic flow fields.

- **`balloon.py`**
  Contains classes for tracking balloon state and simulation state.

- **`forecast_processing/`**
  Subdirectory for handling weather forecasts and related processing:
  - `forecast.py`: Handles loading and subsetting ERA5 and synthetic forecasts.
  - `forecast_visualizer.py`: Generates visualizations for forecast data.
  - `forecast_classifier.py`: Classifies forecasts based on opposing wind criteria.

- **`rendering/`**
  Subdirectory for visualization tools:
  - `Trajectory3DPlotter.py`: Visualizes 3D balloon trajectories.
  - `renderertriple.py`: Handles combined visualization of trajectory and forecasts.

Core Components
================

FlowFieldEnv3d_DUAL
--------------------

This is the central Gym environment that defines the high-altitude balloon simulation. It models a 3D flow field where the balloon adjusts altitude to navigate within dynamic wind patterns.

**Key Features:**
- Customizable reward functions for station-keeping.
- Realistic wind data integration from ERA5 and synthetic forecasts.
- Comprehensive observation and action spaces.

**References:**
- `env_config.env_params` for configuration.
- `balloon.BalloonState` for balloon position, velocity, and altitude tracking.
- `forecast_processing.forecast.Forecast_Subset` for forecast integration.

Balloon State Tracking
-----------------------

The `balloon.py` file includes:
- **`BalloonState`**: Tracks the balloon's position, velocity, altitude, and related metrics.
- **`SimulatorState`**: Manages simulation-wide variables such as timestamps and trajectory history.

Forecast Processing
--------------------

1. **`forecast.py`**:
   - Loads ERA5 and synthetic forecasts.
   - Subsets forecasts to relevant time and spatial regions for efficient simulation.

2. **`forecast_visualizer.py`**:
   - Generates 3D quiver plots to visualize wind flow.
   - Supports visualizing wind direction and speed at multiple altitude levels.

3. **`forecast_classifier.py`**:
   - Classifies forecasts based on opposing wind sectors.
   - Computes scores for selecting suitable forecasts for simulation.

Rendering Tools
----------------

1. **`Trajectory3DPlotter`**:
   - Visualizes the balloon's trajectory in 3D space.
   - Includes altitude plots and station-keeping zones.

2. **`MatplotlibRendererTriple`**:
   - Combines trajectory and forecast visualizations.
   - Provides dynamic updates during simulation.

Workflow
========

1. **Initialization**:
   - Forecast data is loaded and processed into subsets.
   - The Gym environment initializes with a goal and simulation parameters.

2. **Simulation**:
   - The balloon adjusts its altitude based on the flow field to achieve station-keeping.
   - Reward functions evaluate performance.

3. **Visualization**:
   - Forecast and trajectory visualizations provide real-time feedback.
   - Metrics such as time within radius are tracked.

Example Usage
=============

.. code-block:: python

    from env.FlowFieldEnv3d_DUAL import FlowFieldEnv3d_DUAL
    from env.forecast_processing.forecast import Forecast, Forecast_Subset
    from env.rendering.renderertriple import MatplotlibRendererTriple

    # Load forecasts
    era5_forecast = Forecast("era5.nc", forecast_type="ERA5")
    synth_forecast = Forecast("synth.nc", forecast_type="SYNTH")

    # Initialize the environment
    env = FlowFieldEnv3d_DUAL(
        FORECAST_ERA5=era5_forecast,
        FORECAST_SYNTH=synth_forecast,
        render_mode="human"
    )

    # Simulate an episode
    obs, _ = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        if done:
            break

Integration and Customization
=============================

- **Parameterization**:
  Adjust parameters in `env_config.env_params` for custom simulation settings.

- **Reward Functions**:
  Modify reward functions in `FlowFieldEnv3d_DUAL` to optimize for different objectives.

- **Visualization**:
  Customize rendering frequency and quiver density via `MatplotlibRendererTriple`.

This directory provides a robust framework for simulating and visualizing high-altitude balloon dynamics, integrating real-world wind data with flexible tools for analysis and development.

