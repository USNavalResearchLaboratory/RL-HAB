.. toctree::
   :maxdepth: 1
   :caption: Contents:
   :hidden:
   
   installation
   env
   getting_started
   API/modules
   changelog
   citing


RLHAB
=========

RL-HAB is an open-source high altitude balloon (HAB) reinforcement learning simulation environment for training automous HAB agents. HABs can leverage opposing winds to perform station keeping maneuvers for persistent area coverage of a target region over a time period of hours, days, or weeks, which can help with surveillance, in-situ stratospheric meteorologicaldata collection, or communication relays.  With perfect weather forecasts, this would be a straight forward deterministic path planning problem; Unfortunately forecasts frequently have large errors in wind direction (occasionally up to 180 degrees) and also lack vertical and temporal resolution in the altitude region of interest (typically only 5-10 data points for a 10 km region), leading to significant
uncertainty in flow fields.

We provide examples of training and evaluating agents with
DQN in stable-baselines-3.  This package also include optional integration of wandb and optuna for automated hyperparameter tuning and analysis.



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`