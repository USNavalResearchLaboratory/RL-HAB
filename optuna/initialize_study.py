"""Simple script for initializing an optuna hyperparameter study.  Run this script after editing the variables in optuna_config.py"""

import optuna
import optuna_config

study = optuna.create_study(storage=optuna_config.storage,
                            study_name=optuna_config.project_name,
                            direction='maximize',
                            sampler=optuna.samplers.TPESampler(),
                            pruner=optuna.pruners.MedianPruner())