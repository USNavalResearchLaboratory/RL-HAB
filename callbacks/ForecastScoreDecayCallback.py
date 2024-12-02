from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from env.config.env_config import env_params

class ForecastScoreDecayCallback(BaseCallback):
    def __init__(self, initial_percent, final_percent, decay_rate, total_timesteps, verbose=0):
        """
        :param initial_percent: Initial percentage (0 to 1) of the parameter value.
        :param final_percent: Final percentage (0 to 1) of the parameter value.
        :param decay_rate: Linear decay rate (i.e., how many episodes for the full decay).
        :param total_episodes: Total number of episodes over which to decay the parameter.
        :param verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.initial_percent = initial_percent
        self.final_percent = final_percent
        self.decay_rate = decay_rate
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        """
        This is called at the end of each episode.
        We update the parameter value based on the number of episodes.
        """

        # Check if the episode has ended
        done = self.locals['dones'][0]

        if done:
            # Calculate the decayed value based on the current episode
            #current_episode = self.num_timesteps // self.model.env.spec.max_episode_steps
            decay_fraction = min(self.num_timesteps / self.total_timesteps, 1.0)

            # Compute the linear decay (interpolating between initial_percent and final_percent)
            decayed_value = self.initial_percent - (self.initial_percent - self.final_percent) * decay_fraction

            # Optionally print the decayed value for debugging
            if self.verbose > 0:
                print(f"Episode {self.num_timesteps}/{self.total_timesteps}: Decayed Value = {decayed_value}")

            # Set the decayed value in the model (e.g., for exploration, entropy coefficient, etc.)
            # For example, we can set an exploration rate or entropy coefficient
            self.model.ent_coef = decayed_value  # Example of updating a parameter

            env_params['forecast_score_threshold'] = decayed_value

            self.logger.record('custom/forecast_score_threshold', decayed_value)

        return True