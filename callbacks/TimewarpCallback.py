from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from env.config.env_config import env_params

class TimewarpCallback(BaseCallback):
    """
    Custom tensorboard callback to keep track of the mean reward.  Tracks the moving average of the window size.
    """
    def __init__(self, verbose=0):
        super(TimewarpCallback, self).__init__(verbose)
        #self.env = env  # type: Union[gym.Env, VecEnv, None]

    def _on_step(self) -> bool:
        # Check if the episode has ended
        done = self.locals['dones'][0]
        if done:
            self.logger.record('custom/timewarp', env_params['timewarp'])

        return True