from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback


class FlowChangeCallback(BaseCallback):
    """
    Custom tensorboard callback to keep track of the mean reward.  Tracks the moving average of the window size.
    """
    def __init__(self, verbose=0):
        super(FlowChangeCallback, self).__init__(verbose)
        #self.env = env  # type: Union[gym.Env, VecEnv, None]

    def _on_step(self) -> bool:
        # Check if the episode has ended
        done = self.locals['dones'][0]

        if done:
            infos = self.locals['infos'][0]

            self.logger.record('num_flow_changes', infos.get("num_flow_changes"))

        return True