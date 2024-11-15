from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
import numpy as np

class RogueCallback(BaseCallback):
    """
    Custom tensorboard callback to keep track of trajectories that go rogue
    """
    def __init__(self, moving_avg_length=100, verbose=0):
        super(RogueCallback, self).__init__(verbose)
        #self.env = env  # type: Union[gym.Env, VecEnv, None]
        self.moving_avg_length = moving_avg_length
        self.rogue_history = []
        self.rogue_status = []
        self.rogue_step_trigger_history = []


    def _on_step(self) -> bool:
        # Check if the episode has ended
        done = self.locals['dones'][0]

        if done:
            infos = self.locals['infos'][0]

            #Check how Much of Episode is Rogue

            self.rogue_history.append(infos.get("rogue_count")/infos.get("total_steps"))

            if len(self.rogue_history) > self.moving_avg_length:
                self.rogue_history.pop(0)

            moving_avg = np.mean(self.rogue_history)
            self.logger.record('rogue/rogue_percent', moving_avg)

            #Now keep track of Rogue status among episode (moving average of 100, so this is percent)

            if infos.get("rogue_count") > 0:
                self.rogue_status.append(1)
            else:
                self.rogue_status.append(0)

            if len(self.rogue_status) > self.moving_avg_length:
                self.rogue_status.pop(0)

            moving_avg = np.mean(self.rogue_status)
            self.logger.record('rogue/rogue_status', moving_avg)

            #Now track when it goes rogue
            if infos.get("rogue_count") > 0:
                self.rogue_step_trigger_history.append(infos.get("rogue_step_trigger"))
            else:
                pass #only tracking rogue start time for episodes that go rogue

            if len(self.rogue_step_trigger_history) > self.moving_avg_length:
                self.rogue_step_trigger_history.pop(0)

            moving_avg = np.mean(self.rogue_step_trigger_history)
            self.logger.record('rogue/rogue_step_trigger', moving_avg)


        return True