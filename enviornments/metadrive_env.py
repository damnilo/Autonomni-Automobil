from metadrive import MetaDriveEnv
from enviornments.action_mapper import ActionMapper
from enviornments.observation_builder import ObservationBuilder
from enviornments.reward_function import RewardFunction

class MetaDriveEnvWrapper:

    def __init__(self, env_config):

        self.env = MetaDriveEnv(env_config)

        self.action_mapper = ActionMapper()

        self.observation_builder = ObservationBuilder()

        self.reward_function = RewardFunction()

        self.obs_size: int = None

    def reset(self):

        raw_obs, info = self.env.reset()

        processed_obs = self.observation_builder.build(
            self.env, raw_obs, info
        )

        if self.obs_size is None:
            self.obs_size = len(processed_obs)

        return processed_obs, info

    def step(self, discrete_action):

        continuous_action = self.action_mapper.map(
            discrete_action
        )

        raw_obs, _, terminated, truncated, info = self.env.step(
            continuous_action
        )

        processed_obs = self.observation_builder.build(
            self.env, raw_obs, info
        )

        reward = self.reward_function.compute(info)

        return processed_obs, reward, terminated, truncated, info
    
    def close(self):

        self.env.close()

    def num_actions(self) -> int:
        return self.action_mapper.num_actions()