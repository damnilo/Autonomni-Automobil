class Evaluator:

    def __init__(self, env, agent, logger):

        self.env = env
        self.agent = agent
        self.logger = logger

    def evaluate(self, num_episodes):

        for episode in range(num_episodes):

            state, _ = self.env.reset()

            done = False

            total_reward = 0

            info = {}

            while not done:

                action = self.agent.select_action(state, step=0, training=True)

                state, reward, terminated, truncated, info = self.env.step(action)

                done = terminated or truncated

                total_reward += reward

            self.logger.log_episode_result(
                episode = episode,
                success = info.get("arrive_dest", False),
                collision = info.get("crash", False),
                out_of_road = info.get("out_of_road", False),
                reward = total_reward
            )