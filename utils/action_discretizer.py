import numpy as np

class ActionDiscretizer:

    def __init__(self, action_map):

        self.action_map = action_map

    def discretize(self, steering, throttle):

        target = np.asarray([steering, throttle])

        best_idx = 0
        best_dist = float("inf")

        for idx, action in self.action_map.items():

            action = np.array(action)

            dist = np.linalg.norm(target - action)

            if dist < best_dist:

                best_dist = dist
                best_idx = idx

        return best_idx