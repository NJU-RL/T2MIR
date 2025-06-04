import numpy as np


class OptimalAgent:
    def act(self, state, env):
        goal = env._goal_pos
        if np.all(goal == state):
            return 0

        # first up or down
        if goal[0] > state[0]:
            return 3
        if goal[0] < state[0]:
            return 1

        # then left or right
        if goal[1] > state[1]:
            return 2
        if goal[1] < state[1]:
            return 4


class OptimalAgentK2D:
    def act(self, state, env):
        if env.has_key:
            goal = env.door_pos
        else:
            goal = env.key_pos

        if np.all(goal == state):
            return 0

        # first up or down
        if goal[0] > state[0]:
            return 3
        if goal[0] < state[0]:
            return 1

        # then left or right
        if goal[1] > state[1]:
            return 2
        if goal[1] < state[1]:
            return 4
