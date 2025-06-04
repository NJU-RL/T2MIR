# import warnings
# warnings.filterwarnings('ignore')
import numpy as np
from gymnasium import spaces, Env
from typing import Optional


class PointEnv(Env):
    """
    point robot on a 2-D plane with position control
    tasks (aka goals) are positions on the plane
     - tasks sampled from unit square
     - reward is L2 distance
    """

    def __init__(
            self,
            tasks: Optional[np.ndarray] = None,
            max_episode_steps=20,
            n_tasks=50
        ):
        super().__init__()
        self._max_episode_steps = max_episode_steps
        self.step_count = 0
        self.n_tasks = n_tasks
        if tasks is None:
            self.goals = np.array([[np.random.uniform(-1., 1.), np.random.uniform(-1., 1.)] for _ in range(n_tasks)]).astype(np.float32)
        else:
            self.n_tasks = tasks.shape[0]
            self.goals = tasks

        self.reset_task(0)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        self.action_space = spaces.Box(low=-1., high=1., shape=(2,))

    def reset_task(self, idx):
        ''' reset goal AND reset the agent '''
        if idx is not None:
            self._goal = np.array(self.goals[idx], dtype=np.float32)
        self.reset()

    def print_task(self):
        print(f'Task information: Goal position {self._goal}')

    def set_goal(self, goal):
        self._goal = np.asarray(goal, dtype=np.float32)

    def load_all_tasks(self, goals: np.ndarray):
        self.goals = goals.copy()
        self.reset_task(0)

    def reset_model(self):
        self._state = self.np_random.uniform(-0.1, 0.1, size=(2,)).astype(np.float32)
        # self._state = np.zeros(2, dtype=np.float32)
        return self._get_obs()

    def reset(self, **kwargs):
        self.step_count = 0
        super().reset(**kwargs)
        return self.reset_model(), {}

    def _get_obs(self):
        return np.copy(self._state)

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high) * 0.1

        self._state = self._state + action
        x = self._state[0] - self._goal[0]
        y = self._state[1] - self._goal[1]
        reward = - (x ** 2 + y ** 2) ** 0.5
        # done = (abs(x) < 0.01) and (abs(y) < 0.01)
        ob = self._get_obs()
        return ob, reward, False, False, dict()

    def reward(self, state, action=None):
        return - ((state[0] - self._goal[0]) ** 2 + (state[1] - self._goal[1]) ** 2) ** 0.5

    def render(self):
        print('current state:', self._state)
        
    def seed(self, seed: int):
        pass