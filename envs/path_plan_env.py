import gym
from gym import spaces
import numpy as np
from PIL import Image

class PathPlanEnv(gym.Env):
    def __init__(self, file, start, goal):
        img = Image.open(file).convert("L")
        self.background = np.asarray(img, dtype=float)
        self.start = start
        self.goal = goal

        self.current_position = self.start
        self.agent_size = 10
        self.map = self.background.copy()
        self.map_size = np.array(self.map.shape)

        # step_size must be > 2 * agent_size
        step_size = 30
        self._action_to_direction = {
            0: np.array([step_size, 0]),
            1: np.array([0, step_size]),
            2: np.array([-step_size, 0]),
            3: np.array([0, -step_size]),
        }

        self._rewards = {
            "invalid_move": -10,
            "success": 100,
        }

        self.observation_space = spaces.Dict({
            "map": spaces.Box(0, 1, shape=(600, 600), dtype=float),
            "goal": spaces.Box(0, 600, shape=(2, ), dtype=int)
        })

        self.action_space = spaces.Discrete(4)

    def _get_obs(self):
        return { "map": self.map, "goal": self.goal }

    def _get_info(self):
        return {}

    def _get_bounds_slice(self, position, size):
        pos_min = position - size
        pos_max = position + size
        bounds_slice = (slice(pos_min[0], pos_max[0]), slice(pos_min[1], pos_max[1]))
        return bounds_slice

    def _update_map(self, old_position, new_position):
        old_pos_slice = self._get_bounds_slice(old_position, self.agent_size)
        new_pos_slice = self._get_bounds_slice(new_position, self.agent_size)
        self.map[old_pos_slice] = self.background[old_pos_slice]
        self.map[new_pos_slice] = 0

    def reset(self, seed=None, return_info=False, options=None):
        # super().reset()

        self.map = self.background.copy()
        self._update_map(self.current_position, self.start)
        self.current_position = self.start

        observation = self._get_obs()
        info = self._get_info()
        return (observation, info) if return_info else observation

    def step(self, action):
        assert 0 <= action <= 3
        direction = self._action_to_direction[action]
        new_position = self.current_position + direction
        
        clip_pos = new_position.clip([0, 0], self.map_size - 1)
        # print(new_position, clip_pos)
        # print(new_position == clip_pos)
        # print(np.array_equal(new_position, clip_pos))
        if not np.array_equal(new_position, clip_pos):
            return self._get_obs(), self._rewards["invalid_move"], False, self._get_info()
        
        new_pos_slice = self._get_bounds_slice(new_position, self.agent_size)
        if np.any(self.map[new_pos_slice] < 1):
            return self._get_obs(), self._rewards["invalid_move"], False, self._get_info()

        self._update_map(self.current_position, new_position)
        old_distance = np.linalg.norm(self.current_position - self.goal)
        self.current_position = new_position
        new_distance = np.linalg.norm(self.current_position - self.goal)

        observation = self._get_obs()
        info = self._get_info()
        if (
            np.all(self.current_position - self.agent_size <= self.goal) and
            np.all(self.goal <= self.current_position + self.agent_size)
        ):
            reward = self._rewards["success"]
            done = True
        else:
            reward = old_distance - new_distance
            done = False

        return observation, reward, done, info
