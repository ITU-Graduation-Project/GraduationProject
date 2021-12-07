import gym
from gym import spaces
import numpy as np
import time

from render.SimulationGraph import SimulationGraph

MAX_STEPS = 20000


LOOKBACK_WINDOW_SIZE = 40




class SimulationEnv(gym.Env):
    """A 3D flight simulation environment for OpenAI gym"""
    metadata = {'render.modes': ['live', 'file', 'none']}
    visualization = None

    def __init__(self):
        super(SimulationEnv, self).__init__()

        # Set actionSpace
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

        # Set Observation space
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(5, 40 + 2), dtype=np.float16)


    def _next_observation(self):
        frame = np.zeros((5, 40 + 1))

        # Put obs data
        np.put(frame, [0, 4], [
            0, 0, 0, 0, 0
        ])

        # Append additional data if necessary
        obs = np.append(frame, [
            [0], [0], [0], [0], [0],
        ], axis=1)

        return obs

    def _take_action(self, action):

        action_type = action[0]
        amount = action[1]

        if action_type < 1:
            # Buy amount % of balance in shares
            pass
        elif action_type < 2:
            # Sell amount % of shares held
            pass
    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1

        reward = 0
        done = False
        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0

        return self._next_observation()

    def render(self, mode='live', **kwargs):
        # Render the environment to the screen
        if mode == 'live':
            if self.visualization == None:
                self.visualization = SimulationGraph(
                kwargs.get('title', None))
            if self.current_step > 40:
                self.visualization.render()
    def close(self):
        if self.visualization != None:
            self.visualization.close()
            self.visualization = None
