import gym
from gym import spaces
import numpy as np
import random
from render.SimulationGraph import SimulationGraph
from env.uav import UAV

MAX_STEPS = 20000

LOOKBACK_WINDOW_SIZE = 40


class SimulationEnv(gym.Env):
    """A 3D flight simulation environment for OpenAI gym"""
    metadata = {'render.modes': ['live', 'file', 'none']}
    visualization = None

    def __init__(self):
        super(SimulationEnv, self).__init__()
        self.uav1 = UAV()
        self.dt = 0.05  # each step time
        self.g = 9.81
        # Set actionSpace
        # first variable is roll change: 0 -> left; 1 -> steady; 2 -> right
        # second variable is pitch change: 0 -> down; 1 -> steady; 2 -> up
        # first variable is speed change: 0 -> decelerate; 1 -> steady; 2 -> accelerate
        self.action_space = spaces.Tuple((
            spaces.Discrete(3),
            spaces.Discrete(3),
            spaces.Discrete(3)))

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
        self.uav1.roll = self.uav1.roll + (action[0] - 1) * self.dt * self.uav1.der_roll
        self.uav1.pitch = self.uav1.pitch + (action[1] - 1) * self.dt * self.uav1.der_pitch
        self.uav1.speed = self.uav1.speed + (action[2] - 1) * self.dt * self.uav1.der_speed

        # check for boundaries
        if self.uav1.roll > self.uav1.max_roll:
            self.uav1.roll = self.uav1.max_roll
        elif self.uav1.roll < -self.uav1.max_roll:
            self.uav1.roll = -self.uav1.max_roll

        if self.uav1.pitch > self.uav1.max_pitch:
            self.uav1.pitch = self.uav1.max_pitch
        elif self.uav1.pitch < -self.uav1.max_pitch:
            self.uav1.pitch = -self.uav1.max_pitch

        if self.uav1.speed > self.uav1.max_speed:
            self.uav1.speed = self.uav1.max_speed
        elif self.uav1.speed < self.uav1.min_speed:
            self.uav1.speed = self.uav1.min_speed

        n = 1/np.cos(np.radians(self.uav1.roll))
        der_yaw = np.degrees((self.g * np.sqrt(n**2-1))/self.uav1.speed)
        self.uav1.yaw += der_yaw * self.dt
        self.uav1.yaw = self.uav1.yaw % 360

        der_x = self.uav1.speed * np.cos(np.radians(self.uav1.yaw)) * np.cos(np.radians(self.uav1.pitch))
        der_y = self.uav1.speed * np.sin(np.radians(self.uav1.yaw)) * np.cos(np.radians(self.uav1.pitch))
        der_z = self.uav1.speed * np.sin(np.radians(self.uav1.pitch))

        self.uav1.position[0] += der_x * self.dt
        self.uav1.position[1] += der_y * self.dt
        self.uav1.position[2] += der_z * self.dt
        #print("self.uav1.roll, self.uav1.pitch, self.uav1.speed:", self.uav1.roll, self.uav1.pitch, self.uav1.speed)
        #print("der_x, der_y, der_z:", der_x, der_y, der_z)
        #print("self.uav1.position:", self.uav1.position)

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
            if self.current_step is not None:
                self.visualization.render(self.uav1.position)

    def close(self):
        if self.visualization != None:
            self.visualization.close()
            self.visualization = None
