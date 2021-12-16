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
        self.uav2 = UAV()
        self.uav_list = [self.uav1, self.uav2]
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
        for i, uav in enumerate(self.uav_list):
            uav.roll = uav.roll + (action[0] - 1) * self.dt * uav.der_roll
            uav.pitch = uav.pitch + (action[1] - 1) * self.dt * uav.der_pitch
            uav.speed = uav.speed + (action[2] - 1) * self.dt * uav.der_speed

            # check for boundaries
            if uav.roll > uav.max_roll:
                uav.roll = uav.max_roll
            elif uav.roll < -uav.max_roll:
                uav.roll = -uav.max_roll

            if uav.pitch > uav.max_pitch:
                uav.pitch = uav.max_pitch
            elif uav.pitch < -uav.max_pitch:
                uav.pitch = -uav.max_pitch

            if uav.speed > uav.max_speed:
                uav.speed = uav.max_speed
            elif uav.speed < uav.min_speed:
                uav.speed = uav.min_speed

            n = 1/np.cos(np.radians(uav.roll))
            der_yaw = np.degrees((self.g * np.sqrt(n**2-1))/uav.speed)
            uav.yaw += der_yaw * self.dt
            uav.yaw = uav.yaw % 360

            der_x = uav.speed * np.cos(np.radians(uav.yaw)) * np.cos(np.radians(uav.pitch))
            der_y = uav.speed * np.sin(np.radians(uav.yaw)) * np.cos(np.radians(uav.pitch))
            der_z = uav.speed * np.sin(np.radians(uav.pitch))

            uav.position[0] += der_x * self.dt
            uav.position[1] += der_y * self.dt
            uav.position[2] += der_z * self.dt
        #print("uav.roll, uav.pitch, uav.speed:", uav.roll, uav.pitch, uav.speed)
        #print("der_x, der_y, der_z:", der_x, der_y, der_z)
        #print("uav.position:", uav.position)

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
                self.visualization.render(self.uav1.position, self.uav2.position)

    def close(self):
        if self.visualization != None:
            self.visualization.close()
            self.visualization = None
