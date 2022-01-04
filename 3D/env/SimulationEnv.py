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
            spaces.Discrete(3),
            spaces.Discrete(3),
            spaces.Discrete(3),
            spaces.Discrete(3)))

        # Set Observation space
        self.observation_space = spaces.Box(low=0, high=1, shape=(14, 1), dtype=np.float16)

    def get_na(self):
        der_pu = np.array([self.uav1.der_x, self.uav1.der_y, self.uav1.der_z])
        der_pt = np.array([self.uav2.der_x, self.uav2.der_y, self.uav2.der_z])
        pu = np.array(self.uav1.position)
        pt = np.array(self.uav2.position)

        temp1 = (der_pu @ (pt - pu)) / np.linalg.norm(pt - pu)
        temp2 = (der_pt @ (pu - pt)) / np.linalg.norm(pu - pt)

        na = temp1 - temp2

        print("na:", na)
        return na

    def get_nd(self, d_max=3, d_min=50, beta1=1, beta2=-1):
        pu = np.array(self.uav1.position)
        pt = np.array(self.uav2.position)

        temp1 = beta1 * (d_max - np.linalg.norm(pt - pu)) / (d_max - d_min)
        #  temp2 = 1 - np.exp(-((np.linalg.norm(pt-pu) - d_min)**beta2))

        nd = temp1  # * temp2

        print("nd:", nd)
        return nd

    def calculate_advantage(self, w1=0.5, w2=0.5):
        return w1 * self.get_na() + w2 * self.get_nd()

    def _next_observation(self):
        obs = np.zeros((14, 1))

        ###first uav
        obs[0][0] = self.uav1.position[0]
        obs[1][0] = self.uav1.position[1]
        obs[2][0] = self.uav1.position[2]

        obs[3][0] = self.uav1.roll
        obs[4][0] = self.uav1.pitch
        obs[5][0] = self.uav1.yaw

        obs[6][0] = self.uav1.speed

        ###second uav
        obs[7][0] = self.uav2.position[0]
        obs[8][0] = self.uav2.position[1]
        obs[9][0] = self.uav2.position[2]

        obs[10][0] = self.uav2.roll
        obs[11][0] = self.uav2.pitch
        obs[12][0] = self.uav2.yaw

        obs[13][0] = self.uav2.speed
        return obs

    def _take_action(self, action):
        for i, uav in enumerate(self.uav_list):
            uav.roll = uav.roll + (action[i * 3 + 0] - 1) * self.dt * uav.der_roll
            uav.pitch = uav.pitch + (action[i * 3 + 1] - 1) * self.dt * uav.der_pitch
            uav.speed = uav.speed + (action[i * 3 + 2] - 1) * self.dt * uav.der_speed

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

            n = 1 / np.cos(np.radians(uav.roll))
            der_yaw = np.degrees((self.g * np.sqrt(n ** 2 - 1)) / uav.speed)
            uav.yaw += der_yaw * self.dt
            uav.yaw = uav.yaw % 360

            uav.der_x = uav.speed * np.cos(np.radians(uav.yaw)) * np.cos(np.radians(uav.pitch))
            uav.der_y = uav.speed * np.sin(np.radians(uav.yaw)) * np.cos(np.radians(uav.pitch))
            uav.der_z = uav.speed * np.sin(np.radians(uav.pitch))

            uav.position[0] += uav.der_x * self.dt
            uav.position[1] += uav.der_y * self.dt
            uav.position[2] += uav.der_z * self.dt
        # print("uav.roll, uav.pitch, uav.speed:", uav.roll, uav.pitch, uav.speed)
        # print("der_x, der_y, der_z:", der_x, der_y, der_z)
        # print("uav.position:", uav.position)

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        print("self.calculate_advantage():", self.calculate_advantage())

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
            if self.visualization is None:
                self.visualization = SimulationGraph(
                    kwargs.get('title', None))
            if self.current_step is not None:
                self.visualization.render(self.uav1.position, self.uav2.position)

    def close(self):
        if self.visualization != None:
            self.visualization.close()
            self.visualization = None
