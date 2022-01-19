import gym
from gym import spaces
import numpy as np
from render.SimulationGraph import SimulationGraph
from env.uav import UAV

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
        self.current_reward = [0, 0]  # first is orientation reward, second is distance reward
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
        self.observation_space = spaces.Box(low=0, high=1, shape=(11, 1), dtype=np.float16)

    def get_na(self):
        der_pu = np.array([self.uav1.der_x, self.uav1.der_y, self.uav1.der_z])
        der_pt = np.array([self.uav2.der_x, self.uav2.der_y, self.uav2.der_z])
        pu = np.array(self.uav1.position)
        pt = np.array(self.uav2.position)

        temp1 = (der_pu @ (pt - pu)) / np.linalg.norm(pt - pu)
        temp2 = (der_pt @ (pu - pt)) / np.linalg.norm(pu - pt)

        na = temp1 - temp2

        return na

    def get_nd(self, beta1=8, beta2=0.7, d_max=30, d_min=0):

        distance = np.linalg.norm(np.array(self.uav_list[0].position) - np.array(self.uav_list[1].position))

        f_t = (d_max - abs(distance)) / (d_max - d_min)
        s_t = 1 - np.exp(-(abs(distance) - d_min) ** beta2)
        return_val = beta1 * f_t * s_t
        return return_val

    """def ds(self):
        distance = np.linalg.norm(np.array(self.uav_list[0].position) - np.array(self.uav_list[1].position))
        if distance < 30:
            return_value = distance / 2
        elif 30 < distance < 60:
            return_value = 30 - distance / 2
        else:
            return_value = - np.sqrt(distance - 30) * 4
        return return_value"""

    def calculate_advantage(self, w1=0.5, w2=0.5):
        self.current_reward[0], self.current_reward[1] = self.get_na(), self.get_nd()
        return w1 * self.get_na() + w2 * self.get_nd()

    def _next_observation(self):
        obs = np.zeros(11)

        # first uav
        obs[0] = 1 / (1 + np.exp(-(self.uav1.position[0] - self.uav2.position[0])/30))
        obs[1] = 1 / (1 + np.exp(-(self.uav1.position[1] - self.uav2.position[1])/30))
        obs[2] = 1 / (1 + np.exp(-(self.uav1.position[2] - self.uav2.position[2])/30))

        obs[3] = 1 / (1 + np.exp(-self.uav1.roll/45))
        obs[4] = 1 / (1 + np.exp(-self.uav1.pitch/45))
        obs[5] = 1 / (1 + np.exp(-self.uav1.yaw/90))

        obs[6] = 1 / (1 + np.exp(-self.uav1.speed/5))

        # second uav
        obs[7] = 1 / (1 + np.exp(-self.uav2.roll/45))
        obs[8] = 1 / (1 + np.exp(-self.uav2.pitch/45))
        obs[9] = 1 / (1 + np.exp(-self.uav2.yaw/90))

        obs[10] = 1 / (1 + np.exp(-self.uav2.speed/5))
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

    def step(self, action, episode=None):
        # Execute one time step within the environment
        self._take_action(action)

        # print("self.calculate_advantage():", self.calculate_advantage())

        self.current_step += 1

        if episode is None:
            thresh = 25
        else:
            thresh = 2.5 + (episode / 1000)
            thresh = min(thresh, 15)

        reward = self.calculate_advantage()
        if reward > 8:
            reward *= 1.25
        done = 1 if reward > thresh else 0
        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.uav1 = UAV()
        self.uav2 = UAV()
        self.uav_list = [self.uav1, self.uav2]
        self.current_step = 0

        return self._next_observation()

    def render(self, mode='live', **kwargs):
        # Render the environment to the screen
        if mode == 'live':
            if self.visualization is None:
                self.visualization = SimulationGraph(
                    kwargs.get('title', None))
            if self.current_step is not None:
                self.visualization.render(self.uav1.position, self.uav2.position, self.current_reward)

    def close(self, episode):
        if self.visualization != None:
            self.visualization.close(episode)
            self.visualization = None