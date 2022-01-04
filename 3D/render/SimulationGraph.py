import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

#style.use('dark_background')

VOLUME_CHART_HEIGHT = 1



class SimulationGraph:
    """A simulation visualization using matplotlib made to render OpenAI gym environments"""

    def __init__(self, title=None):
        self.map = plt.figure()
        self.map_ax = Axes3D(self.map)
        self.map_ax.autoscale(enable=True, axis='both', tight=True)

        # # # Setting the axes properties

    def gen(self, n):
        phi = 0
        while phi < 2*np.pi:
            yield np.array([np.cos(phi), np.sin(phi), phi])
            phi += 2*np.pi/n

    def update(self, num, data, line):
        line.set_data(data[:2, :num])
        line.set_3d_properties(data[2, :num])


    def render(self, new_data1, new_data2):

        self.map_ax.scatter3D([new_data1[0]], [new_data1[1]], [new_data1[2]], color='blue')
        self.map_ax.scatter3D([new_data2[0]], [new_data2[1]], [new_data2[2]], color='red')
        plt.show(block=False)
        # Necessary to view frames before they are unrendered
        plt.pause(0.0001)

    def close(self):
        plt.close()
