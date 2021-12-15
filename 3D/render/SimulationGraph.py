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
        self.map_ax.set_xlim3d([0.0, 10.0])
        self.map_ax.set_ylim3d([0.0, 10.0])
        self.map_ax.set_zlim3d([0.0, 10.0])

    def gen(self, n):
        phi = 0
        while phi < 2*np.pi:
            yield np.array([np.cos(phi), np.sin(phi), phi])
            phi += 2*np.pi/n

    def update(self,num, data, line):
        line.set_data(data[:2, :num])
        line.set_3d_properties(data[2, :num])


    def render(self, new_data):
        print("plotting")
        self.map_ax.scatter3D([new_data[0]], [new_data[1]], [new_data[2]], color='black')
        plt.show(block=False)
        # Necessary to view frames before they are unrendered
        plt.pause(0.01)

    def close(self):
        plt.close()
