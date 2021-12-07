

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation


style.use('dark_background')

VOLUME_CHART_HEIGHT = 1



class SimulationGraph:
    """A simulation visualization using matplotlib made to render OpenAI gym environments"""

    def __init__(self, title=None):

        # Create a figure on screen and set the title
        fig = plt.figure(figsize=(15, 15), dpi=80)
        fig.suptitle(title)

        self.ax = fig.add_subplot(111, projection='3d')

        N = 100
        data = np.array(list(self.gen(N))).T
        line, = self.ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1])

        # Setting the axes properties
        self.ax.set_xlim3d([-1.0, 1.0])
        self.ax.set_xlabel('X')

        self.ax.set_ylim3d([-1.0, 1.0])
        self.ax.set_ylabel('Y')

        self.ax.set_zlim3d([0.0, 10.0])
        self.ax.set_zlabel('Z')

        self.ani = animation.FuncAnimation(fig, self.update, N, fargs=(data, line), interval=10000/N, blit=False)

        # Add padding to make graph easier to view
        plt.subplots_adjust(left=0.11, bottom=0.24,
                            right=0.90, top=0.90, wspace=0.2, hspace=0)

        # Show the graph without blocking the rest of the program
        plt.show(block=False)

    def gen(self,n):
        phi = 0
        while phi < 2*np.pi:
            yield np.array([np.cos(phi), np.sin(phi), phi])
            phi += 2*np.pi/n

    def update(self,num, data, line):
        line.set_data(data[:2, :num])
        line.set_3d_properties(data[2, :num])


    def render(self):

        # Necessary to view frames before they are unrendered
        plt.pause(0.001)

    def close(self):
        plt.close()
