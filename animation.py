import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
def animate(i, position, position2,val,val2, ax):
    # erase previous plot

    ax.cla()

    # draw point's trajectory
    ax.plot(position[:i + 1, 0], position[:i + 1, 1], linestyle = '-', color = 'blue')

    # draw point's current position
    ax.plot(position[i, 0], position[i, 1], marker = 'o', markerfacecolor = 'blue', markeredgecolor = 'blue')

    ax.plot(position2[:i + 1, 0], position2[:i + 1, 1], linestyle = '-', color = 'red')

    # draw point's current position
    ax.plot(position2[i, 0], position2[i, 1], marker = 'o', markerfacecolor = 'red', markeredgecolor = 'red')
    ax.text(0.15, 0.9, 'i: '+str(i), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    ax.text(0.85, 0.9, 'val: '+str(val[i][0])+"u: "+str(val[i][1]), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    #print(i)
    # fix axes limits
    ax.set_xlim(0, 2000)
    ax.set_ylim(0, 2000)

    

def animation(position, position2, val, val2):
    # position array generation
    
    print(len(position))
    i = len(position)
    while (i>0):
        i-=1
        if(i%40  != 0):
            position = np.delete(position, i, axis=0)
            position2 = np.delete(position2, i, axis=0)
            val = np.delete(val, i, axis=0)
            val2 = np.delete(val2, i, axis=0)
    print(len(position))
    print(len(val))
    print(val)
    #print(position)
    # generate figure and axis
    fig, ax = plt.subplots(figsize = (5, 5))
    N = len(position)-1
    # define the animation
    ani = FuncAnimation(fig = fig, func = animate, fargs=(position, position2, val, val2, ax,), interval = 0.02, frames = N)

    # show the animation
    plt.show()

def show(env):
    blue_pos, blue_val = env.blue_uav.history.getHistory()
    red_pos, red_val = env.red_uav.history.getHistory()
    animation(blue_pos, red_pos, blue_val, red_val)