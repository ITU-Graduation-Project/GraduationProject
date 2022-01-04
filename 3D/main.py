from env.SimulationEnv import SimulationEnv
from pynput import keyboard


env = SimulationEnv()

obs = env.reset()
i = 0
a = b = c = 0

while True:
    """action = [1, 1, 1]
    with keyboard.Events() as events:
        event = events.get(1e6)
        if event.key == keyboard.KeyCode.from_char('s'):
            action[1] = 0
        if event.key == keyboard.KeyCode.from_char('w'):
            action[1] = 2
        if event.key == keyboard.KeyCode.from_char('d'):
            action[0] = 2
        if event.key == keyboard.KeyCode.from_char('a'):
            action[0] = 0"""
    input()
    action = env.action_space.sample()
    #print(action)
    #action = [1, 0, 1]
    #print("action:", action)
    _, rewards, done, info = env.step(action)
    if(i%1 == 0):
        env.render(title="3D Environment")
        i = 1
    i += 1
    #print("i:", i)
    if action[1] == 0:
        a += 1
    if action[1] == 1:
        b += 1
    if action[1] == 2:
        c += 1
    #print("a:", a)
    #print("b:", b)
    #print("c:", c)



