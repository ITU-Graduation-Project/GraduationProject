from env.SimulationEnv import SimulationEnv


env = SimulationEnv()

obs = env.reset()
i = 0
while True:
    action = env.action_space.sample()
    _, rewards, done, info = env.step(action)
    env.render(title="3D Environment")




