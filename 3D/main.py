from env.SimulationEnv import SimulationEnv


env = SimulationEnv()

obs = env.reset()
while True:
    action = env.action_space.sample()
    obs, rewards, done, info = env.step(action)
    env.render(title="3D Environment")
    print("I have change some things")
