from env.SimulationEnv import SimulationEnv
#from pynput import keyboard
import  tqdm
from env.dqn import DQNAgent
import numpy as np
import random

MAX_STEPS = 20000
# Environment settings
EPISODES = 20_000
MAX_ITER_FOR_EPISODE = 5_000
# Exploration settings
epsilon = 0.9  # not a constant, going to be decayed
EPSILON_DECAY = 0.95
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 10  # episodes
SHOW_EVERY = 100
SHOW_PREVIEW = True

ep_rewards = [-200]

agent = DQNAgent()
env = SimulationEnv()

MIN_REWARD = -500
MODEL_NAME = '2x256'
ACTION_SPACE_SIZE = 3

obs = env.reset()
i = 0
a = b = c = 0
import time

for episode in tqdm.tqdm(range(810, EPISODES + 1), ascii=True, unit='episodes'):
    # Update tensorboard step every episode
    agent.tensorboard.step = episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    current_state = env.reset()

    # Reset flag and start iterating until episode ends
    done = False
    print("new episode")
    counter = 0
    while not done and counter < MAX_ITER_FOR_EPISODE:
        counter += 1
        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            #print("current_state:", current_state)
            decoded_action = np.argmax(agent.get_qs(current_state))
            decoded_action = np.base_repr(decoded_action, base=3)
            while(len(decoded_action) < 3):
                decoded_action = "0" + decoded_action
            action = [int(decoded_action[0]), int(decoded_action[1]), int(decoded_action[2])]
        else:
            # Get random action
            action = [random.randint(0, 2), random.randint(0, 2), random.randint(0, 2)]
        action += [0, 0, 0]

        #print("action:", action)
        new_state, reward, done, _ = env.step(action, episode)
        #TO DO: revert back to action
        action = action[0] * (3**2) + action[1] * (3**1) + action[2] * (3**0)
        #print(reward)
        # Transform new continuous state to new discrete state and count reward
        episode_reward += reward

        if counter % 200 == 0:
            print(reward)
        if done:
            print("done with reward:", reward)

        if SHOW_PREVIEW and not episode % SHOW_EVERY:
            env.render()
            #pass

        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done)

        current_state = new_state
        step += 1
    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        print("min_reward:", min_reward)
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
                                       epsilon=epsilon)

        # Save model, but only when min reward is greater or equal a set value
        if min_reward >= MIN_REWARD:
            agent.model.save(
                f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

prev = time.time()

while True:
    #print("time.time()-prev:", time.time()-prev)
    prev = time.time()
    action = env.action_space.sample()
    obs = env.observation_space.sample()
    _, rewards, done, info = env.step(action)
    if(i%1 == 0):
        env.render(title="3D Environment")
        i = 1
    i += 1



