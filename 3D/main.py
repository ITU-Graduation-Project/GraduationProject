from env.SimulationEnv import SimulationEnv
# from pynput import keyboard
import tqdm
from env.dqn import DQNAgent
import numpy as np
import random
import time

# Environment settings
EPISODES = 500_000
MAX_ITER_FOR_EPISODE = 30
# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99982
MIN_EPSILON = 0.2

#  Stats settings
AGGREGATE_STATS_EVERY = 20  # episodes
SAVE_EVERY = 300
SHOW_EVERY = 10

ep_rewards = [-200]

action_counter = [0] * 27


agent = DQNAgent()
# agent2 = DQNAgent()
env = SimulationEnv()

MODEL_NAME = 'straight'

obs = env.reset()

for episode in tqdm.tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    # MAX_ITER_FOR_EPISODE += 0.01
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
            # print("current_state:", current_state)
            q_vals = agent.get_qs(current_state)
            print("currebt state:", current_state)
            print("prew reward:", reward)
            print("q_vals:", q_vals)
            decoded_action = np.argmax(q_vals)
            action_counter[decoded_action] += 1

            print("action counter:", "*"*20, " : ", action_counter)
            print("decoded_action:", decoded_action)

            decoded_action = np.base_repr(decoded_action, base=3)

            while len(decoded_action) < 3:
                decoded_action = "0" + decoded_action
            action = [int(decoded_action[0]), int(decoded_action[1]), int(decoded_action[2])]
        else:
            # Get random action
            action = [random.randint(0, 2), random.randint(0, 2), random.randint(0, 2)]

        # rival_decoded = revert_obs(current_state.copy())
        # rival_decoded = np.argmax(agent2.get_qs(rival_decoded))
        # rival_decoded = np.base_repr(rival_decoded, base=3)

        # while len(rival_decoded) < 3:
        #    rival_decoded = "0" + rival_decoded

        # action += [int(rival_decoded[0]), int(rival_decoded[1]), int(rival_decoded[2])]
        action += [1, 1, 1]
        # print("action:", action)
        new_state, reward, done, _ = env.step(action, episode)
        # print(reward)
        # print("new_state:", new_state)

        # revert the action
        action = action[0] * (3 ** 2) + action[1] * (3 ** 1) + action[2] * (3 ** 0)
        # print(reward)
        # Transform new continuous state to new discrete state and count reward
        episode_reward += reward

        if counter % 499 == 0:
            print(reward)
        if done:
            print("done with reward:", reward)

        if not episode % SHOW_EVERY:
            env.render()
            # print(reward)
            # pass

        # Every step we update replay memory and train main network

        agent.update_replay_memory((current_state, action, reward, new_state, done))

        current_state = new_state
        step += 1
        agent.train(done)

    if not episode % SHOW_EVERY:
        env.close(episode)
    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) \
                         / (len(ep_rewards[-AGGREGATE_STATS_EVERY:]) * MAX_ITER_FOR_EPISODE)
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:]) / MAX_ITER_FOR_EPISODE
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:]) / MAX_ITER_FOR_EPISODE
        print("min_reward:", min_reward)
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward,
                                       reward_max=max_reward, epsilon=epsilon)

    if not episode % SAVE_EVERY:
        # Save model, but only when min reward is greater or equal a set value
        agent.model.save(
            f'models/fix_max_iter_relative_episode{episode}__{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)