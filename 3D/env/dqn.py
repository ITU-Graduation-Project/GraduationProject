import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import adam_v2 as Adam
import os
from collections import deque
import numpy as np
import time
import random

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 100_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 10_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 512  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 50  # Terminal states (end of episodes)
MODEL_NAME = '2x256'


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = self.log_dir

    def set_model(self, model):
        self.model = model

        self._train_dir = os.path.join(self._log_write_dir, 'train')
        self._train_step = self.model._train_counter

        self._val_dir = os.path.join(self._log_write_dir, 'validation')
        self._val_step = self.model._test_counter

        self._should_write_train_graph = False

    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_end(self, _):
        pass

    def update_stats(self, **stats):
        with self.writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, value, step=self.step)
                self.writer.flush()


class DQNAgent:
    def __init__(self):
        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        # print("state.shape:", state.shape)
        q_vals = self.model.predict(state.reshape(1, 11))[0]
        # print("q vals:", q_vals)
        return q_vals

    def create_model(self, path=None):
        """
            Builds a deep neural net which predicts the Q values for all possible
            actions given a state. The input should have the shape of the state
            (which is 4 in CartPole), and the output should have the same shape as
            the action space (which is 2 in CartPole) since we want 1 Q value per
            possible action.
            :return: the Q network
            """
        if path is not None:
            q_net = tf.keras.models.load_model(path)
            return q_net
        q_net = Sequential()
        initializer = tf.keras.initializers.VarianceScaling(
            scale=0.1, mode='fan_in', distribution='uniform')
        q_net.add(Dense(512, input_dim=11, activation='tanh', kernel_initializer=initializer, bias_initializer='zeros'))
        q_net.add(Dense(1024, activation='tanh', kernel_initializer=initializer, bias_initializer='zeros'))
        q_net.add(Dense(512, activation='tanh', kernel_initializer=initializer, bias_initializer='zeros'))
        # q_net.add(Dense(128, activation='tanh', kernel_initializer='zeros'))
        q_net.add(Dense(9, kernel_initializer='he_uniform'))
        q_net.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss='mse')

        return q_net

    # Trains main network every step during episode
    def train(self, terminal_state):
        # print("memory length:", len(self.replay_memory) )
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            print("not training")
            return
        #print("training")
        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            # print("current state:", current_state)
            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False,
                       callbacks=[self.tensorboard])

        # Update target network counter every episode
        self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
