import tensorflow as tf
import tensorflow.contrib as contrib
import numpy as np
import random
from collections import deque

ACTIONS = 2
GAMMA = 0.99
OBSERVE = 1000000
EXPLORE = 2000000
# INITIAL_EPSILON = 0.022
INITIAL_EPSILON = 0
FINAL_EPSILON = 0.0001
REPLAY_MEMORY = 50000
BATCH = 32


class DeepQNetwork:
    def __init__(self):
        self.time_step = 0
        self.current_state = None
        self.memory = deque()
        self.epsilon = INITIAL_EPSILON
        self._build()
        self.saver = tf.train.Saver()
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())
        target_params = tf.get_collection('target_net_params')
        eval_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [
            tf.assign(t, e) for t, e in zip(target_params, eval_params)]
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    def _build(self):
        def build_layers(state, weights_initializer, biases_initializer, collection_name, trainable):
            with tf.variable_scope('conv2d_1'):
                conv2d = contrib.layers.convolution2d(state, 32, (8, 8), stride=4, scope='conv2d_1',
                                                      weights_initializer=weights_initializer,
                                                      biases_initializer=biases_initializer,
                                                      variables_collections=collection_name,
                                                      trainable=trainable)
            with tf.variable_scope('max_pool2d'):
                max_pool2d = contrib.layers.max_pool2d(
                    conv2d, (2, 2), padding='same', scope="max_pool2d")
            with tf.variable_scope('conv2d_2'):
                conv2d = contrib.layers.convolution2d(max_pool2d, 64, (4, 4), 2, scope='conv2d_2',
                                                      weights_initializer=weights_initializer,
                                                      biases_initializer=biases_initializer,
                                                      variables_collections=collection_name,
                                                      trainable=trainable)
            with tf.variable_scope('conv2d_3'):
                conv2d = contrib.layers.convolution2d(conv2d, 64, (3, 3), 1, scope='conv2d_3',
                                                      weights_initializer=weights_initializer,
                                                      biases_initializer=biases_initializer,
                                                      variables_collections=collection_name,
                                                      trainable=trainable)
            with tf.variable_scope('flatten'):
                flatten = contrib.layers.flatten(conv2d)
            with tf.variable_scope('dense'):
                dense = contrib.layers.fully_connected(flatten, 512, scope="dense",
                                                       weights_initializer=weights_initializer,
                                                       biases_initializer=biases_initializer,
                                                       variables_collections=collection_name,
                                                       trainable=trainable)
            with tf.variable_scope('output'):
                output = contrib.layers.fully_connected(dense, ACTIONS,
                                                        activation_fn=None,
                                                        weights_initializer=weights_initializer,
                                                        biases_initializer=biases_initializer,
                                                        variables_collections=collection_name,
                                                        trainable=trainable,
                                                        scope="output")
            return output

        self.state = tf.placeholder(
            tf.float32, (None, 80, 80, 4), name='state')
        self.q_target = tf.placeholder(
            tf.float32, (None, ACTIONS), name='q_target')
        w_initializer = tf.truncated_normal_initializer(stddev=0.01)
        b_initializer = tf.constant_initializer(0.01)
        with tf.variable_scope('eval_net'):
            self.q_value = build_layers(self.state, w_initializer, b_initializer,
                                        ["eval_net_params", tf.GraphKeys.GLOBAL_VARIABLES], True)
        with tf.variable_scope('loss'):
            self.loss = tf.losses.mean_squared_error(
                self.q_target, self.q_value)
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(
                1e-6).minimize(loss=self.loss)
        self.next_state = tf.placeholder(
            tf.float32, (None, 80, 80, 4), name='next_state')
        with tf.variable_scope('target_net'):
            self.q_next = build_layers(self.next_state, w_initializer, b_initializer,
                                       ["target_net_params", tf.GraphKeys.GLOBAL_VARIABLES], False)

    def trainNetwork(self, time_step):
        if time_step % 500 == 0:
            self.session.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')
        batch_memory = random.sample(self.memory, BATCH)
        x_batch = np.zeros((BATCH, 80, 80, 4))
        next_state = np.zeros((BATCH, 80, 80, 4))
        terminal = []
        reward = []
        action_index = []
        for i in range(0, BATCH):
            current_state = batch_memory[i][0]
            x_batch[i] = current_state
            next_state[i] = batch_memory[i][3]
            reward.append(batch_memory[i][2])
            terminal.append(batch_memory[i][4])
            action = batch_memory[i][1]
            action_index.append(np.argmax(action))
        q_eval4next, q_next = self.session.run([self.q_value, self.q_next],
                                               feed_dict={self.state: next_state, self.next_state: next_state})
        q_eval = self.session.run(
            self.q_value, feed_dict={self.state: x_batch})
        q_target = q_eval.copy()
        for i in range(0, BATCH):
            if terminal[i]:
                q_target[i][action_index[i]] = reward[i]
            else:
                max_act4next = np.argmax(q_eval4next[i])
                q_target[i][action_index[i]] = reward[i] + \
                    GAMMA * q_next[i][max_act4next]
                # q_target[i][action_index[i]] = reward[i] + GAMMA * np.max(q_next[i])
        loss, _ = self.session.run([self.loss, self.train_op],
                                   feed_dict={self.state: x_batch, self.q_target: q_target})
        # save network every 1000 iteration
        if time_step % 1000 == 0:
            self.saver.save(self.session, 'saved_networks/' +
                            'network' + '-dqn', global_step=time_step + 2363000)
        return loss

    def getAction(self):
        action = np.zeros(2)
        q_value = None
        if random.random() <= self.epsilon:
            print("----------Random Action----------")
            action_index = random.randint(0, ACTIONS - 1)
            action[action_index] = 1
        else:
            q_value = self.session.run(self.q_value, feed_dict={
                                       self.state: self.current_state[np.newaxis, :]})[0]
            action_index = np.argmax(q_value)
            action[action_index] = 1
        return action, q_value

    def setPerception(self, time_step, action, reward, observation, terminal):
        nextObservation = np.append(
            observation, self.current_state[:, :, :3], axis=2)
        self.memory.append((self.current_state, action,
                            reward, nextObservation, terminal))
        loss = None
        if len(self.memory) > REPLAY_MEMORY:
            self.memory.popleft()
        if time_step > OBSERVE:
            loss = self.trainNetwork(time_step)
        self.current_state = nextObservation
        if self.epsilon > FINAL_EPSILON and time_step > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
        return loss

    def setInitState(self, observation):
        self.current_state = np.stack(
            (observation, observation, observation, observation), axis=2)
