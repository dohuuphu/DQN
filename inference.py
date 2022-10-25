
import os
import random
import numpy as np
import tensorflow as tf

from tensorflow import keras
from env_kyon import SimStudent
from utils.variables import *

os.environ["CUDA_VISIBLE_DEVICES"]="-1"   

RANDOM_SEED = 5
tf.random.set_seed(RANDOM_SEED)

# env = gym.make('CartPole-v1')
# env.seed(RANDOM_SEED)
# np.random.seed(RANDOM_SEED)

# print("Action Space: {}".format(env.action_space))
# print("State space: {}".format(env.observation_space))

env = SimStudent()

class Agent(keras.Model):
    def __init__(self, state_shape, action_shape, number_topic):
        super().__init__()
        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        init = tf.keras.initializers.HeUniform()

        self.dense1 = keras.layers.Dense(1024, activation=tf.nn.tanh, kernel_initializer=init)
        self.dense2 = keras.layers.Dense(512, activation=tf.nn.tanh, kernel_initializer=init)
        self.dense3 = keras.layers.Dense(256, activation=tf.nn.tanh, kernel_initializer=init)
        self.dense4 = keras.layers.Dense(action_shape, activation='linear', kernel_initializer=init)
        self.topic_embedding = keras.layers.Embedding(number_topic, 8, input_length=1, trainable=False)
        # self.dense_topic = keras.layers.Dense(512)


    def call(self, inputs):
        (observation, topic_ids) = inputs
        topic_embedding = self.topic_embedding(topic_ids)
        # topic_feature = self.dense_topic(topic_embedding)

        x = self.dense1(observation)
        # x += topic_embedding
        # topic_embedding = topic_embedding if topic_embedding.ndim == 2 else tf.expand_dims(topic_embedding, axis=0)
        x = keras.layers.Concatenate(axis=1)([x, topic_embedding])
        x = self.dense2(x)
        x = self.dense3(x)
        output = self.dense4(x)
        return output

def infer():

    model = keras.models.load_model(MODEL_INFERENCE)

    observation, zero_list = env.reset()
    raw_observation = observation
    
    total_zero = (observation == 0.0).sum()
    pos_zero = np.where(observation==0.0)
    topic = [0, 1]
    for topic_number in topic:
        pred_actions = []
        print('topic_number: ', topic_number)
        print('observation: ', observation)
        while 0.0 in observation:
            # encoded = observation
            # encoded_reshaped = encoded.reshape([1, encoded.shape[0]])
            random_topic = np.array([0])#np.random.randint(0, NUM_TOPIC, size=(1,))
            encoded = observation
            encoded_reshaped = encoded.reshape([1, encoded.shape[0]])
            predicted = model((encoded_reshaped, np.array([topic_number])))
            action = np.argmax(predicted)
            pred_actions.append(action)
            new_observation, reward, done, info = env.step(action, zero_list, 0, 0)
            while np.array_equal(observation, new_observation):
                action = random.randint(0, STATE_ACTION_SPACE-1)
                new_observation, reward, done, info = env.step(action, zero_list, 0, 0)
            observation = new_observation

        observation = raw_observation.copy()
        env.reset(observation)
        miss_pos = np.setdiff1d(pos_zero, pred_actions)
        wrong_pred = np.setdiff1d(pred_actions, pos_zero)
        duplicate = []
        for i in np.unique(np.array(pred_actions)):
            if pred_actions.count(i) >=2:
                duplicate.append(i)
        print(f'numzero = {total_zero}/{len(pred_actions)}\npos = {pos_zero}\npred = {pred_actions}\nmiss_pos = {miss_pos}\nwrong_pred = {wrong_pred}\nduplicate = {duplicate} ')        
        print('\n=======================================\n')
if __name__ == '__main__':
    infer()