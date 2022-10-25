
import os
import random
import numpy as np
import tensorflow as tf

from tensorflow import keras
from dqn.model import Agent
from dqn.environment import SimStudent
from dqn.variables import *

os.environ["CUDA_VISIBLE_DEVICES"]="-1"   

RANDOM_SEED = 5
tf.random.set_seed(RANDOM_SEED)

env = SimStudent()

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