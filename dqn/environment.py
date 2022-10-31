# from variables import *
# from utils_ import sigmoid, test_gen, test_gen_after, topic_recommender, mask_others_lp_not_in_topic


# from gym import spaces
from collections import Counter
import ast
# from gym.spaces.box import Box
# from params import train_params
import random
import numpy as np

import numpy as np
from tensorflow import keras
import tensorflow as tf

from dqn.variables import *

  
class SimStudent():
      # An observation of the environment is a list of skill mastery, each value [0,1] represents the student's learning progress of a skill
  def __init__(self, intelligence = 50, luck=50, level = "10"):

    self.true_masteries = []
    # Initialize history
    self.history = []
    self.history_topic = []

    self.percent_done_topic = 0

  def step(self, action, zero_list:list, current_step:int, topic:int):   
    reward = 0
    reward_scale = 5
    done = False
    action = round(float(action))
    self.history.append(action)
    
    if self.true_masteries[action] == 1.0:
      # reward -= reward_scale*2
      reward-=1
    else:
      if topic == 0:
        pos_zero = zero_list.index(action)#/len(self.true_masteries)*reward_scale
        reward += pos_zero
        # if topic == 0:
        #   reward += 1 if not RELATION else pos_zero
        # else:
        #   reward += 1 if not RELATION else STATE_ACTION_SPACE - pos_zero
      else:
        pos_zero = len(zero_list) - zero_list.index(action)
        reward += pos_zero
    self.true_masteries[action] = 1.0

    

    reward -= len(self.history)*0.02 if REWARD_LENGTH else 0

    if not 0.0 in self.true_masteries or current_step >= MAX_STEP_EPISODE:
      done = True

    return self._get_obs(self.true_masteries), reward, done, {}

  def step_api(self, action:int, zero_list:list, score:int): 
    reward = 0
    done = False
    
    return reward, done

  def set_topicDone(self, topic):
    start_idx, stop_idx = self.lp_segment[topic]
    for i in range(start_idx, stop_idx, 1):
      self.true_masteries[i] = 1.0

    #reset 
    self.reset_infoInTopic()
    # Get new state
    curr_topic = self.topic_recommender()
    self.history_topic.append(curr_topic)
    segment_LPs = self.mask_others_lp_not_in_topic(curr_topic)

    return self._get_obs(segment_LPs)

  def reset(self, observation=None):
    if observation is not None:
      self.true_masteries = observation.copy()
    else:
      self.true_masteries = np.random.choice([0,1], p=[0.9, 0.1], size= STATE_ACTION_SPACE)

    self.history = []

    # Get zero index
    zero_list = [i for i in range(len(self.true_masteries)) if self.true_masteries[i] == 0.0]
    return self._get_obs(self.true_masteries), zero_list

  def _get_obs(self, masteries):
    np_value = np.array([i*1.0 for i in masteries], np.float32)
    return np_value

  def count_consecutive_actions(self, action):
    count = 1
    if len(self.history)==0: return count
    else:
      # temp_acts = self.history
      # temp_acts.reverse()
      # for act in temp_acts:
      #   if act==action: count+=1
      #   else: break

      for action_ in self.history:
        if action_ == action: count+=1
      return count-1