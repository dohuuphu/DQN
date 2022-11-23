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

    self.masteries = []
    # Initialize history
    self.num_history = 0


    self.percent_done_topic = 0


  def step_api(self, total_step:int, action:int, observation_:list, num_items_inPool:int, score:int=None): 
    '''
      action was recommended by pass observation to model
      total_step was read from database
    '''
    reward = 0
    done = False
    observation = observation_.copy()
    # self.history.update({action:observation[action]})

    if observation[action] == 1.0: # Negative sample
      reward-=1
    else:
      reward +=1
    
    #  Calcualate with score
    if score is not None and observation[action] == 0 and score >= 5:  
      reward += score-4 

      # Update observation
      observation[action] = 1

    # Check done observation (a topic)
    if not 0.0 in observation:
      done = True

      # All recommened action is correct
      if total_step <= num_items_inPool:
        reward += 10
      
      # Exist wrong recommended action (select 1)
      if total_step > num_items_inPool:
        reward -= total_step - num_items_inPool
    
    return reward, done

  def reset(self, observation=None):
    if observation is not None:
      self.masteries = observation.copy()
    else:
      self.masteries = np.random.choice([0,1], p=[0.9, 0.1], size= STATE_ACTION_SPACE)

    self.history = []

    # Get zero index
    zero_list = [i for i in range(len(self.masteries)) if self.masteries[i] == 0.0]
    return self._get_obs(self.masteries), zero_list

  def _get_obs(self, masteries):
    np_value = np.array([i*1.0 for i in masteries], np.float32)
    return np_value
