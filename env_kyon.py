# from variables import *
# from utils_ import sigmoid, test_gen, test_gen_after, topic_recommender, mask_others_lp_not_in_topic


from gym import spaces
from collections import Counter
import ast
from gym.spaces.box import Box
# from params import train_params
import random
import numpy as np

import numpy as np
from tensorflow import keras
import tensorflow as tf

from variables import *
# def log_INFO(message):
#   # print(message)
#   logger.info(message)
  
class SimStudent():
      # An observation of the environment is a list of skill mastery, each value [0,1] represents the student's learning progress of a skill
  def __init__(self, intelligence = 50, luck=50, level = "10"):
    self.observation_space = [0, 1, 80] # min max shape
    self.action_space = Box(0.,80.,(1,))
    self.v_min = -100.0
    self.v_max = 0.

    self.answer_rate1 = 1
    self.answer_rate2 = 0.0
    self.forget_rate = 0
    self.learn_prob = 1

    self.true_masteries = []

    # Initialize history
    self.history = []
    self.history_topic = []

    self.percent_done_topic = 0
  #===============
  def get_state_dims(self):
    return (self.observation_space[2],)

  def get_state_bounds(self):
    return np.array([self.observation_space[0]], dtype=np.float32), np.array([self.observation_space[1]], dtype=np.float32)

  def get_action_dims(self):
    return (self.action_space[2],)

  def get_action_bounds(self):  
     return np.array([self.action_space[0]], dtype=np.float32), np.array([self.action_space[1]], dtype=np.float32)
  
  def normalise_state(self, state):
    return state

  def normalise_reward(self, reward):
    return reward

  # def set_random_seed(self, seed)
  #   self._np_random, seed = seeding.np_random(seed)
  #       return [seed]

#=================================
  def mask_others_lp_not_in_topic(self,topic:str):
    
    mask_masteries = LP_PER_TOPICS.copy()
    for idx, val in enumerate(self.true_masteries[self.lp_segment[topic][0]:self.lp_segment[topic][1]]):
      mask_masteries[idx] *= val

    return mask_masteries

  def get_LPvalue(self) -> dict:
    list_LPvalue = {}
    for topic in self.lp_segment: 
      start_idx, stop_idx = self.lp_segment[topic]
      list_LPvalue.update({topic: {LP_VALUE_STR:[self.true_masteries[i] for i in range(start_idx, stop_idx, 1)],
                                  }}) #LP_DIFFICULT_STR:[self.lp_difficult_value[i] for i in range(start_idx, stop_idx, 1)]

    return list_LPvalue

  def calculate_topicWeight(self) -> dict:
    '''Calculate topic_weight by LP_value and LP_difficult
      Return format: {topic : weight (0-1)} '''
    dict_LPvalue = self.get_LPvalue()
    topic_weights = {}
    for topic in dict_LPvalue:
      weight = 0
      total_element = 0
      for value, difficult in zip(dict_LPvalue[topic][LP_VALUE_STR], dict_LPvalue[topic][LP_DIFFICULT_STR]):
        weight += (value*difficult)
        total_element += difficult
      
      #update dict_result
      topic_weights.update({topic : (weight/total_element)})

    return topic_weights

  def find_minTopicWeight(self, topic_weights:dict) -> str:
    '''find minimum topicWeight
      Return topic name'''
    return min(topic_weights, key=topic_weights.get)
    
  def topic_recommender(self)-> str: 
    try:
      curr_topic = self.history_topic[-1] # define after
    except:
      curr_topic = None
    topic_Weights = self.calculate_topicWeight()
    if curr_topic is None or topic_Weights[curr_topic] == 1:
      curr_topic = self.find_minTopicWeight(topic_Weights)
      # self.history = [] # reset history
    
    return curr_topic

  def answer_question(self, question, masteries):
    raw_prob = 1
    for req in question:
      raw_prob = raw_prob*masteries[req]/self.skill_level[req]
    answer_prob = raw_prob*self.answer_rate1 + self.answer_rate2
    if answer_prob>1: answer_prob = 1
    if answer_prob<0: answer_prob = 0
    answer = np.random.choice([1,0],p=[answer_prob, 1-answer_prob])
    return answer

  def get_test_score(self, test, masteries):
    score = 0
    for question in test:
      answer = self.answer_question(question, masteries)
      if answer==1: score+=1
    return int(score/NUM_QUESTIONS_PER_TEST*10)

  def test_update_masteries(self, test):
    correct, all = {}, {}
    new_masteries = np.zeros(len(self.skill_INDs))
    for skill in self.skill_INDs:
      correct[skill] = 0
      all[skill] = 0
    for question in test:
      answer = self.answer_question(question, self.true_masteries)
      for skill in question:
        if answer==1: correct[skill]+=1
        all[skill]+=1
    for skill in all:
      if all[skill]!=0:
        new_masteries[skill] = int(correct[skill]/all[skill]*self.skill_level[skill])
    return [int(i) for i in new_masteries]
    
  def lesson_update_masteries(self, lesson_ind):
    new_masteries = self.true_masteries
    contained_skills = self.lesson[int(lesson_ind)]['skills']
    for skill,mastery in enumerate(self.true_masteries):
      if mastery < self.skill_level[skill] and skill in contained_skills:
        gain = np.random.choice([1,0],p=[self.learn_prob, 1-self.learn_prob])
        new_masteries[skill] += gain
    return [int(i) for i in new_masteries]

  def forget_update_masteries(self):
    new_masteries = self.true_masteries
    num_actions = len(self.history)
    for skill,mastery in enumerate(self.true_masteries):
      forget_prob = self.forget_rate*sigmoid(num_actions)*0.2
      loss = np.random.choice([1,0],p=[forget_prob, 1-forget_prob])
      if self.true_masteries[skill]>0: new_masteries[skill] -= loss
    return [int(i) for i in new_masteries]

  def is_complete_topic(self, topic_name):
    dict_LPvalue = self.get_LPvalue()
    count_zeros = dict_LPvalue[topic_name][LP_VALUE_STR].count(0)
    # for value in dict_LPvalue[topic_name][LP_VALUE_STR]:
    if count_zeros > 0 :
      self.percent_done_topic = count_zeros/len(dict_LPvalue[topic_name][LP_VALUE_STR])
      return False
    return True  

  def is_complete_topic_api(self, prev_state, action):
    # prev_state[action]
    pass

  def reset_infoInTopic(self):
    self.last_score = 0
    self.history = []
    self.history_topic = []

  def step(self, action, zero_list:list, current_step:int, topic:int, topic_feature):   
    reward = 0
    reward_scale = 5
    done = False
    # action = self.align_action(action) if random_value else int(action)
    action = round(float(action))
    self.history.append(action)
    
    if self.true_masteries[action] == 1.0:
      # reward -= reward_scale*2
      reward-=1
    else:
      pos_zero = zero_list.index(action)#/len(self.true_masteries)*reward_scale
      if topic == 0:
        reward += 1 if not RELATION else pos_zero
      else:
        reward += 1 if not RELATION else STATE_ACTION_SPACE - pos_zero
    self.true_masteries[action] = 1.0

    

    reward -= len(self.history)*0.02 if REWARD_LENGTH else 0

    if not 0.0 in self.true_masteries or current_step >= 500:
      done = True

    return self._get_obs(self.true_masteries, topic_feature), reward, done, {}

  def align_action(self, raw_action:np.array):
    '''
    -1 -> 1
    0  -> 20
    ((x - (-1))/2)*19
    '''
    return int(int(raw_action+1)*9.5)
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

  def step_api(self, index, level, curr_topic, action, prev_state, history_score): 
    reward = 0
    done = False
    # lp_segment = LESSON_DATABASE.get_LPsegments(level)
    if prev_state is not None:  
      # action = history_action[-1]
      # action = np.where(action == np.amax(action))[0] # using for discrete
      # action = action.astype(np.int32)
      # history_topic = list(history_action.keys()) # need mapping
      # action_mapping = lp_segment[curr_topic][0] + action
      action_mapping = action

      # reward for predict prev_action
      # if len(self.history)>0:
      #   for i in range(len(self.history)-1 , -1 , -1):
      #     if self.history[i]==action_mapping: 
      #       reward+=0
      #     else:
      #       break


      # if action >= (lp_segment[curr_topic][1]-lp_segment[curr_topic][0]) or action < 0:
      #   reward += -1
      #   num_same_act = self.count_consecutive_actions(action_mapping)
      #   # reward += (num_same_act-1)*(-5)

      # else:
      if prev_state[int(action)] == 1:
        reward += -1
      else:
        reward += 0

      reward -= index
        # num_same_act = self.count_consecutive_actions(action_mapping)
        # reward += (num_same_act-1)*(-1)

      # Check learing a topic is done
      # done = self.is_complete_topic_api(prev_state, action)
      # if done: 
      #   reward+=1
        # self.reset_infoInTopic()

    return np.array(reward, dtype=np.float32), done

  def reset(self):
    # self.true_masteries =  ast.literal_eval(open('/mnt/c/Users/dohuu/Desktop/kyons_AI/Deep-Reinforcement-Learning-in-Large-Discrete-Action-Spaces/fix_masteries.txt', 'r').read())
    # self.true_masteries = np.random.randint(2, size=STATE_ACTION_SPACE)
    self.true_masteries = np.random.choice([0,1], p=[0.9, 0.1], size= STATE_ACTION_SPACE)

    self.history = []

    # Get zero index
    zero_list = [i for i in range(len(self.true_masteries)) if self.true_masteries[i] == 0.0]
    return self._get_obs(self.true_masteries), zero_list

  def _get_obs(self, masteries, topic_feature = None):
    np_value = np.array([i*1.0 for i in masteries], np.float32)
    if tf.is_tensor(topic_feature):
      observation_concate = keras.layers.Concatenate(axis=0)([np_value.reshape([1, np_value.shape[0]]), topic_feature])
    else: 
      observation_concate = None
    return observation_concate, np_value #.reshape(len(masteries),1)

  def preview(self):
    print('Learning probability:',self.learn_prob)
    print('Forget rate:',self.forget_rate)
    print('Known answer rate:',self.answer_rate1)
    print('Unknown answer rate:',self.answer_rate2)
    print('Skill masteries:',self.masteries)

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