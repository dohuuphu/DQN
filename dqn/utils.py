
import time
import functools
import numpy as np
import logging
import pickle

from dqn.variables import STATE_ACTION_SPACE, SYSTEM_LOG

class Item_relayBuffer:
    def __init__(self, total_step, observation:list, topic_id:int, action_index, next_observation:list, num_items_inPool, reward= None, done=False, score=None):
        self.total_step:int = total_step
        self.observation:np.ndarray = np.array(observation)
        self.topic_id:int = int(topic_id)
        self.action_index:int = int(action_index)
        self.reward:float = reward      #reward for choice action_index
        self.next_observation:np.ndarray = np.array(next_observation)   # run action_index => next_observation
        self.done:bool = done
        self.score:float = score       # Score of action_index
        self.num_items_inPool:int = num_items_inPool

class Item_cache():
   def __init__(self, step, episode, relay_buffer) -> None:
      self.step = step
      self.episode = episode
      self.relay_buffer = relay_buffer


def timer(func):
    @functools.wraps(func)
    def time_counter(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        run_time = time.time() - start_time
        return result, run_time
    return time_counter

def get_time():
  return time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())


def save_pkl(obj, path):
  with open(path, 'wb') as f:
    pickle.dump(obj, f)
    print("  [*] save %s" % path)


def load_pkl(path):
  try:
    with open(path, 'rb') as f:
      obj = pickle.load(f)
      print("  [*] load %s" % path)
      return obj
  except:
    pass

@timer
def save_npy(obj, path):
  np.save(path, obj)
  print("  [*] save %s" % path)

@timer
def load_npy(path):
  obj = np.load(path)
  print("  [*] load %s" % path)
  return obj


def rawObservation_to_standarObservation(raw_observation:list, topic:str)->list:
        standar_observation = [1.0]*STATE_ACTION_SPACE

        if len(raw_observation) > len(standar_observation):
            info = f'topic {topic} have length more than standar_observation'
            logging.getLogger(SYSTEM_LOG).error(info)

        # Update value from raw to standar
        for id, val in enumerate(raw_observation):
            standar_observation[id] = val

        return standar_observation

# @safety_thread
# def read_from_DB(student_id:str, subject:str, level:int):
#     # Doing something
#     action, topic_id, zero_list, observation = None
        
#     return action, topic_id, zero_list, observation
    
# @safety_thread
# def write_to_DB( student_id:str, subject:str, level:int, masteries_of_topic:dict, action_index:int, action_id:int, prev_score:list, topic_name:str):

    
    
#     '''	
#         - student_id
#         - student_gmail(optinal)
#         - subject:
#             + English
#                 + 10:
#                     + path_1 (mocktest_1):
#                         + status: pending/inprocess/done
#                         + total_topic:
#                             + topic_1: 1
#                             + topic_3: 2
#                             + topic 2: 3
#                             + ...
#                         + base_score:
#                         + topic_1 (topic_name): 
#                             + step:
#                                 + 0: 
#                                     + action_recommend: (depend on idex in masteries)
#                                     + action_ID: (lesson_id)
#                                     + score:
#                                     + masteries{lesson_id:value, ...} # (step_inference) (masteries of topic)
#                                 + 1: 
#                                     + action_recommend: (depend on idex in masteries)
#                                     + action_ID: (lesson_id)
#                                     + score:
#                                     + masteries{lesson_id:value, ...} # (step_inference) (masteries of topic)
#                             + id: 1 # Need a function to create and map all topic_name->id
#                             + flow: 1   # Indicate subject selection
#                             + status: pending/inprocess/done

#                         + topic_2:
#                             + masteries:
#                                 + 0: {lesson_id:value, ...}
#                                 + 1:  {lesson_id:value, ...}
#                             + id: 2
#                             + flow: None 
#                             + status: pending/inprocess/done
#                         ...
#                         + topic_n: ... # number of topic is depend on num_quest that related to the topic

#                     + path_2:
#                         ...
#                 + 11: ..
            
#             + Math:
#                 ...


#     - content:
#         - Englist:
#             - 11:
#                 - Vocabulary (category):
#                     - topic_1 (topic name): [id_1, id_2, ...]
#                     - topic_2 (topic name): [id_1, id_2, ...]
#                 - Grammar:
#                     - topic_1 (topic name): [id_1, id_2, ...]
            
#             - 12:
#                 - Vocabulary (category):
#                     - topic_1 (topic name): [id_1, id_2, ...]
#                     - topic_2 (topic name): [id_1, id_2, ...]
#                 - Grammar:
#                     - topic_1 (topic name): [id_1, id_2, ...]
#         - Math:

#     '''
#     # return True/False
#     pass

    