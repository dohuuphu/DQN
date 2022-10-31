
import time
import functools
import numpy as np

from threading import Lock

def safety_thread(func):
    @functools.wraps(func)
    def threading_lock(locker:Lock, *args, **kwargs):
        locker.acquire()
        result = func(*args, **kwargs)
        locker.release()
        return result
    return threading_lock

def timer(func):
    @functools.wraps(func)
    def time_counter(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        run_time = end_time - start_time
        # print("   [-] %s : %2.5f sec" % (f.__name__, end_time - start_time))
        return result, run_time
    return time_counter

def get_time():
  return time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())

@timer
def save_pkl(obj, path):
  with open(path, 'w') as f:
    cPickle.dump(obj, f)
    print("  [*] save %s" % path)

@timer
def load_pkl(path):
  with open(path) as f:
    obj = cPickle.load(f)
    print("  [*] load %s" % path)
    return obj

@timer
def save_npy(obj, path):
  np.save(path, obj)
  print("  [*] save %s" % path)

@timer
def load_npy(path):
  obj = np.load(path)
  print("  [*] load %s" % path)
  return obj

@safety_thread
def read_from_DB(student_id:str, subject:str, level:int):
    # Doing something
    action, topic_id, zero_list, observation = None
        
    return action, topic_id, zero_list, observation
    
@safety_thread
def write_to_DB(student_id:str, subject:str, level:int, masteries_of_topic:dict, action_index:int, action_id:int, prev_score:list, topic_name:str):

    check_exist_student_id = database.mycol.count_documents({"student_id":student_id})
    if not check_exist_student_id:
        database.mycol.insert_one(
            {
                "student_id":student_id,
                "subject":
                {
                    subject:{
                        level:{
                            "mocktest_1":{
                                "status":"inprocess",
                                "base_score": prev_score,
                                topic_name:[
                                    {
                                        "action_recommend": action_index,
                                        "action_ID": action_id,
                                        "score": None,
                                        "masteries":masteries_of_topic
                                    }
                                ],
                                "id" : 1,
                                "flow":1,
                                "status":"inprocess"
                            }
                        }
                    }
                }
            }
        )
    
    '''	
        - student_id
        - student_gmail(optinal)
        - subject:
            + English
                + 10:
                    + path_1 (mocktest_1):
                        + status: pending/inprocess/done
                        + total_topic:
                            + topic_1: 1
                            + topic_3: 2
                            + topic 2: 3
                            + ...
                        + base_score:
                        + topic_1 (topic_name): 
                            + step:
                                + 0: 
                                    + action_recommend: (depend on idex in masteries)
                                    + action_ID: (lesson_id)
                                    + score:
                                    + masteries{lesson_id:value, ...} # (step_inference) (masteries of topic)
                                + 1: 
                                    + action_recommend: (depend on idex in masteries)
                                    + action_ID: (lesson_id)
                                    + score:
                                    + masteries{lesson_id:value, ...} # (step_inference) (masteries of topic)
                            + id: 1 # Need a function to create and map all topic_name->id
                            + flow: 1   # Indicate subject selection
                            + status: pending/inprocess/done

                        + topic_2:
                            + masteries:
                                + 0: {lesson_id:value, ...}
                                + 1:  {lesson_id:value, ...}
                            + id: 2
                            + flow: None 
                            + status: pending/inprocess/done
                        ...
                        + topic_n: ... # number of topic is depend on num_quest that related to the topic

                    + path_2:
                        ...
                + 11: ..
            
            + Math:
                ...


    - content:
        - Englist:
            - 11:
                - Vocabulary (category):
                    - topic_1 (topic name): [id_1, id_2, ...]
                    - topic_2 (topic name): [id_1, id_2, ...]
                - Grammar:
                    - topic_1 (topic name): [id_1, id_2, ...]
            
            - 12:
                - Vocabulary (category):
                    - topic_1 (topic name): [id_1, id_2, ...]
                    - topic_2 (topic name): [id_1, id_2, ...]
                - Grammar:
                    - topic_1 (topic name): [id_1, id_2, ...]
        - Math:

    '''
    # return True/False
    pass

    
