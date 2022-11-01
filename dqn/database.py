
import logging
import pymongo
import zlib

from variables import *
from threading import Lock


class Format_reader():
    def __init__(self, total_topic:list, topic_name:str, prev_action:int, zero_list:list, observation:list) -> None:
        self.total_topic = total_topic
        self.topic_name = topic_name
        self.prev_action = prev_action
        self.zero_list = zero_list
        self.observation = observation

class Format_writer():
    def __init__(self, **kwarg) -> None:
        pass
        

class User():
    def __init__(self, user_id:str, user_mail:str, subject:str, level:int, plan_name:str, topic_masteries:dict, action_index:int, action_id:int, prev_score:list, topic_name:str, init_score:int = None, flow_topic:list=None, path_status:str=INPROCESS, topic_status:str=INPROCESS) -> None:
        self.id = user_id
        self.mail = user_mail
        self.subject= subject
        self.level = str(level)
        self.flow_topic = flow_topic
        self.plan_name = plan_name # depend backend
        
        self.init_score = init_score
        self.topic_name = topic_name

        self.topic_masteries = topic_masteries
        self.action_index = action_index
        self.action_id = action_id
        self.prev_score = prev_score
        
        self.path_status = path_status
        self.topic_status = topic_status 

class Data_formater():
    def __init__(self, user:User) -> None:
        self.user:User = user
    
    def user_info(self):
        return {
            "user_id": self.user.id,
            "user_gmail": self.user.mail,
            "subject": self.subject_info()
            }
    
    def subject_info(self, prefix:str=None):
        key = f'{prefix}.{self.user.subject}' if prefix else self.user.subject
        return { key: self.level_info() }
                
    def level_info(self, prefix:str=None):
        key = f'{prefix}.{self.user.level}' if prefix else self.user.level
        return {key: self.path_info()}

    def path_info(self, prefix:str=None):
        key = f'{prefix}.{self.user.plan_name}' if prefix else self.user.plan_name
        return {key:{
                        "status": self.user.path_status,
                        "total_topic": self.user.flow_topic,
                        "init_score": self.user.init_score,
                        "flow": self.topic_info()
                    }}
                
    def topic_info(self, prefix:str=None):
        key = f'{prefix}.{self.user.topic_name}' if prefix else self.user.topic_name
        return {key:[
                    self.step_info()
                    ],}

    def step_info(self):
        return {
                "action_recommend": self.user.action_index,
                "action_ID": self.user.action_id,
                "score": None,
                "masteries":self.user.topic_masteries
            }
    
    def get_score_path(self, num_step:int):
        return f'subject.{self.user.subject}.{self.user.level}.{self.user.plan_name}.flow.{self.user.topic_name}.{num_step}.score'


    def get_step_path(self):
        return f'subject.{self.user.subject}.{self.user.level}.{self.user.plan_name}.flow.{self.user.topic_name}'

class MongoDb:
    def __init__(self):
        self.client =  pymongo.MongoClient(COLLECTION_PATH)
        self.mydb = self.client[MONGODB_NAME]
        self.user_db = self.mydb[COLLECTION_NAME]
        self.content_db = self.mydb[COLLECTION_DATA]
        self.content_doc = self.content_db.find_one() # review

        self.locker = Lock()


    # def create_dict_topicID(self):
    #     class Topic_id():
    #         lv_10 = None
    #         lv_11 = None
    #         lv_12 = None

    #     for level in ['10','11', '12']
    #     topic_id_10 = self.get_topic_id('10')

    def get_topic_id(self, subject:str):
        dict_id = {}
        id = 0
        for level in self.content_doc[subject]:
            for category in self.content_doc[subject][level]:
                for topic_name in self.content_doc[level][category]:
                    dict_id.update({topic_name:id})
        return dict_id

    def preprocess_userInfo(self, user_info:User):

        if user_info.topic_status == DONE:
            # Check plan is done
            next_index = user_info.flow_topic.index(user_info.topic_name) + 1
            if next_index >= len(user_info.flow_topic) :
                print('done path')
                user_info.path_status = DONE

            # else:
            #     # Get next topic_name & topic_masteries
            #     user_info.topic_name = user_info.flow_topic[next_index]  
            #     user_info.topic_masteries = self.get_topic_masteries(user_info.topic_name)

        else:
            pass

        return user_info

    def is_newUser(self, user_id:int)->bool:
        num_doc = self.user_db.count_documents({"user_id":user_id})
        if num_doc  == 0:
            logging.getLogger(SYSTEM_LOG).info(f"New user {user_id} ")
            return True

        return False
    
    def is_newSubject(self, user:User)->bool:
        try: # Incase user doesn't have path before
            myquery = {"user_id":user.id}
            doc = self.user_db.find(myquery)[0]
            exist_path = list(doc['subject'].keys())
            if user.subject not in exist_path:
                return True
        except:
            pass

        return False

    def is_newLevel(self, user:User)->bool:
        try: # Incase user doesn't have path before
            myquery = {"user_id":user.id}
            doc = self.user_db.find(myquery)[0]
            exist_path = list(doc['subject'][user.subject].keys())
            if user.level not in exist_path:
                return True
        except:
            pass

        return False

    def is_newPath(self, user:User)->bool:
        try: # Incase user doesn't have path before
            myquery = {"user_id":user.id}
            doc = self.user_db.find(myquery)[0]
            exist_path = list(doc['subject'][user.subject][user.level].keys())
            if user.plan_name not in exist_path:
                return True
        except:
            pass

        return False
    
    def is_newTopic(self, user:User)->bool:
        try: 
            myquery = {"user_id":user.id}
            doc = self.user_db.find(myquery)[0]
            exist_path = list(doc['subject'][user.subject][user.level][user.plan_name]['flow'].keys())
            if user.topic_name not in exist_path:
                return True
        except:
            pass

        return False

    def get_topic_masteries(self,  subject:str, level:str, topic_name:str=None, total_masteries:dict=None)->dict:
        content = self.content_doc[subject][level].copy()
        for category in content:
            if topic_name in content[category]:
                topic_masteries = content[category][topic_name]
                for lesson_id in total_masteries:
                    if lesson_id in topic_masteries:
                        topic_masteries[lesson_id] = float(total_masteries[lesson_id]) # Update masteries from total to topic
            else:
                print('topic does not exist in database')

        return  topic_masteries

    def prepare_flow_topic(self, subject:str, level:str, total_masteries:dict=None)->list:
        '''
            Repare topic_flow and return curr_topic value
        '''
        dict_exist_topic = {}
        for lesson_id in total_masteries:
            topic_name, dict_values = self.get_topic_info(subject, level, lesson_id)

            # Check lesson_id is exist in database and update lesson value
            if topic_name is not None and dict_values is not None:

                if topic_name not in list(dict_exist_topic.keys()): # New topic
                   
                    dict_values[lesson_id] = float(total_masteries[lesson_id])  # update values of lessson
                    dict_exist_topic.update({topic_name:dict_values})
                
                else: # Exist topic
                    dict_exist_topic[topic_name][lesson_id] =  float(total_masteries[lesson_id])  # update values of lessson
            else:
                pass # Log lesson_id not exist in database

        # Calulate weight 
        dict_topic_weight = self.calculate_topic_weight(dict_exist_topic)
 
        flow_topic = [topic_name for topic_name, _ in dict_topic_weight]
        return flow_topic,  dict_exist_topic[flow_topic[0]]
    
    def calculate_topic_weight(self,  dict_topic:dict)->dict:
        dict_topic_weight = {}
        for topic_name in dict_topic:
            list_value = list(dict_topic[topic_name].values())
            avarage_value = float(sum(list_value) / len(list_value))
            dict_topic_weight.update({topic_name:avarage_value})
        
        return sorted(dict_topic_weight.items(), key=lambda item: item[1], reverse=True) # sort value High -> Low

    def get_topic_info(self, subject:str, level:str, lesson_id:str):
        '''
            English:
                10:
                    grammar:
                        topic_1: {lesson_id:1, lesson_id:1,...}         
        '''
        content = self.content_doc[subject][level].copy()
        for category in content:
            for topic_name in content[category]:
                if lesson_id in list(content[category][topic_name].keys()):
                    return topic_name, content[category][topic_name]

        return None, None


    def read_from_DB(self, user_id:str, subject:str, level:int, plan_name:str):

        doc = self.user_db.find({'user_id': user_id })[0]
        try:
            total_topic = doc['subject'][subject][level][plan_name]['total_topic']
            dict_flow = doc['subject'][subject][level][plan_name]['flow']
            prev_topic_name:str = list(dict_flow.keys())[-1]
            prev_step:dict = dict_flow[prev_topic_name][-1]
            prev_action:int = prev_step['action_recommend']
            prev_masteries:dict = prev_step['masteries']

            observation = list(prev_masteries.values())
            zero_list = [i for i in range(len(observation)) if observation[i] == 0.0] 
      
        except: # new user
            total_topic = prev_topic_name = prev_action = zero_list = observation = None
            
        return Format_reader(total_topic, prev_topic_name, prev_action, zero_list, observation)

    def write_to_DB(self, raw_info:User):
        
        # Preprocess data
        parsed_user = self.preprocess_userInfo(raw_info)

        # Formated data 
        data = Data_formater(parsed_user)

        user_indentify = {'user_id': data.user.id }

        if self.is_newUser(data.user.id): 
            self.user_db.insert_one(data.user_info())

        elif self.is_newSubject(data.user):
            prefix = 'subject'
            self.user_db.update_one(user_indentify, {'$set':data.subject_info(prefix)})

        elif self.is_newLevel(data.user):
            prefix = f'subject.{data.user.subject}'
            self.user_db.update_one(user_indentify, {'$set':data.level_info(prefix)})

        elif self.is_newPath(data.user):
            prefix = f'subject.{data.user.subject}.{data.user.level}'
            self.user_db.update_one(user_indentify, {'$set':data.path_info(prefix)})

        elif self.is_newTopic(data.user):
            prefix = f'subject.{data.user.subject}.{data.user.level}.{data.user.plan_name}.flow'
            self.user_db.update_one(user_indentify, {'$set':data.topic_info(prefix)})

        else:
            self.update_step(data)
            self.update_prev_score(data)

    def update_step(self, data:Data_formater):
        myquery = {"user_id":data.user.id}
        step_path = data.get_step_path()

        new_val = {"$push":{step_path : data.step_info()} }
        self.user_db.update_one(myquery, new_val)

    def update_prev_score(self, data:Data_formater):
        myquery = {"user_id":data.user.id}
        doc = self.user_db.find(myquery)[0]
        prev_step = len(doc['subject'][data.user.subject][data.user.level][data.user.plan_name]['flow'][data.user.topic_name]) - 2 
        score_path = data.get_score_path(prev_step)

        new_val = {"$set":{score_path : data.user.prev_score} }
        self.user_db.update_one(myquery, new_val)

    def is_userExist(self, user:User):
        num_doc = self.user_db.count_documents({"user_id":user.id})
        if num_doc  == 1:
            return True
        else:
            if num_doc > 1:
                logging.getLogger(SYSTEM_LOG).error(f"Confict {num_doc} doc, user {user.id} ")
        return False



if __name__ == "__main__":
    database = MongoDb()

    # flow = database.prepare_flow_topic(subject='English', level='11', total_masteries={'32':'1', '36':'0', '9':'0', '18':'0'})
    # print(flow)

    readed_data:Format_reader = database.read_from_DB(user_id="3",subject='English', level='10', plan_name='mocktest_1')
    print(readed_data)
    #new user
    # database.write_to_DB(user_id="3", user_mail='3@gmail.com', subject="English", level=10, topic_masteries={"10":1, "11":1, "12":0}, action_index=1, action_id=1, prev_score=None, topic_name="topic_2", init_score = 4, flow_topic= ['topic_2', 'topic_1', 'topic_3'], plan_name='mocktest_1')

    # new step
    # database.write_to_DB(user_id="3", user_mail='3@gmail.com', subject="English", level=10, topic_masteries={"10":2, "11":1, "12":0}, action_index=1, action_id=1, prev_score=5, topic_name="topic_2", init_score = None, flow_topic= None, plan_name='mocktest_1')

    # database.write_to_DB(user_id="3", user_mail='3@gmail.com', subject="English", level=10, topic_masteries={"10":3, "11":1, "12":0}, action_index=2, action_id=2, prev_score=6, topic_name="topic_2", init_score = None, flow_topic= None, plan_name='mocktest_1')

    # new topic
    # database.write_to_DB(user_id="3", user_mail='3@gmail.com', subject="English", level=10, topic_masteries={"10":5, "11":1, "12":0}, action_index=3, action_id=3, prev_score=7, topic_name="topic_1", init_score = 6, flow_topic= None, plan_name='mocktest_1')

    # database.write_to_DB(user_id="3", user_mail='3@gmail.com', subject="English", level=10, topic_masteries={"10":10, "11":1, "12":0}, action_index=3, action_id=4, prev_score=9, topic_name="topic_1", init_score = None, flow_topic= None, plan_name='mocktest_1')


    # #new plan
    # database.write_to_DB(user_id="3", user_mail='3@gmail.com', subject="English", level=10, topic_masteries={"10":0, "11":0, "12":0}, action_index=1, action_id=1, prev_score=None, topic_name="topic_1", init_score = 4, flow_topic= ['topic_1', 'topic_2', 'topic_3'], plan_name='mocktest_2')

    # database.write_to_DB(user_id="3", user_mail='3@gmail.com', subject="English", level=10, topic_masteries={"10":1, "11":0, "12":0}, action_index=2, action_id=1, prev_score=5, topic_name="topic_1", init_score = None, flow_topic= None, plan_name='mocktest_2')

    # database.write_to_DB(user_id="3", user_mail='3@gmail.com', subject="English", level=10, topic_masteries={"10":2, "11":0, "12":0}, action_index=3, action_id=3, prev_score=10, topic_name="topic_1", init_score = None, flow_topic= None, plan_name='mocktest_2')


    # #new level
    # database.write_to_DB(user_id="3", user_mail='3@gmail.com', subject="English", level=11, topic_masteries={"10":1, "11":1, "12":0}, action_index=1, action_id=1, prev_score=None, topic_name="topic_2", init_score = 4, flow_topic= ['topic_2', 'topic_1', 'topic_3'], plan_name='mocktest_1')

    # database.write_to_DB(user_id="3", user_mail='3@gmail.com', subject="English", level=11, topic_masteries={"10":0, "11":0, "12":0}, action_index=1, action_id=1, prev_score=None, topic_name="topic_1", init_score = 4, flow_topic= ['topic_1', 'topic_2', 'topic_3'], plan_name='mocktest_2')

    # database.write_to_DB(user_id="3", user_mail='3@gmail.com', subject="English", level=11, topic_masteries={"10":1, "11":0, "12":0}, action_index=2, action_id=2, prev_score=5, topic_name="topic_1", init_score = 4, flow_topic= ['topic_1', 'topic_2', 'topic_3'], plan_name='mocktest_2')
    
    #new subject
    # database.write_to_DB(user_id="3", user_mail='3@gmail.com', subject="Math", level=11, topic_masteries={"10":1, "11":1, "12":0}, action_index=1, action_id=1, prev_score=None, topic_name="topic_2", init_score = 4, flow_topic= ['topic_2', 'topic_1', 'topic_3'], plan_name='mocktest_1')

    # database.write_to_DB(user_id="3", user_mail='3@gmail.com', subject="Math", level=11, topic_masteries={"10":0, "11":0, "12":0}, action_index=1, action_id=1, prev_score=None, topic_name="topic_1", init_score = 4, flow_topic= ['topic_1', 'topic_2', 'topic_3'], plan_name='mocktest_2')

    # database.write_to_DB(user_id="3", user_mail='3@gmail.com', subject="Math", level=11, topic_masteries={"10":1, "11":0, "12":0}, action_index=2, action_id=2, prev_score=5, topic_name="topic_1", init_score = 4, flow_topic= ['topic_1', 'topic_2', 'topic_3'], plan_name='mocktest_2')