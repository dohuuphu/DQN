
import logging
import pymongo

from variables import *


class User():
    def __init__(self, user_id:str, user_mail:str, subject:str, level:int, plan_name:str, topic_masteries:dict, action_index:int, action_id:int, prev_score:list, topic_name:str, init_score:int = None, flow_topic:list=None ) -> None:
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
        
        self.path_status = INPROCESS
        self.topic_status = INPROCESS 




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
                        self.user.topic_name:[
                            self.step_info()
                        ],
                    }}
        

    def step_info(self):
        return {
                "action_recommend": self.user.action_index,
                "action_ID": self.user.action_id,
                "score": None,
                "masteries":self.user.topic_masteries
            }
    
    def get_score_path(self, num_step:int, num_plan:int=None):
        return f'subject.{self.user.subject}.{self.user.level}.{self.user.plan_name}.{self.user.topic_name}.{num_step}.score'


    def get_step_path(self, num_plan:int=None):
        return f'subject.{self.user.subject}.{self.user.level}.{self.user.plan_name}.{self.user.topic_name}'

class MongoDb:
    def __init__(self):
        self.client =  pymongo.MongoClient(COLLECTION_PATH)
        self.mydb = self.client[MONGODB_NAME]
        self.mycol = self.mydb[COLLECTION_NAME]


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
            # Update step
            pass

        return user_info

    def is_newUser(self, user_id:int)->bool:
        num_doc = self.mycol.count_documents({"user_id":user_id})
        if num_doc  == 0:
            logging.getLogger(SYSTEM_LOG).info(f"New user {user_id} ")
            return True

        return False
    
    def is_newSubject(self, user:User)->bool:
        try: # Incase user doesn't have path before
            myquery = {"user_id":user.id}
            doc = self.mycol.find(myquery)[0]
            exist_path = list(doc['subject'].keys())
            if user.subject not in exist_path:
                return True
        except:
            pass

        return False

    def is_newLevel(self, user:User)->bool:
        try: # Incase user doesn't have path before
            myquery = {"user_id":user.id}
            doc = self.mycol.find(myquery)[0]
            exist_path = list(doc['subject'][user.subject].keys())
            if user.level not in exist_path:
                return True
        except:
            pass

        return False

    def is_newPath(self, user:User)->bool:
        try: # Incase user doesn't have path before
            myquery = {"user_id":user.id}
            doc = self.mycol.find(myquery)[0]
            exist_path = list(doc['subject'][user.subject][user.level].keys())
            if user.plan_name not in exist_path:
                return True
        except:
            pass

        return False

    def get_topic_masteries(self, topic_name:str=None, total_masteries:dict=None)->dict:
        topic_masteries = {1:0, 2:1, 3:0} # connect to another database
        return  topic_masteries

    def prepare_flow_topic(self, total_masteries:dict=None)->list:
        flow_topic = ['topic_2', 'topic_1', 'topic_3']
        return flow_topic
    

    def write_to_DB(self, **kwargs):
        
        # Preprocess data
        raw_info = User(**kwargs)
        parsed_user = self.preprocess_userInfo(raw_info)

        # Formated data 
        data = Data_formater(parsed_user)

        user_indentify = {'user_id': data.user.id }

        if self.is_newUser(data.user.id): 
            self.mycol.insert_one(data.user_info())

        elif self.is_newSubject(data.user):
            prefix = 'subject'
            self.mycol.update_one(user_indentify, {'$set':data.subject_info(prefix)})

        elif self.is_newLevel(data.user):
            prefix = f'subject.{data.user.subject}'
            self.mycol.update_one(user_indentify, {'$set':data.level_info(prefix)})

        elif self.is_newPath(data.user):
            prefix = f'subject.{data.user.subject}.{data.user.level}'
            self.mycol.update_one(user_indentify, {'$set':data.path_info(prefix)})

        else:
            self.update_step(data)
            self.update_prev_score(data)

    def update_step(self, data:Data_formater):
        myquery = {"user_id":data.user.id}
        step_path = data.get_step_path()

        new_val = {"$push":{step_path : data.step_info()} }
        self.mycol.update_one(myquery, new_val)

    def update_prev_score(self, data:Data_formater):
        myquery = {"user_id":data.user.id}
        doc = self.mycol.find(myquery)[0]
        prev_step = len(doc['subject'][data.user.subject][data.user.level][data.user.plan_name][data.user.topic_name]) - 2 
        score_path = data.get_score_path(prev_step)

        new_val = {"$set":{score_path : data.user.prev_score} }
        self.mycol.update_one(myquery, new_val)


    def is_userExist(self, user:User):
        num_doc = self.mycol.count_documents({"user_id":user.id})
        if num_doc  == 1:
            return True
        else:
            if num_doc > 1:
                logging.getLogger(SYSTEM_LOG).error(f"Confict {num_doc} doc, user {user.id} ")
        return False




if __name__ == "__main__":
    database = MongoDb()

    #new user
    # database.write_to_DB(user_id="3", user_mail='3@gmail.com', subject="English", level=10, topic_masteries={"10":1, "11":1, "12":0}, action_index=1, action_id=1, prev_score=None, topic_name="topic_2", init_score = 4, flow_topic= ['topic_2', 'topic_1', 'topic_3'], plan_name='mocktest_1')

    # # new step
    # database.write_to_DB(user_id="3", user_mail='3@gmail.com', subject="English", level=10, topic_masteries={"10":2, "11":1, "12":0}, action_index=1, action_id=1, prev_score=5, topic_name="topic_2", init_score = None, flow_topic= None, plan_name='mocktest_1')

    # database.write_to_DB(user_id="3", user_mail='3@gmail.com', subject="English", level=10, topic_masteries={"10":3, "11":1, "12":0}, action_index=2, action_id=2, prev_score=6, topic_name="topic_2", init_score = None, flow_topic= None, plan_name='mocktest_1')

    # database.write_to_DB(user_id="3", user_mail='3@gmail.com', subject="English", level=10, topic_masteries={"10":4, "11":1, "12":0}, action_index=3, action_id=3, prev_score=7, topic_name="topic_2", init_score = None, flow_topic= None, plan_name='mocktest_1')


    # #new plan
    # database.write_to_DB(user_id="3", user_mail='3@gmail.com', subject="English", level=10, topic_masteries={"10":0, "11":0, "12":0}, action_index=1, action_id=1, prev_score=None, topic_name="topic_1", init_score = 4, flow_topic= ['topic_1', 'topic_2', 'topic_3'], plan_name='mocktest_2')

    # database.write_to_DB(user_id="3", user_mail='3@gmail.com', subject="English", level=10, topic_masteries={"10":1, "11":0, "12":0}, action_index=2, action_id=1, prev_score=5, topic_name="topic_1", init_score = None, flow_topic= None, plan_name='mocktest_2')

    # database.write_to_DB(user_id="3", user_mail='3@gmail.com', subject="English", level=10, topic_masteries={"10":2, "11":0, "12":0}, action_index=3, action_id=3, prev_score=10, topic_name="topic_1", init_score = None, flow_topic= None, plan_name='mocktest_2')


    # #new level
    # database.write_to_DB(user_id="3", user_mail='3@gmail.com', subject="English", level=11, topic_masteries={"10":1, "11":1, "12":0}, action_index=1, action_id=1, prev_score=None, topic_name="topic_2", init_score = 4, flow_topic= ['topic_2', 'topic_1', 'topic_3'], plan_name='mocktest_1')

    # database.write_to_DB(user_id="3", user_mail='3@gmail.com', subject="English", level=11, topic_masteries={"10":0, "11":0, "12":0}, action_index=1, action_id=1, prev_score=None, topic_name="topic_1", init_score = 4, flow_topic= ['topic_1', 'topic_2', 'topic_3'], plan_name='mocktest_2')

    # database.write_to_DB(user_id="3", user_mail='3@gmail.com', subject="English", level=11, topic_masteries={"10":1, "11":0, "12":0}, action_index=2, action_id=2, prev_score=5, topic_name="topic_1", init_score = 4, flow_topic= ['topic_1', 'topic_2', 'topic_3'], plan_name='mocktest_2')
    
    #new subject
    database.write_to_DB(user_id="3", user_mail='3@gmail.com', subject="Math", level=11, topic_masteries={"10":1, "11":1, "12":0}, action_index=1, action_id=1, prev_score=None, topic_name="topic_2", init_score = 4, flow_topic= ['topic_2', 'topic_1', 'topic_3'], plan_name='mocktest_1')

    database.write_to_DB(user_id="3", user_mail='3@gmail.com', subject="Math", level=11, topic_masteries={"10":0, "11":0, "12":0}, action_index=1, action_id=1, prev_score=None, topic_name="topic_1", init_score = 4, flow_topic= ['topic_1', 'topic_2', 'topic_3'], plan_name='mocktest_2')

    database.write_to_DB(user_id="3", user_mail='3@gmail.com', subject="Math", level=11, topic_masteries={"10":1, "11":0, "12":0}, action_index=2, action_id=2, prev_score=5, topic_name="topic_1", init_score = 4, flow_topic= ['topic_1', 'topic_2', 'topic_3'], plan_name='mocktest_2')