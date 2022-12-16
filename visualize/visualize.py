import numpy as np
import pandas as pd
import seaborn as sns

import sys

sys.path.append('/home/ubuntu/DQN/')
from dqn.variables import *
from dqn.database import MongoDb

class User_Step():
    def __init__(self, action_re:int, action_ID:int, score:int, reward:int) -> None:
        self.action_recommend:int = action_re
        self.action_ID:int = action_ID
        self.score:int = score
        self.reward:int = reward

class User_Level():
    def __init__(self) -> None:
        self.init_masteries_byTopic:dict = {}
        self.num_mocktest:int = 0
        # self.num_done:int = 0
        # self.num_inprocess:list = 0 
        self.list_mocktest:list = None
        self.step:dict = {}

class User_Category():
    def __init__(self, name) -> None:
        self.c_name = name
        self.level:dict = {"10":User_Level(), "11":User_Level(), "12":User_Level()}

class User_Manager():
    def __init__(self, raw_data:dict) -> None:
        self.id:str = raw_data[USER_ID]
        self.gmail:str = raw_data[USER_GMAIL]
        self.categories:dict = raw_data[CATEGORY]

        # self.categories = {}

        self.Grammar = User_Category(name = GRAMMAR[C_REAL_NAME])
        self.Vocabulary = User_Category(name = VOVABULARY[C_REAL_NAME])

        self.Algebra = User_Category(name = ALGEBRA[C_REAL_NAME])
        self.Geometry = User_Category(name = GEOMETRY[C_REAL_NAME])
        self.Probability_statistics = User_Category(name = PROBABILITY[C_REAL_NAME])
        self.Analysis = User_Category(name = ANALYSIS[C_REAL_NAME])

        self.prerocess_data()

    def prerocess_data(self):
        '''
        Format data from db to class type
        '''
        # Get exist category
        for category_name in self.categories:
            # Get exist level
            for level in self.categories[category_name]:                
                level_doc:dict = self.categories[category_name][level]

                # Select manager field to save data
                _, level_manager = self.get_field_manager(category_name, level)

                # Handle num_mocktest
                level_manager.num_mocktest:int = len(level_doc.keys())

                # Get exist mocktest
                for mocktest_name in level_doc:
                    mocktest_doc:dict = level_doc[mocktest_name]

                    # Handle init_masteries_byTopic
                    level_manager.init_masteries_byTopic:dict =  self.format_initMasteries_byTopic(mocktest_doc)

                    # Handle step
                    level_manager.step = mocktest_doc[FLOW] 



    def format_initMasteries_byTopic(self, mocktest_doc:dict)->dict :
        '''
        Get total masteries and format by topic
        Ex: total_masteies = {'a':0, 'b':1, 'c':0} format by topic 1&2
        => {1:{'a':0, 'b':1}, 2:{'c':0}}
        '''
        pool:dict = mocktest_doc[POOL]
        total_topic:dict = mocktest_doc[TOTAL_TOPIC]
        total_masteries:dict = mocktest_doc[TOTAL_MASTERIES]

        init_masteries_byTopic = {}
        for lpd in total_masteries:
            for topic_name, ldps_inTopic in total_topic.items():
                if lpd in ldps_inTopic:
                    val = 0 if lpd in pool else 1 
                    # Append  value
                    if topic_name not in init_masteries_byTopic: # first value
                        init_masteries_byTopic[topic_name] = {lpd:val}
                    else:
                        init_masteries_byTopic[topic_name].update({lpd:val})

                    break
        
        return init_masteries_byTopic
    
    def get_field_manager(self, category:str, level:str):
        '''
        Return manager and level_manager
        '''
        if category in self.Grammar.c_name:
            category_manager = self.Grammar        
        elif category in self.Vocabulary.c_name:
            category_manager = self.Vocabulary            
        elif category in self.Algebra.c_name:
            category_manager = self.Algebra            
        elif category in self.Geometry.c_name:
            category_manager = self.Geometry            
        elif category in self.Probability_statistics.c_name:
            category_manager = self.Probability_statistics            
        elif category in self.Analysis.c_name:
            category_manager = self.Analysis  

        return category_manager, category_manager.level[level]          


class Visualize():
    def __init__(self, output_file:str=None):
        self.database = MongoDb()
        self.output_file = output_file

    
    def repare_data(self):
        '''
        Preprocess data from database to class structure
        '''
        # Loop all user
        for user_doc in self.database.get_all_userInfor():
            # Init user_manager
            user_manager = User_Manager(user_doc)

            # Select special category and level
            category_manager, level_manager = user_manager.get_field_manager(category= GRAMMAR[C_REAL_NAME], level= '12')
            print(1)

            

    


v = Visualize()
v.repare_data()




