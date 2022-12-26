import numpy as np
import pandas as pd
import seaborn as sns
from pandas import json_normalize
import sys
import os
sys.path.append('/home/ubuntu/DQN/')
from dqn.variables import *
from dqn.database import MongoDb
from collections import Counter


class User_Step:
    def __init__(self, action_re:int, action_ID:int, score:int, reward:int) -> None:
        self.action_recommend:int = action_re
        self.action_ID:int = action_ID
        self.score:int = score
        self.reward:int = reward

class User_Level:
    def __init__(self) -> None:
        self.init_masteries_byTopic:dict = {}
        self.num_mocktest:int = 0
        # self.num_done:int = 0
        # self.num_inprocess:list = 0 
        self.list_mocktest:list = None
        self.step:dict = {}

class User_Category:
    def __init__(self, name) -> None:
        self.c_name = name
        self.level:dict = {"10":User_Level(), "11":User_Level(), "12":User_Level()}

class User_Manager:
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

        # init_masteries_byTopic = {}
        # for lpd in total_masteries:
        #     for topic_name, ldps_inTopic in total_topic.items():
        #         if lpd in ldps_inTopic:
        #             val = 0 if lpd in pool else 1 
        #             # Append  value
        #             if topic_name not in init_masteries_byTopic: # first value
        #                 init_masteries_byTopic[topic_name] = {lpd:val}
        #             else:
        #                 init_masteries_byTopic[topic_name].update({lpd:val})

        #             break
        
        init_masteries_byTopic = []
        for lpd in total_masteries:
            item = {
                V_TOPIC : '',
                V_LESSON : '',
                V_VALUE : ''
            }
            for topic_name, ldps_inTopic in total_topic.items():
                if lpd in ldps_inTopic:
                    val = 0 if lpd in pool else 1 
                    item[V_TOPIC] = topic_name
                    item[V_LESSON] = lpd
                    item[V_VALUE] = val

                    break
            init_masteries_byTopic.append(item)
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


class Visualize:
    def __init__(self, output_file:str=None):
        self.database = MongoDb()
        self.output_file = output_file

    
    def prepare_data(self, category, level):
        '''
        Preprocess data from database to class structure
        '''
        total_init_masteries = []
        total_step = []
        # Loop all user
        for user_doc in self.database.get_all_userInfor():
            # Init user_manager
            user_manager = User_Manager(user_doc)

            # Select special category and level
            category_manager, level_manager = user_manager.get_field_manager(category= category, level= level)
            # total_init_masteries.append(level_manager.init_masteries_byTopic)
            total_init_masteries += level_manager.init_masteries_byTopic
            total_step.append(level_manager.step)
            # print(1)
        # return category_manager, level_manager
        return total_init_masteries, total_step

    def analysis_initial_masteries(self, category, level):
        '''
        plot chart display number of 0 and 1 value in lesson of each topic
        '''
        path = f'./{level}/initial_masteries_analysis/{category}'
        if not os.path.exists(path):
            os.makedirs(path)
        total_init_masteries,_ = self.prepare_data(category, level)
        df_total_init_masteries = json_normalize(total_init_masteries)
        list_topic = df_total_init_masteries['topic'].unique()
        list_topic = list_topic[(list_topic!='')]
        unique_topic_ls = []
        for topic in list_topic:
            unique_topic_ls.append(df_total_init_masteries[df_total_init_masteries['topic']==topic])
        for l in range(len(unique_topic_ls)):
            new_ls_0 = []
            new_ls_1 = []
            for i in unique_topic_ls[l]['lesson'].unique():
                new_ls_0.append(unique_topic_ls[l][(unique_topic_ls[l]['lesson']==i) & (unique_topic_ls[l]['value']==0)])
                new_ls_1.append(unique_topic_ls[l][(unique_topic_ls[l]['lesson']==i) & (unique_topic_ls[l]['value']==1)])
            hist_df = pd.DataFrame()
            hist_df['lesson'] = unique_topic_ls[l]['lesson'].unique()
            hist_df['0'] = [len(value) for value in new_ls_0]
            hist_df['1'] = [len(value) for value in new_ls_1]
            new_df = pd.DataFrame(columns=['lesson', 'amount', 'value'])
            for i in range(len(hist_df)):
                new_df = new_df.append({'lesson':hist_df.iloc[i]['lesson'], 'value':'0', 'amount':hist_df.iloc[i]['0']},ignore_index=True)
                new_df = new_df.append({'lesson':hist_df.iloc[i]['lesson'], 'value':'1', 'amount':hist_df.iloc[i]['1']},ignore_index=True)
            new_df.to_csv(f'{path}/topic_{list_topic[l]}.csv', index=False)
            ax = hist_df.plot(x="lesson", y=['0','1'], kind="bar")
            ax.figure.savefig(f'{path}/topic_{list_topic[l]}.png')

    
    def analysis_score_follow_step(self, category, level):
        '''
        analysis score 
        '''
        _, total_step =  self.prepare_data(category, level)
        score_byTopic = []
        for step in total_step:
            for topic, value in step.items():
                for i in range(len(value)):
                    item = {
                        V_TOPIC : '',
                        V_LESSON : '',
                        V_SCORE : ''
                        }
                    if value[i]['score'] is not None:
                        # print(topic)
                        item[V_TOPIC] = topic
                        item[V_LESSON] = value[i]['action_ID']
                        item[V_SCORE] = value[i]['score']
                    score_byTopic.append(item)

        df_score_by_topic = json_normalize(score_byTopic)
        df_score_by_topic.replace('', np.nan, inplace=True)
        df_score_by_topic.dropna()
        lesson_by_topic = []
        ls_unique_lesson = []

        for topic in df_score_by_topic['topic'].unique():
            lesson_by_topic.append(df_score_by_topic[df_score_by_topic['topic']==topic])
        for unique_topic in lesson_by_topic:
            for unique_lesson in unique_topic['lesson'].unique():
                ls_unique_lesson.append(unique_topic[unique_topic['lesson']==unique_lesson])
        for i in range(len(ls_unique_lesson)):
            topic = ls_unique_lesson[i]['topic'].unique()[0]
            path =  f'./{level}/analysis_score_in_lesson/{category}/{topic}/'
            if not os.path.exists(path):
                os.makedirs(path)
            df = pd.DataFrame(columns=['lesson', 'score', 'value'])
            print(ls_unique_lesson[i]['topic'].unique()[0], ls_unique_lesson[i]['lesson'].unique()[0]) 
            sum_n_repeat = [cnt for val, cnt in ls_unique_lesson[i]['score'].value_counts().items()]
            sum_n_repeat = sum(sum_n_repeat)
            for val, cnt in ls_unique_lesson[i]['score'].value_counts().items():
                df = df.append({'lesson':ls_unique_lesson[i]['lesson'].unique()[0], 'score':str(val), 'value':cnt/sum_n_repeat}, ignore_index=True)
            lesson = ls_unique_lesson[i]['lesson'].unique()[0]
            # print(f'{path}/lesson_{lesson}.csv')
            ax = df.plot(x="score", y=['value'], kind="bar")
            ax.figure.savefig(f'{path}/lesson_{lesson}.png')
            df.to_csv(f'{path}/lesson_{lesson}.csv', index=False)


    def analysis_repeated_lesson(self, category, level):
        '''
        analysis repeated lesson
        '''
        _, total_step =  self.prepare_data(category, level)
        total_repeated = []
        for step in total_step:
            for topic, value in step.items():
                item = {
                        V_TOPIC : '',
                        V_LESSON : '',
                        V_REPEAT : ''
                        }
                repeated_ls = []
                for i in range(len(value)):
                    repeated_ls.append(value[i]['action_ID'])
                repeated_dict = dict(Counter(repeated_ls))
                for lesson, n_repeat in repeated_dict.items():
                    item[V_TOPIC] = topic
                    item[V_LESSON] = lesson
                    item[V_REPEAT] = n_repeat
                    total_repeated.append(item)
        df_total_repeated = json_normalize(total_repeated)                
        df_total_repeated.replace('', np.nan, inplace=True)
        df_total_repeated.dropna()
        lesson_by_topic = []
        ls_unique_lesson = []  
        for topic in df_total_repeated['topic'].unique():
            lesson_by_topic.append(df_total_repeated[df_total_repeated['topic']==topic])
        for unique_topic in lesson_by_topic:
            for unique_lesson in unique_topic['lesson'].unique():
                ls_unique_lesson.append(unique_topic[unique_topic['lesson']==unique_lesson])
        for i in range(len(ls_unique_lesson)):
            topic = ls_unique_lesson[i]['topic'].unique()[0]
            path =  f'./{level}/analysis_repeated_in_lesson/{category}/{topic}/'
            if not os.path.exists(path):
                os.makedirs(path)
            df = pd.DataFrame(columns=['lesson', 'repeat', 'n_repeat'])
            print(ls_unique_lesson[i]['topic'].unique()[0], ls_unique_lesson[i]['lesson'].unique()[0]) 
            print(ls_unique_lesson[i]['repeat'].value_counts())
            sum_n_repeat = [cnt for val, cnt in ls_unique_lesson[i]['repeat'].value_counts().items()]
            sum_n_repeat = sum(sum_n_repeat)
            for val, cnt in ls_unique_lesson[i]['repeat'].value_counts().items():
                df = df.append({'lesson':ls_unique_lesson[i]['lesson'].unique()[0], 'repeat':str(int(val)), 'n_repeat':cnt/sum_n_repeat}, ignore_index=True)
            lesson = ls_unique_lesson[i]['lesson'].unique()[0]
            ax = df.plot(x="repeat", y=['n_repeat'], kind="bar")
            ax.figure.savefig(f'{path}/lesson_{lesson}.png')
            df.to_csv(f'{path}/lesson_{lesson}.csv', index=False)

        def analysis_score_step_by_step(self, category, level):
                '''
                analysis score step by step
                '''
                _, total_step =  self.prepare_data(category, level)
                score_byTopic = []
                for step in total_step:
                    for topic, value in step.items():
                        for i in range(len(value)):
                            item = {
                                V_TOPIC : '',
                                V_LESSON : '',
                                V_SCORE : ''
                                }
                            if value[i]['score'] is not None:
                                # print(topic)
                                item[V_TOPIC] = topic
                                item[V_LESSON] = value[i]['action_ID']
                                item[V_SCORE] = value[i]['score']
                            score_byTopic.append(item)
                




if __name__ == "__main__":
    test = Visualize()
    test.analysis_score_follow_step(category= GRAMMAR[C_REAL_NAME], level= '11')            

    


# v = Visualize()
# v.prepare_data()




