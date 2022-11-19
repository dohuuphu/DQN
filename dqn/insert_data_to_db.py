import logging
import pymongo
import requests
from dqn.variables import *
import json
import pandas as pd

class DBLesson:
    def __init__(self):
        self.client =  pymongo.MongoClient(COLLECTION_PATH)
        self.mydb = self.client[MONGODB_NAME]
        self.mycol = self.mydb[COLLECTION_LESSON]

    def insert_lesson_to_mongo(self, data):
        self.mycol.insert_one(data)

class DB_Backend():
    def __init__(self, method:str, url:str, header:dict, json:dict) -> None:
        # self.url = url  #"https://api.tuhoconline.org/ai/lessons"
        # self.headers = {key:val} #{"X-Authenticated-User":"kyons-ai-api-key"}
        self.current_DB = DBLesson()
        self.subject_level = json.program.split(" ")
        self.subject = self.subject_level[0]
        self.level = self.subject_level[1]
        self.respone = requests.request(method, url, headers=header, json={"program":json.program, "level": json.level})
    def normalize_input(self):
        response_json = json.loads(self.respone.text)
        df = pd.DataFrame(response_json)
        # print(df)
        df = df[df['content'].notnull()]
        df = df[df['content']!=''] 
        topics = list(df['topic_id'].unique())
        topics = sorted(topics)
        category_id = list(df['category_id'].unique())
        category_id = list(map(str,category_id))
        lesson = dict.fromkeys(category_id, dict())
        # print(lesson)
        for index, topic in enumerate(topics):
            list_lp = dict()
            df_temp = df.loc[(df['topic_id'] == topic)].values
            for i in range(len(df_temp)):
                list_lp[str(df_temp[i][0])]=1
                lesson[str(df_temp[i][3])].update({str(topic): list_lp})
            
        return lesson
    
    def modify_current_db(self):
        modify_db = self.normalize_input()
        self.current_DB.mycol.update_many({
        f"{str(self.subject)}": {"$exists": True}
        },
        {
        "$set": {
            f"{str(self.subject)}.{str(self.level)}": modify_db
        }
        })



if __name__ == "__main__":
    database = DBLesson()
    lesson_from_api = DB_Backend()
    data = lesson_from_api.normalize_input()
    database.insert_lesson_to_mongo(data)