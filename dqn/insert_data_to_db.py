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
        self.mycol = self.mydb[COLLECTION_DATA]

    def insert_lesson_to_mongo(self, data):
        self.mycol.insert_one(data)

class GetReturnForBackend():
    def __init__(self) -> None:
        self.url = "https://api.tuhoconline.org/ai/lessons"
        self.headers = {"X-Authenticated-User":"kyons-ai-api-key"}
        self.respone = requests.request("GET",  self.url, headers=self.headers)
    def normalize_input(self):
        response_json = json.loads(self.respone.text)
        df = pd.DataFrame(response_json)
        print(df.shape)
        df = df[df['content'].notnull()]
        df = df[df['content']!=''] 
        topics = list(df['topic'].unique())
        topics = sorted(topics)
        lesson = {"10":{
                        "Grammar":dict(),
                        "Vocabulary":dict()
                        }, 
                        "11":{
                        "Grammar":dict(),
                        "Vocabulary":dict()
                        }, 
                        "12":{
                        "Grammar":dict(),
                        "Vocabulary":dict()
                    }}
        for index, topic in enumerate(topics):
            list_lp_10 = dict()
            list_lp_11 = dict()
            list_lp_12 = dict()
            df_temp = df.loc[(df['topic'] == topic)].values
            for i in range(len(df_temp)):
                if 10 in df_temp[i][5]:
                    list_lp_10[str(df_temp[i][0])]=1
                    lesson['10'][df_temp[i][3]].update({topic: list_lp_10})
                if 11 in df_temp[i][5]:
                    list_lp_11[str(df_temp[i][0])]=1
                    lesson['11'][df_temp[i][3]].update({topic: list_lp_11})
                if 12 in df_temp[i][5]:
                    list_lp_12[str(df_temp[i][0])]=1
                    lesson['12'][df_temp[i][3]].update({topic: list_lp_12})
        return {"English":lesson}


if __name__ == "__main__":
    database = DBLesson()
    lesson_from_api = GetReturnForBackend()
    data = lesson_from_api.normalize_input()
    database.insert_lesson_to_mongo(data)