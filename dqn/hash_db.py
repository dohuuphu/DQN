import json
import pymongo
from dqn.variables import *
#create dict to map mongodb to json
class create_dict(dict): 
  
    # __init__ function 
    def __init__(self): 
        self = dict() 
          
    # Function to add key:value 
    def add(self, key, value): 
        self[key] = value 

class HashDB:
    def __init__(self):
        self.client =  pymongo.MongoClient(COLLECTION_PATH)
        self.mydb = self.client[MONGODB_NAME]
        self.coldb = self.mydb[COLLECTION_LESSON]
        self.colhash = self.mydb[COLLECTION_LESSON_ID]
        #map mongodb to json
        self.mydict = create_dict()
        i = 1
        for y in self.coldb.find():
            self.mydict.add(i, (y))
            i = i+1
    def add_hash_db(self):
        # create hash field
        hash_subject = 0
        hash_level = 0    
        hash_category = "0"
        hash_topic = "000"
        dict_subject = dict()
        dict_level = dict()
        dict_category = dict()
        dict_topic = dict()
        hash_dict = dict()
        #update hash each field
        # for each key from top to bottom subject ->level->category->topic
        for i,v in self.mydict.items():
            subject = list(v.keys())[-1]
            for level, category_topic in self.mydict[i][subject].items():
                for category, topic_lp in self.mydict[i][subject][level].items():
                    for topic, lp in self.mydict[i][subject][level][category].items():
                        if topic not in dict_topic:
                            hash_topic = str(int(hash_topic)+1).zfill(len(hash_topic))
                            dict_topic.update({topic:hash_topic})
                        else:
                            hash_topic = dict_topic[topic]
                    if category not in dict_category:
                        hash_category = str(int(hash_category)+1).zfill(len(hash_category))
                        dict_category.update({category:hash_category})
                    else:
                        hash_category = dict_category[category]
                if level not in dict_level:
                    hash_level = len(dict_level) + 1
                    dict_level.update({level:hash_level})
                else:
                    hash_level = dict_level[level]
            if subject not in dict_subject:
                hash_subject = len(dict_subject)+1
                dict_subject.update({subject:hash_subject})
            else:
                hash_subject = dict_subject[subject]
            hash_category = "0"
            hash_topic = "000"
        for i,v in self.mydict.items():
            subject = list(v.keys())[-1]
            for level, category_topic in self.mydict[i][subject].items():
                for category, topic_lp in self.mydict[i][subject][level].items():
                    for topic, lp in self.mydict[i][subject][level][category].items():
                        hash_dict.update({f"{subject}_{level}_{topic}":f"{dict_subject[subject]}{dict_level[level]}{dict_topic[topic]}"})
        #drop current hash DB
        self.colhash.drop()
        #insert new hash DB
        self.colhash.insert_one(hash_dict)

if __name__ == "__main__":
    hash_DB = HashDB()
    hash_DB.add_hash_db()
