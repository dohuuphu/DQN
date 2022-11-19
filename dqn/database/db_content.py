from dqn.variables import *

class DB_Content():
    def __init__(self, database) -> None:
        self.db_content = database[COLLECTION_LESSON]
        self.data = self.get_content()


    def get_content(self):
        data = {subject: {} for subject in TOTAL_SUBJECT}
        for subject in data:
            myquery = {"subject":subject}
            data[subject].update(self.db_content.find(myquery)[0])
        
        return data
