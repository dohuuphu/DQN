# ====== PARAMETER ======

# Model
STATE_ACTION_SPACE = 80
MAX_STEP_EPISODE = 200
STEP_UPDATE_TARGETR_MODEL = 500
NUM_EPISODE_TO_SAVE_MODEL = 100
REWARD_LENGTH = True
RELATION = True


# Topic table
NUM_TOPIC = 30000


# Train
RETRAIN = False
MODEL_SAVE = F'action_{STATE_ACTION_SPACE}' + ('_relation' if RELATION else '' )+ ('_length' if REWARD_LENGTH else '') + ('_test')
MODEL_RETRAIN =  '/home/hoangtv/phudh/DQN/weight/action_80_relation_length'

train_episodes = 10000

# Test
MODEL_INFERENCE = '/home/ubuntu/DQN/weight/action_30_relation_length_2feature'

# Logging
SYSTEM_LOG = 'system_logger'
RECOMMEND_LOG = 'recommender_logger'
CHECKDONE_LOG = 'checkdone_logger'

SYSTEM_PATH = 'logs/system.log'
RECOMMEND_PATH = 'logs/recommender.log'
CHECKDONE_PATH = 'logs/check_done.log'



#Config database
COLLECTION_PATH = "mongodb://localhost:27017"
MONGODB_NAME = "AI_Kyons_new"
COLLECTION_USER = "data_beta"
COLLECTION_LESSON = "content_data"
COLLECTION_LESSON_ID = "hashing_data"



# Fixed name
DONE = 'done'
INPROCESS = 'inprocess'
PENDING = 'pending'
ENGLISH = "English"
ALGEBRA = "Algebra"
GEOMETRY = "Geometry"
