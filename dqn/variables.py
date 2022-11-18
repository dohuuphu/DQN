# ====== PARAMETER ======

# Model
STATE_ACTION_SPACE = 80
MAX_STEP_EPISODE = 200
STEP_UPDATE_TARGETR_MODEL = 500
NUM_EPISODE_TO_SAVE_MODEL = 100
REWARD_LENGTH = True
RELATION = True


# Topic table
NUM_TOPIC = 300000


# Train
BATCH_SIZE = 16 #128
MIN_REPLAY_SIZE = 32#500 
train_episodes = 10000

NUM_EPISODE_TRAIN = 2
RETRAIN = False
MODEL_SAVE = F'action_{STATE_ACTION_SPACE}'
MODEL_RETRAIN =  '/home/hoangtv/phudh/DQN/weight/action_80_relation_length'
EPISODE_SAVE = 100


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
MONGODB_NAME = "AI"
COLLECTION_USER = "User"
COLLECTION_LESSON = "Content_dev"   # Content_dev, Content
COLLECTION_LESSON_ID = "Content_dev_ID" # Content_ID, Content_dev_ID



# Fixed name
DONE = 'done'
INPROCESS = 'inprocess'
PENDING = 'pending'
ENGLISH = "English"

MATH = "Math"
ALGEBRA = "Algebra"
GEOMETRY = "Geometry"
PROBABILITY= "Probability_statistics"
ANALYSIS = "Analysis"


# Cache file
ENGLISH_R_BUFFER = "cache/English_relay_buffer.pickle"
ALGEBRA_R_BUFFER = "cache/Algebra_relay_buffer.pickle"
GEOMETRY_R_BUFFER = "cache/Geometry_relay_buffer.pickle"
PROBABILITY_R_BUFFER= "cache/Probability_statistics_relay_buffer.pickle"
ANALYSIS_R_BUFFER = "cache/Analysis_relay_buffer.pickle"

