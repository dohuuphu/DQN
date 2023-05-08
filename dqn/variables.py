# ====== PARAMETER ======

# Model
STATE_ACTION_SPACE = 80
MAX_STEP_EPISODE = 200
STEP_TRAIN = 4
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
# COLLECTION_USER = "User_test"
COLLECTION_LESSON = "Content_dev_test"   # Content_dev, Content
COLLECTION_LESSON_ID = "Content_dev_ID_test" # Content_ID, Content_dev_ID




# Fixed name
DONE = 'done'
INPROCESS = 'inprocess'
PENDING = 'pending'

# CATEGORY
C_VISUALE_NAME = 0
C_REAL_NAME = 1
ENGLISH = "English"
GRAMMAR = ["Grammar", "2"]
VOVABULARY = ["Vocabulary", "1"]

MATHTEST = "MathTest"
MATHCEREBRY = ["Math_Cerebry", "3"]

MATH = "Math"
ALGEBRA = ["Algebra", "000"]
GEOMETRY = ["Geometry", "000"]
PROBABILITY= ["Probability_statistics", "000"]
ANALYSIS = ["Analysis", "000"]
TOTAL_SUBJECT = [MATH, ENGLISH, MATHTEST]
TOTAL_LEVEL = [10, 11,12]



# Name in database
USER_ID = "user_id"
USER_GMAIL = "user_gmail"
CATEGORY = "category"
STATUS = "status"
POOL = "pool"
TOTAL_TOPIC = "total_topic"
TOTAL_MASTERIES = "total_masteries"
INIT_SCORE = "init_score"
FLOW = "flow"
ACTION_RECOMMEND = "action_recommend"
ACTION_ID = "action_ID"
SCORE = "score"
REWARD = "reward"
MASTERIES = "masteries"

# Cache file
GRAMMAR_R_BUFFER = "cache/Grammar_relay_buffer.pickle"
VOCABULARY_R_BUFFER = "cache/Vocabulary_relay_buffer.pickle"
ALGEBRA_R_BUFFER = "cache/Algebra_relay_buffer.pickle"
GEOMETRY_R_BUFFER = "cache/Geometry_relay_buffer.pickle"
PROBABILITY_R_BUFFER= "cache/Probability_statistics_relay_buffer.pickle"
ANALYSIS_R_BUFFER = "cache/Analysis_relay_buffer.pickle"
TOTAL_R_BUFFER = "cache/Total_relay_buffer.pickle"

# Visualize
V_USER_ID = 'user_id'
V_TOPIC = 'topic'
V_LESSON = 'lesson'
V_VALUE = 'value'
V_SCORE = 'score'
V_REPEAT = 'repeat'

class MathTest():
    name = 'MathTest'
    total = 'Total'
    total_id = '1'

class English():
    name = 'English'
    grammar = 'Grammar'
    grammar_id = '2'

    vocabulary = 'Vocabulary'
    vocabulary_id = '1'

class Math():
    name = 'Math'
    algebra = 'Algebra'
    algebra_id = '000'

    geometry = 'Geometry'
    geometry_id = '000'

    probability_statistics = 'Probability_statistics'
    probability_statistics_id = '000'

    analysis = 'Analysis'
    analysis_id = '000'

class Graph():
    analysis_score_in_lesson = 'analysis_score_in_lesson'
    initial_masteries_analysis = 'initial_masteries_analysis'
    analysis_repeated_in_lesson = 'analysis_repeated_in_lesson'
