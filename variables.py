# PARAMETER
STATE_ACTION_SPACE = 100
REWARD_LENGTH = True
RELATION = True

# Train
RETRAIN = False
MODEL_SAVE = F'action_{STATE_ACTION_SPACE}' + ('_relation' if RELATION else '' )+ ('_length' if REWARD_LENGTH else '')+"1024"
MODEL_RETRAIN =  '/home/hoangtv/phudh/DQN/weight/action_80_relation_length'

train_episodes = 10000

# Test
MODEL_INFERENCE = '/home/hoangtv/phudh/DQN/weight/action_100_relation_length1024'
