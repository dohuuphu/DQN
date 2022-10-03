# PARAMETER
STATE_ACTION_SPACE = 30
MAX_STEP_EPISODE = 200
REWARD_LENGTH = True
RELATION = True


# Topic table
NUM_TOPIC = 2


# Train
RETRAIN = False
MODEL_SAVE = F'action_{STATE_ACTION_SPACE}' + ('_relation' if RELATION else '' )+ ('_length' if REWARD_LENGTH else '')
MODEL_RETRAIN =  '/home/hoangtv/phudh/DQN/weight/action_80_relation_length'

train_episodes = 10000

# Test
MODEL_INFERENCE = '/mnt/d/src/RL/DQN/weight/action_30_relation_length'
