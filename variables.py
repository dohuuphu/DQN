# PARAMETER
STATE_ACTION_SPACE = 30
REWARD_LENGTH = True
RELATION = True

# Train
RETRAIN = False
MODEL_SAVE = F'action_{STATE_ACTION_SPACE}' + ('_relation' if RELATION else '' )+ ('_length' if REWARD_LENGTH else '')
MODEL_RETRAIN = ''

train_episodes = 10000

# Test
MODEL_INFERENCE = ''
