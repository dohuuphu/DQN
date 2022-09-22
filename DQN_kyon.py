import tensorflow as tf

import numpy as np
from tensorflow import keras
import gc

import keras.backend as K
from collections import deque
import datetime
import random
from env_kyon import SimStudent
from variables import *
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"   

RANDOM_SEED = 5
tf.random.set_seed(RANDOM_SEED)

# env = gym.make('CartPole-v1')
# env.seed(RANDOM_SEED)
# np.random.seed(RANDOM_SEED)

# print("Action Space: {}".format(env.action_space))
# print("State space: {}".format(env.observation_space))

env = SimStudent()

TOPIC_TABLE = keras.layers.Embedding(NUM_TOPIC, STATE_ACTION_SPACE, input_length=1)

def agent(state_shape, action_shape):
    """ The agent maps X-states to Y-actions
    e.g. The neural network output is [.1, .7, .1, .3]
    The highest value 0.7 is the Q-Value.
    The index of the highest action (0.7) is action #1.
    """
    learning_rate = 0.001
    init = tf.keras.initializers.HeUniform()
    model = keras.Sequential()

    model.add(keras.layers.Conv1D(512, 2, activation='relu',input_shape=state_shape[0:]))
    # model.add(keras.layers.Dense(512, input_shape=state_shape, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(512, activation='relu', kernel_initializer=init))

    model.add(keras.layers.Dense(256, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(action_shape, activation=tf.keras.activations.softmax, kernel_initializer=init))
    model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])

    return model

def get_qs(model, state, step):
    return model.predict(state.reshape([1, state.shape[0]]))[0]

def train(env, replay_memory, model, target_model, done, train_summary_writer, episode):
    learning_rate = 0.7 # Learning rate
    discount_factor = 0.618

    MIN_REPLAY_SIZE = 1000
    if len(replay_memory) < MIN_REPLAY_SIZE:
        return

    batch_size = 64 * 2
    mini_batch = random.sample(replay_memory, batch_size)
    current_states = np.array([transition[0] for transition in mini_batch])
    current_qs_list = model.predict(current_states,verbose = 0)
    new_current_states = np.array([transition[3] for transition in mini_batch])
    future_qs_list = target_model.predict(new_current_states,verbose = 0)
    K.clear_session()


    X = []
    Y = []
    for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
        if not done:
            max_future_q = reward + discount_factor * np.max(future_qs_list[index])
        else:
            max_future_q = reward

        current_qs = current_qs_list[index][0]
        current_qs[action] = (1 - learning_rate) * current_qs[action] + learning_rate * max_future_q

        X.append(observation)
        Y.append(current_qs)
        
    his = model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True, workers=1)
    # print(list(his.history.values())[0])
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', his.history['loss'][0], step=episode)
        tf.summary.scalar('acc', his.history['accuracy'][0], step=episode)

    
def main():
    epsilon = 1 # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
    max_epsilon = 1 # You can't explore more than 100% of the time
    min_epsilon = 0.01 # At a minimum, we'll always explore 1% of the time
    decay = 0.01
    

    # 1. Initialize the Target and Main models
    # Main Model (updated every 4 steps)
    if not RETRAIN:
        model = agent(( 2, STATE_ACTION_SPACE), STATE_ACTION_SPACE)
    else:
        model = keras.models.load_model(MODEL_RETRAIN)
    # Target Model (updated every 100 steps)
    target_model = agent(( 2, STATE_ACTION_SPACE), STATE_ACTION_SPACE)
    target_model.set_weights(model.get_weights())

    # Init logger
    train_log_dir =  "logs/" + MODEL_SAVE + '/reward'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    replay_memory = deque(maxlen=1_500)

    target_update_counter = 0

    # X = states, y = actions
    X = []
    y = []

    steps_to_update_target_model = 0

    for episode in range(train_episodes):
        total_training_rewards = 0
        step_per_episode = 0
        (_, observation), zero_list = env.reset()
        
        total_zero = (observation == 0.0).sum()

        done = False
        old_action = ''
        while not done:
            steps_to_update_target_model += 1
            step_per_episode += 1

            random_number = np.random.rand()
            random_topic = np.array([0])#np.random.randint(0, NUM_TOPIC, size=(1,))
            topic_feature = TOPIC_TABLE(random_topic)

            # concat topic_feature to observation
            observation_concate = keras.layers.Concatenate(axis=0)([observation.reshape([1, observation.shape[0]]), topic_feature])
            # observation_concate = keras.layers.Concatenate(axis=0)([observation, topic_feature])

            # observation_concate = tf.expand_dims(observation_concate, axis=0)
            # 2. Explore using the Epsilon Greedy Exploration Strategy
            if random_number == epsilon:
                # Explore
                action = random.randint(0, len(observation) - 1)
            else:
                # Exploit best known action
                # model dims are (batch, env.observation_space.n)
                # encoded = observation
                # encoded_reshaped = encoded.reshape([1, encoded.shape[0]])
                predicted = model.predict(tf.expand_dims(observation_concate, axis=0), verbose = 0).flatten()
                K.clear_session()
                action = np.argmax(predicted)

                # random action
                if old_action == action:
                    if random.randint(0,1):
                        action = random.randint(0, len(observation) - 1)
                            
                old_action = action
            (new_observation_concat, new_observation), reward, done, info = env.step(action, zero_list, int(random_topic), topic_feature)
            replay_memory.append([observation_concate, action, reward, new_observation_concat, done])

            # 3. Update the Main Network using the Bellman Equation
            if steps_to_update_target_model % 4 == 0 or done:
                train(env, replay_memory, model, target_model, done, train_summary_writer, episode)

            observation = new_observation
            total_training_rewards += reward
    
            # K.clear_session()
            gc.collect()
            if done:
                print('Total training rewards: {} after n steps = {} ({}) with final reward = {}'.format(total_training_rewards, step_per_episode, episode, reward))
                with train_summary_writer.as_default():
                    tf.summary.scalar('reward', total_training_rewards, step=episode)
                    tf.summary.scalar('deviation', (step_per_episode - total_zero)/total_zero , step=episode)

                total_training_rewards = 0

                if steps_to_update_target_model >= 500:
                    print('Copying main network weights to the target network weights')
                    target_model.set_weights(model.get_weights())
                    steps_to_update_target_model = 0
                break
        
        if episode %100 == 0:
            print("save model =======")
            model.save(f'weight/{MODEL_SAVE}')

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)
        
    # env.close()

def infer():

    model = keras.models.load_model(MODEL_INFERENCE)

    observation, zero_list = env.reset()
    total_zero = (observation == 0.0).sum()
    pos_zero = np.where(observation==0.0)

    pred_actions = []
    print(observation)
    while 0.0 in observation:
        encoded = observation
        encoded_reshaped = encoded.reshape([1, encoded.shape[0]])
        predicted = model.predict(encoded_reshaped).flatten()
        action = np.argmax(predicted)
        pred_actions.append(action)
        new_observation, reward, done, info = env.step(action, zero_list,0)
        while np.array_equal(observation, new_observation):
            action = random.randint(0, STATE_ACTION_SPACE-1)
            # new_observation, reward, done, info = env.step(action, zero_list, 0)
        observation = new_observation

    miss_pos = np.setdiff1d(pos_zero, pred_actions)
    wrong_pred = np.setdiff1d(pred_actions, pos_zero)
    print(f'numzero = {total_zero}/{len(pred_actions)}\npos = {pos_zero}\npred = {pred_actions}\nmiss_po {len(miss_pos)}s = {miss_pos}\nwrong_pred = {wrong_pred}')        
if __name__ == '__main__':
    main()
    # infer()