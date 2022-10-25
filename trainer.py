import os
import gc
import random
import numpy as np
import tensorflow as tf
import keras.backend as K

from tensorflow import keras
from collections import deque

from dqn.model import Agent
from dqn.environment import SimStudent
from dqn.variables import *

os.environ["CUDA_VISIBLE_DEVICES"]="-1"   

RANDOM_SEED = 5
tf.random.set_seed(RANDOM_SEED)

env = SimStudent()

def train(replay_memory, model, target_model, done, train_summary_writer, episode):
    learning_rate = 0.7 # Learning rate
    discount_factor = 0.618

    MIN_REPLAY_SIZE = 500
    if len(replay_memory) < MIN_REPLAY_SIZE:
        return

    batch_size = 64 * 2
    mini_batch = random.sample(replay_memory, batch_size)

    current_states = []
    current_topicID = []
    new_states = []
    for transition in mini_batch:
        state = transition[0]
        current_states.append(state.reshape([1, state.shape[0]]))

        topic_number = transition[1]
        current_topicID.append(topic_number)

        new_state = transition[4]
        new_states.append(new_state.reshape([1, new_state.shape[0]]))
    
    current_states = np.vstack(current_states)
    new_states = np.vstack(new_states)

    current_qs_list = model((current_states, np.array(current_topicID))).numpy()
    future_qs_list = target_model((new_states, np.array(current_topicID))).numpy()
    K.clear_session()


    X = []
    Y = []
    T = []
    for index, (observation, topic_number, action, reward, new_observation, done) in enumerate(mini_batch):
        if not done:
            max_future_q = reward + discount_factor * np.max(future_qs_list[index])
        else:
            max_future_q = reward

        current_qs = current_qs_list[index]
        current_qs[action] = ((1 - learning_rate) * current_qs[action] + learning_rate * max_future_q)

        X.append(observation)
        Y.append(current_qs)
        T.append(topic_number)
        
    his = model.fit((np.array(X),np.array(T)), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True, workers=1)

    with train_summary_writer.as_default():
        tf.summary.scalar('loss', his.history['loss'][0], step=episode)
        tf.summary.scalar('acc', his.history['accuracy'][0], step=episode)

    
def main():
    epsilon = 0.1 # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
    max_epsilon = 1 # You can't explore more than 100% of the time
    min_epsilon = 0.01 # At a minimum, we'll always explore 1% of the time
    decay = 0.01
    learning_rate = 0.001

    # Init model
    model = Agent(STATE_ACTION_SPACE, NUM_TOPIC)
    target_model = Agent(STATE_ACTION_SPACE, NUM_TOPIC)

    model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
    target_model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])

    target_model.set_weights(model.get_weights())

    # Init logger
    train_log_dir =  "output/logs/" + MODEL_SAVE 
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
        current_step = 0
        observation, zero_list = env.reset()
        
        total_zero = (observation == 0.0).sum()

        done = False
        old_action = ''
        while not done:
            steps_to_update_target_model += 1
            step_per_episode += 1
            current_step +=1

            random_number = np.random.rand()
            topic_number = random.randint(0, NUM_TOPIC - 1)

            # 2. Explore using the Epsilon Greedy Exploration Strategy
            if random_number <= epsilon:
                # Explore
                action = random.randint(0, len(observation) - 1)
            else:
                # Exploit best known action
                # model dims are (batch, env.observation_space.n)
                encoded = observation
                encoded_reshaped = encoded.reshape([1, encoded.shape[0]])
                predicted = model((encoded_reshaped, np.array([topic_number])))
                K.clear_session()
                action = np.argmax(predicted)

                # random action after explore 
                if old_action == action:
                    if random.randint(0,1):
                        action = random.choice(zero_list)
                            
                old_action = action
            new_observation, reward, done, _ = env.step(action, zero_list, current_step, topic_number)
            replay_memory.append([observation, topic_number, action, reward, new_observation, done])

            # 3. Update the Main Network using the Bellman Equation
            if steps_to_update_target_model % 4 == 0 or done:
                train(replay_memory, model, target_model, done, train_summary_writer, episode)

            observation = new_observation
            total_training_rewards += reward
    
            gc.collect()
            if done:
                print('Total training rewards: {} after n steps = {}/{} ({}) with final reward = {}'.format(total_training_rewards, step_per_episode, total_zero, episode, reward))
                with train_summary_writer.as_default():
                    tf.summary.scalar('reward', total_training_rewards, step=episode)
                    tf.summary.scalar('deviation', (step_per_episode - total_zero)/total_zero , step=episode)
                    tf.summary.scalar('epsilon', epsilon, step=episode)

                total_training_rewards = 0

                if steps_to_update_target_model >= STEP_UPDATE_TARGETR_MODEL:
                    print('Copying main network weights to the target network weights')
                    target_model.set_weights(model.get_weights())
                    steps_to_update_target_model = 0
                break
        
        if episode % NUM_EPISODE_TO_SAVE_MODEL == 0:
            model.save(f'output/weight/{MODEL_SAVE}')

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)
        
if __name__ == '__main__':
    main()
