import gc
import random
import logging
import threading
import numpy as np
import tensorflow as tf
import keras.backend as K

from threading import Lock, Event
from tensorflow import keras
from collections import deque

from dqn.variables import *
from dqn.utils import *
from dqn.environment import SimStudent
from dqn.database import MongoDb, Format_reader, User, Data_formater
from api.route import Item

class Agent(keras.Model):
    def __init__(self, action_shape, number_topic):
        super().__init__()
        init = tf.keras.initializers.HeUniform()

        self.dense1 = keras.layers.Dense(1024, activation=tf.nn.tanh, kernel_initializer=init)
        self.dense2 = keras.layers.Dense(512, activation=tf.nn.tanh, kernel_initializer=init)
        # self.dense3 = keras.layers.Dense(256, activation=tf.nn.tanh, kernel_initializer=init)
        self.dense4 = keras.layers.Dense(action_shape, activation='linear', kernel_initializer=init)
        self.topic_embedding = keras.layers.Embedding(number_topic, 32, input_length=1, trainable=False)


    def call(self, inputs):
        (observation, topic_ids) = inputs
        topic_embedding = self.topic_embedding(topic_ids)

        x = self.dense1(observation)
        x = keras.layers.Concatenate(axis=1)([x, topic_embedding])
        x = self.dense2(x)
        # x = self.dense3(x)
        output = self.dense4(x)
        return output

class Learner():
    def __init__(self, agent, event_copy_weight, learning_rate, replay_memory, episode):
        self.agent = agent
        self.lock = Lock()
        self.event_copy_weight:Event = event_copy_weight
        self.episode = episode
        self.model = Agent(STATE_ACTION_SPACE, NUM_TOPIC)
        self.target_model = Agent(STATE_ACTION_SPACE, NUM_TOPIC)

        self.model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
        self.target_model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])

        self.replay_memory:deque = replay_memory

        self.train_summary_writer = tf.summary.create_file_writer("logs/" + MODEL_SAVE)
    
    def train(self):
        MIN_REPLAY_SIZE = 500
        batch_size = 1
        learning_rate = 0.7 # Learning rate
        discount_factor = 0.618
        # temp_memory = 0

        while True:

            # if len(self.replay_memory) < MIN_REPLAY_SIZE and len(self.replay_memory) - temp_memory >=  batch_size:
            #     temp_memory = len(self.replay_memory)
            if self.episode != 0 and self.episode % 10 == 0:

                # Get random data and remove them in replay_memory
                self.lock.acquire()
                mini_batch = random.sample(self.replay_memory, batch_size)
                [self.replay_memory.remove(item) for item in mini_batch]
                self.lock.release()

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

                current_qs_list = self.model((current_states, np.array(current_topicID))).numpy()
                future_qs_list = self.target_model((new_states, np.array(current_topicID))).numpy()

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
                    
                his = self.model.fit((np.array(X),np.array(T)), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True, workers=1)

                with self.train_summary_writer.as_default():
                    tf.summary.scalar('loss', his.history['loss'][0], step=self.episode)
                    tf.summary.scalar('acc', his.history['accuracy'][0], step=self.episode)
                

                # Update weight
                if self.episode % STEP_UPDATE_TARGETR_MODEL == 0:
                    # Log
                    logging.getLogger(SYSTEM_LOG).info(f'Update weight at {self.episode} episode !!!')
                    weight = self.model.get_weights()
                    self.target_model.set_weights(weight)

                    # Copy weight to Agent
                    self.event_copy_weight.clear()
                    self.agent.set_weights(weight)
                    self.event_copy_weight.set()

                
                K.clear_session()



class Recommend_core():
    def __init__(self, learning_rate=0.001):
        self.database = MongoDb()

        self.episode = 0
        self.lock = Lock()
        self.event_copy_weight = threading.Event()
        self.env = SimStudent()      
        self.replay_memory = deque(maxlen=1_500)
        self.model = Agent(STATE_ACTION_SPACE, NUM_TOPIC)
        self.learner = Learner(self.model, self.event_copy_weight, learning_rate, self.replay_memory, self.episode)

        thread = threading.Thread(target=self.learner.train)
        thread.start()
    

    def predict_action(self, observation:list, topic_number:int, episode:int, zero_list:list, prev_action:int=None):
        max_epsilon = 1 # You can't explore more than 100% of the time
        min_epsilon = 0.01 # At a minimum, we'll always explore 1% of the time
        decay = 0.01
        action = None

        # Calculate epsilon
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)

        if np.random.rand() <= epsilon:
            # Explore
            action = random.randint(0, len(observation) - 1)
        else:
            # Exploit best known action
            self.event_copy_weight.wait()
            encoded_reshaped = observation.reshape([1, observation.shape[0]])
            action = self.model((encoded_reshaped, np.array([topic_number])))

            # Random action after explore 
            if prev_action == action:
                    action = random.choice(zero_list)
        
        K.clear_session()
            
        return action 

    @timer
    def get_learning_point(self, inputs:Item): 
        # Processing input and store database
        data_readed:Format_reader = self.database.read_from_DB(inputs.user_id, inputs.subject, str(inputs.program_level), inputs.plan_name)

        flow_topic = None
        init_score = None
        prev_score = None
        plan_done = False


        # New plan
        if data_readed.prev_action is None :  
            flow_topic = self.database.prepare_flow_topic(subject=inputs.subject, level=inputs.program_level, total_masteries=inputs.masteries)
            curr_topic_name = flow_topic[0]
            prev_action = None
            init_score = inputs.score
            total_masteries = inputs.masteries

        
        # Exist plan
        else: 
            prev_action = data_readed.prev_action
            prev_score = inputs.score

            # Update total_masteries from single lesson
            self.lock.acquire()
            total_masteries = self.database.update_total_masteries(inputs.user_id, inputs.subject, inputs.program_level, inputs.plan_name, inputs.masteries)
            self.lock.release()
            # Get prev_observation 
            masteries_of_topic:dict = self.database.get_topic_masteries(subject=inputs.subject, level=inputs.program_level, 
                                                                    topic_name=data_readed.topic_name, total_masteries=total_masteries) # process from masteries_of_test
            curr_observation:list = list(masteries_of_topic.values())

            # Calculate reward for prev_observation and store experience  
            curr_topic_name = data_readed.topic_name
            curr_topic_id = 1 # Process from curr_topic_name
            reward, done = self.env.step_api(data_readed.prev_action, data_readed.zero_list, inputs.score)

            # Update episode
            if done:
                self.lock.acquire()
                self.episode += 1
                self.lock.release()

            self.lock.acquire()
            self.replay_memory.append([data_readed.observation, curr_topic_id, data_readed.prev_action, reward, curr_observation, done])
            self.lock.release()

            # Topic is DONE
            if 0 not in curr_observation:
                # Render next topic
                next_topic = data_readed.total_topic.index(data_readed.topic_name) + 1 

                # DONE PLAN
                if next_topic >= len(data_readed.total_topic):
                    plan_done = True

                    self.lock.acquire()
                    self.database.update_interuptedPlan(inputs.user_id, inputs.subject, inputs.program_level, inputs.plan_name)
                    self.lock.release()

                else:
                    # Get next topic
                    curr_topic_name = data_readed.total_topic[next_topic]

                # Update prev_score, path_status
                info = User(user_id = inputs.user_id ,user_mail = inputs.user_mail, subject = inputs.subject,           # depend on inputs
                            level = inputs.program_level, plan_name = inputs.plan_name, prev_score = inputs.score,     # depend on inputs
                            total_masteries=total_masteries, topic_masteries = masteries_of_topic, 
                            action_index = None, action_id = None, topic_name = data_readed.topic_name, init_score = None, 
                            flow_topic = None)
                
                self.lock.acquire()
                self.database.update_prev_score(Data_formater(info))
                self.lock.release()

        if plan_done:
            return "DONE"

        # Get lasted masteries, topic (update topic_masteries from total_masteries)
        masteries_of_topic:dict = self.database.get_topic_masteries(subject=inputs.subject, level=inputs.program_level, 
                                                                    topic_name=curr_topic_name, total_masteries=total_masteries) # process from masteries_of_test
        curr_observation:list = list(masteries_of_topic.values())
        curr_topic_id:int = None # process from curr_topic_name
        curr_zero_list = [i for i in range(len(curr_observation)) if curr_observation[i] == 0.0]

        # Take action
        action_index = 1#self.predict_action(observation=curr_observation, topic_id=curr_topic_id, episode=self.episode, zero_list=curr_zero_list, prev_action=prev_action)
        action_id = 99 # process from action index

       
        # Update info to database
        info = User(user_id = inputs.user_id ,user_mail = inputs.user_mail, subject = inputs.subject,        # depend on inputs
                    level = inputs.program_level, plan_name = inputs.plan_name, prev_score = prev_score,     # depend on inputs
                    total_masteries=total_masteries, topic_masteries = masteries_of_topic, 
                    action_index = action_index, action_id = action_id, topic_name = curr_topic_name, init_score = init_score, 
                    flow_topic = flow_topic)
        
        self.lock.acquire()
        self.database.write_to_DB(info)
        self.lock.release()

        gc.collect()

        return action_id

    @timer
    def is_done_program(self, student_ID:str, subject:str, level:int):
        '''
            Check status in database
        '''
        # return 
        pass
    