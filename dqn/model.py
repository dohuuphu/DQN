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
    def __init__(self, name, agent, event_copy_weight, learning_rate, replay_memory, episode):
        self.name = name
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
        # a = 0

        while True:
            # a+=1
            # if len(self.replay_memory) < MIN_REPLAY_SIZE and len(self.replay_memory) - temp_memory >=  batch_size:
            #     temp_memory = len(self.replay_memory)
            if self.episode != 0 and self.episode % 10 == 0:
                # if a % 1000000 == 0:
                    # print(self.name, self.replay_memory)
                if len(self.replay_memory) >  MIN_REPLAY_SIZE:
                    
                    # Get random data and remove them in replay_memory
                    self.lock.acquire()
                    mini_batch = random.sample(self.replay_memory, batch_size)
                    [self.replay_memory.remove(item) for item in mini_batch]
                    self.lock.release()

                    current_states = []
                    current_topicID = []
                    new_states = []
                    for transition in mini_batch:
                        state:np.ndarray = transition[0]
                        current_states.append(state.reshape([1, state.shape[0]]))

                        topic_number:int = transition[1]
                        current_topicID.append(topic_number)

                        new_state:np.ndarray = transition[4]
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
                    
                    if self.episode %100 == 0:
                        print("save model =======")
                        self.model.save(f'weight/{MODEL_SAVE}')

                    
                    K.clear_session()

class Subject_core():
    def __init__(self, name:str, learning_rate:float) -> None:
        self.episode = 0
        self.env = SimStudent() 
        self.replay_memory = deque(maxlen=1_500)
        self.event_copy_weight = threading.Event()  
        self.model = Agent(STATE_ACTION_SPACE, NUM_TOPIC)
        self.learner = Learner(name, self.model, self.event_copy_weight, learning_rate, self.replay_memory, self.episode) 

class Recommend_core():
    def __init__(self, learning_rate=0.001):
        self.lock = Lock()

        self.database = MongoDb()

        threads = [] 
        self.english = Subject_core(name=ENGLISH, learning_rate=learning_rate)
        self.math_Algebra = Subject_core(name=ALGEBRA, learning_rate=learning_rate)
        self.math_Geometry = Subject_core(name=GEOMETRY, learning_rate=learning_rate)

        threads.append(threading.Thread(target=self.english.learner.train))
        threads.append(threading.Thread(target=self.math_Algebra.learner.train))
        threads.append(threading.Thread(target=self.math_Geometry.learner.train))
        
        for thread in threads:
            thread.start()
    

    def predict_action(self, subject:str, observation:np.ndarray, topic_number:int, episode:int, zero_list:list, prev_action:int=None):
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
            encoded_reshaped = observation.reshape([1, observation.shape[0]])
            if subject == ENGLISH:
                self.english.event_copy_weight.wait() # wait if learner is setting weight
                action = self.english.model((encoded_reshaped, np.array([topic_number])))
            elif subject == ALGEBRA:
                self.math_Algebra.event_copy_weight.wait()
                action = self.math_Algebra.model((encoded_reshaped, np.array([topic_number])))
            elif subject == GEOMETRY:
                self.math_Geometry.event_copy_weight.wait()
                action = self.math_Geometry.model((encoded_reshaped, np.array([topic_number])))

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
        episode = 0

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

            # Get curr_observation 
            masteries_of_topic:dict = self.database.get_topic_masteries(subject=inputs.subject, level=inputs.program_level, 
                                                                    topic_name=data_readed.topic_name, total_masteries=total_masteries) # process from masteries_of_test
            curr_observation:np.ndarray = np.array(list(masteries_of_topic.values())) # if done topic => curr_observation is full 1

            # Calculate reward for prev_observation
            curr_topic_name = data_readed.topic_name    #
            prev_topic_id = self.database.get_topic_id(inputs.subject, inputs.program_level, curr_topic_name) # Process from curr_topic_name

            # Update replay_buffer
            init_topic_observation:list = list(data_readed.total_topic[curr_topic_name].values())
            raw_zerolist:list = [i for i in range(len(init_topic_observation)) if init_topic_observation[i] == 0.0] 

            self.lock.acquire()
            # action:int, observation_:list, zero_list:list, score:int
            if inputs.subject == ENGLISH:
                reward, done = self.english.env.step_api(action=data_readed.prev_action, observation_=curr_observation, zero_list=raw_zerolist, score=inputs.score) 
                self.english.episode += 1 if done else 0 
                episode = self.english.episode
                self.english.replay_memory.append([data_readed.observation, prev_topic_id, data_readed.prev_action, reward, curr_observation, done])
            elif inputs.subject == ALGEBRA:
                reward, done = self.math_Algebra.env.step_api(data_readed.prev_action, raw_zerolist, inputs.score)
                self.math_Algebra.episode += 1 if done else 0 
                episode = self.math_Algebra.episode
                self.math_Algebra.replay_memory.append([data_readed.observation, prev_topic_id, data_readed.prev_action, reward, curr_observation, done])
            elif inputs.subject == GEOMETRY:
                reward, done = self.math_Geometry.env.step_api(data_readed.prev_action, raw_zerolist, inputs.score)
                self.math_Geometry.episode += 1 if done else 0 
                episode = self.math_Geometry.episode
                self.math_Geometry.replay_memory.append([data_readed.observation, prev_topic_id, data_readed.prev_action, reward, curr_observation, done])

            self.lock.release()

            # Topic is DONE
            if 0 not in curr_observation:
                # Render next topic
                next_topic = list(data_readed.total_topic.keys()).index(data_readed.topic_name) + 1 

                # Plan is DONE
                if next_topic >= len(data_readed.total_topic):
                    plan_done = True

                    self.lock.acquire()
                    self.database.update_interuptedPlan(inputs.user_id, inputs.subject, inputs.program_level, inputs.plan_name)
                    self.lock.release()

                else:
                    # Get next topic
                    curr_topic_name = list(data_readed.total_topic.keys())[next_topic]

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
        curr_topic_id:int = self.database.get_topic_id(inputs.subject, inputs.program_level, curr_topic_name) # process from curr_topic_name
        curr_zero_list = [i for i in range(len(curr_observation)) if curr_observation[i] == 0.0]

        # Take action
        action_index = self.predict_action(subject=inputs.subject, observation=curr_observation, topic_number=curr_topic_id, episode=episode, zero_list=curr_zero_list, prev_action=prev_action)
        action_id = action_index # process from action index

       
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
    def is_done_program(self, user_id:str, subject:str, level:str, plan_name:str):
        '''
            Check status in database
        '''
        return self.database.get_plan_status(user_id, subject, level, plan_name)
    