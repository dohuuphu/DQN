import sys
import time
import random
import logging
import threading
import asyncio
import numpy as np
import tensorflow as tf
import keras.backend as K

from os.path import join, dirname, abspath, isdir
from tensorflow import keras
from collections import deque
from threading import Lock, Event
from concurrent.futures import ThreadPoolExecutor

from api.route import Item
from dqn.utils import *
from dqn.variables import *
from dqn.environment import SimStudent
from dqn.database import MongoDb, Format_reader, User, Data_formater


class Agent(keras.Model):
    def __init__(self, action_shape, number_topic):
        super().__init__()
        init = tf.keras.initializers.HeUniform()

        self.dense1 = keras.layers.Dense(1024, activation=tf.nn.relu, kernel_initializer=init)
        self.dense2 = keras.layers.Dense(512, activation=tf.nn.relu, kernel_initializer=init)
        # self.dense3 = keras.layers.Dense(256, activation=tf.nn.tanh, kernel_initializer=init)
        self.dense4 = keras.layers.Dense(action_shape, activation='linear', kernel_initializer=init)
        self.topic_embedding = keras.layers.Embedding(number_topic, 32, input_length=1, trainable=False)


    def call(self, inputs):
        try:
            (observation, topic_ids) = inputs

            topic_embedding = self.topic_embedding(topic_ids)

            x = self.dense1(observation)
            x = keras.layers.Concatenate(axis=1)([x, topic_embedding])
            x = self.dense2(x)
            # x = self.dense3(x)
            output = self.dense4(x)

            return output
        except:
            pass

class Learner():
    def __init__(self, name, agent, event_copy_weight, learning_rate, replay_memory, episode, step):
        self.name = name
        self.agent = agent
        self.lock = Lock()
        self.step_training = step
        self.episode = episode

        self.event_copy_weight:Event = event_copy_weight
        self.path_model = join("weight", self.name, MODEL_SAVE)

        if not isdir(self.path_model):
            self.model = Agent(STATE_ACTION_SPACE, NUM_TOPIC)
            self.target_model = Agent(STATE_ACTION_SPACE, NUM_TOPIC)

            self.model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
            self.target_model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
        else:
            self.model = keras.models.load_model(self.path_model)
            self.target_model = keras.models.load_model(self.path_model)

        # Copy weight to agent
        # self.agent.set_weights(self.model.get_weights())

        self.replay_memory:deque = replay_memory

        self.train_summary_writer = tf.summary.create_file_writer(join("logs", self.name, MODEL_SAVE))
    
    def train(self):
        
        learning_rate = 0.7 # Learning rate
        discount_factor = 0.618
        temp_step = 0
        temp_episode = 0
        temp_update_target = 0

        while True:
            if self.step_training[0] != 0:
                if self.step_training[0] - temp_step >= STEP_TRAIN and len(self.replay_memory)  >  MIN_REPLAY_SIZE:
                    temp_step = self.step_training[0]
                    logging.getLogger(SYSTEM_LOG).info(f'{self.name} at step {self.step_training[0]} -> Start trainning')
                    
                    # Get random data and remove them in replay_memory
                    mini_batch = random.sample(self.replay_memory, BATCH_SIZE)

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
                        
                    his = self.model.fit((np.array(X),np.array(T)), np.array(Y), batch_size=BATCH_SIZE, verbose=0, shuffle=True, workers=1)

                    with self.train_summary_writer.as_default():
                        tf.summary.scalar('loss', his.history['loss'][0], step=self.episode[0])
                        tf.summary.scalar('acc', his.history['accuracy'][0], step=self.episode[0])
                    
                    # Copy weight to Agent
                    try:
                        self.event_copy_weight.clear()
                        self.agent.set_weights(self.model.get_weights())
                        self.event_copy_weight.set()
                    except:
                        logging.getLogger(SYSTEM_LOG).info(f'Copy weight from Model to Agent error at step {self.step_training[0]}')

                    # Update weight
                    
                    if self.step_training[0] - temp_update_target >= STEP_UPDATE_TARGETR_MODEL:
                        # Log
                        logging.getLogger(SYSTEM_LOG).info(f'UPDATE weight {self.name} model at {self.episode[0]} episode !!!')
                        weight = self.model.get_weights()
                        self.target_model.set_weights(weight)
                        temp_update_target = self.step_training[0] 
                    
                    if self.episode[0] - temp_episode >= EPISODE_SAVE:
                        logging.getLogger(SYSTEM_LOG).info(f'SAVE weight {self.name} model at {self.episode[0]} episode !!!')
                        self.model.save(self.path_model)
                        temp_episode = self.episode[0]
                        
                        # K.clear_session()

class Subject_core():
    def __init__(self, name:list, learning_rate:float, item_cache:Item_cache) -> None:
        # Get cache value of relay_buffer and episode
        if not bool(item_cache): # cache is empty
            self.step = deque(maxlen=1)
            self.step.append(0)
            self.episode = deque(maxlen=1)
            self.episode.append(0)
            self.replay_memory = deque(maxlen=1_500)
        else:
            self.step = item_cache.step
            self.episode = item_cache.episode
            self.replay_memory = item_cache.relay_buffer
        
        self.reward_per_episode = 0

        self.env = SimStudent() 
        self.event_copy_weight = threading.Event()  
        self.event_copy_weight.set()
        self.model = Agent(STATE_ACTION_SPACE, NUM_TOPIC)
        self.learner = Learner(name[0], self.model, self.event_copy_weight, learning_rate, self.replay_memory, self.episode, self.step) 

class Recommend_core():
    def __init__(self, learning_rate=0.001):
        self.lock = Lock()

        self.database = MongoDb()

        threads = [] 
        self.english_Grammar = Subject_core(name=GRAMMAR, learning_rate=learning_rate, item_cache=load_pkl(GRAMMAR_R_BUFFER))
        self.english_Vocabulary = Subject_core(name=VOVABULARY, learning_rate=learning_rate, item_cache=load_pkl(VOCABULARY_R_BUFFER))
        self.math_Algebra = Subject_core(name=ALGEBRA, learning_rate=learning_rate, item_cache=load_pkl(ALGEBRA_R_BUFFER))
        self.math_Geometry = Subject_core(name=GEOMETRY, learning_rate=learning_rate, item_cache=load_pkl(GEOMETRY_R_BUFFER))
        self.math_Probability = Subject_core(name=PROBABILITY, learning_rate=learning_rate, item_cache=load_pkl(PROBABILITY_R_BUFFER))
        self.math_Analysis = Subject_core(name=ANALYSIS, learning_rate=learning_rate, item_cache=load_pkl(ANALYSIS_R_BUFFER))

        threads.append(threading.Thread(target=self.english_Grammar.learner.train))
        threads.append(threading.Thread(target=self.english_Vocabulary.learner.train))
        threads.append(threading.Thread(target=self.math_Algebra.learner.train))
        threads.append(threading.Thread(target=self.math_Geometry.learner.train))
        threads.append(threading.Thread(target=self.math_Probability.learner.train))
        threads.append(threading.Thread(target=self.math_Analysis.learner.train))
        
        for thread in threads:
            thread.start()
    
    def predict_action(self, category:str, observation:list, topic_number:int, episode:int, zero_list:list, prev_action:int=None):
        max_epsilon = 1 # You can't explore more than 100% of the time
        min_epsilon = 0.01 # At a minimum, we'll always explore 1% of the time
        decay = 0.01
        action = None
        # start = time.time()
        # Calculate epsilon
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)
        if np.random.rand() <= epsilon:
            # Explore

            if np.random.choice([1,0],p=[0.3, 0.7]):
                action = random.randint(0, len(observation) - 1)
            else:
                index = random.randint(0, len(zero_list) - 1)
                action = zero_list[index]
            
        else:
            # Exploit best known action
            topic_number = int(topic_number)
            observation:np.ndarray = np.array(observation)
            encoded_reshaped = observation.reshape([1, observation.shape[0]])
            if category in GRAMMAR:
                self.english_Grammar.event_copy_weight.wait() # wait if learner is setting weight
                predicted = self.english_Grammar.model((encoded_reshaped, np.array([topic_number])))
            elif category in VOVABULARY:
                self.english_Vocabulary.event_copy_weight.wait()
                predicted = self.english_Vocabulary.model((encoded_reshaped, np.array([topic_number])))
            elif category in ALGEBRA:
                self.math_Algebra.event_copy_weight.wait()
                predicted = self.math_Algebra.model((encoded_reshaped, np.array([topic_number])))
            elif category in GEOMETRY:
                self.math_Geometry.event_copy_weight.wait()
                predicted = self.math_Geometry.model((encoded_reshaped, np.array([topic_number])))
            elif category in PROBABILITY:
                self.math_Probability.event_copy_weight.wait()
                predicted = self.math_Probability.model((encoded_reshaped, np.array([topic_number])))
            elif category in ANALYSIS:
                self.math_Analysis.event_copy_weight.wait()
                predicted = self.math_Analysis.model((encoded_reshaped, np.array([topic_number])))

            # Get action 
            action = np.argmax(predicted)

            # Random action after explore 
            if prev_action == action:
                action = random.choice(zero_list)
        
        # K.clear_session()
            
        return action 

    def arrange_usage_category(self, item:Item)->list:
        '''
            Create new items corresponding to sub_subject in raw_masteries 
        '''
        result_items = []
        sub_subjects = {}
        # Get total LPDs in the categorys
        LPD_in_categorys:dict = self.database.get_LPD_in_category(item.subject, item.program_level)
        for category in LPD_in_categorys:
            total_LPD:list = LPD_in_categorys[category]
            used_LPDs = {}
            for lpd in item.masteries:
                if lpd in total_LPD:
                    used_LPDs.update({lpd:item.masteries[lpd]})

            # Add LPD of category that exist in mock test   
            if used_LPDs != {}:
                sub_subjects.update({category:used_LPDs})

        # Create new item and add to result
        for sub_subject_name in sub_subjects:
            parsed_item = item.copy()
            parsed_item.category = sub_subject_name
            parsed_item.masteries = sub_subjects[sub_subject_name]
            result_items.append(parsed_item)
            
        return result_items

    @timer
    def get_learning_point(self, input:Item): 
        # Arrange LPD to correct_category, Math only
        parsed_inputs:list = self.arrange_usage_category(item=input)

        # Run with multi-threading
        results = {}
        mssgs = ''
        with ThreadPoolExecutor() as executor:
            for result, mssg in executor.map(self.process_learningPath, parsed_inputs):
                results.update(result)
                mssgs += mssg

        # Run with sequence
        # for inputs in parsed_inputs:
        #     results.update(self.process_learningPath(inputs))


        return results, mssgs

    # Interact with user_data => using category as subject
    # Interact with user_data => usingg category as category

    def process_learningPath(self, inputs:Item): 
        start = time.time()
        log_mssg = ''
        if len(inputs.masteries)>1 and 0 not in list(inputs.masteries.values()):
                return   {inputs.category:"Done"} 
        # Processing input and store database
        data_readed:Format_reader = self.database.read_from_DB(inputs.user_id, inputs.category, str(inputs.program_level), inputs.plan_name)

        flow_topic = None
        init_score = None
        prev_score = None
        plan_done = False
        episode = 0

        # New plan
        if data_readed.prev_action is None :  
            flow_topic = self.database.prepare_flow_topic(subject=inputs.subject, level=inputs.program_level, total_masteries=inputs.masteries)

            if not bool(flow_topic):    # don't have flow_topic when inputs is all 1
                log_mssg += f'Category_{inputs.category} dont have flow_topic when inputs is all 1\n'
                return {inputs.category:"done"}, log_mssg

            curr_topic_name = flow_topic[0]
            prev_action = None
            init_score = inputs.score
            total_masteries = inputs.masteries
            log_mssg += f'Category_{inputs.category} New_plan {(time.time()-start):.3f}\n'
  
        # Exist plan
        else: 
            prev_action = data_readed.prev_action
            prev_score = inputs.score

            # Update total_masteries from single lesson
            self.lock.acquire()
            total_masteries = self.database.update_total_masteries(inputs.user_id, inputs.category, inputs.program_level, inputs.plan_name, inputs.masteries)
            self.lock.release()

            # Get current_observation 
            masteries_of_topic:dict = self.database.get_topic_masteries(user_id=inputs.user_id, subject=inputs.subject, category=inputs.category, level=inputs.program_level, 
                                                                    topic_name=data_readed.topic_name, total_masteries=total_masteries) # process from masteries_of_test
            if masteries_of_topic is None:
                log_mssg += f'Category_{inputs.category} Topic is Error, masteries_of_topic\n'
                return   {inputs.category:"error"}, log_mssg  

            # Match raw_observ to standard_observ (equal to action_space)
            curr_observation:list = rawObservation_to_standarObservation(list(masteries_of_topic.values()), data_readed.topic_name) 
            
            # Calculate reward for prev_observation
            curr_topic_name = data_readed.topic_name   
            prev_topic_id = self.database.get_topic_id(inputs.subject, inputs.program_level, curr_topic_name) # Process from curr_topic_name

            # Update replay_buffer
            observation = rawObservation_to_standarObservation(data_readed.observation, data_readed.topic_name)

            item_relayBuffer = Item_relayBuffer(total_step= data_readed.total_step, observation=observation, topic_id=prev_topic_id, 
                                                action_index=data_readed.prev_action, next_observation=curr_observation,
                                                score=inputs.score, num_items_inPool=data_readed.num_items_inPool)

            episode = self.update_relayBuffer(item_relayBuffer, category=inputs.category)
            log_mssg += f'Category_{inputs.category} Before checkdone {(time.time()-start):.3f}\n'
            # Topic is DONE
            if 0 not in curr_observation:
                # Render next topic
                next_topic = list(data_readed.total_topic.keys()).index(data_readed.topic_name) + 1 

                # Plan is DONE
                if next_topic >= len(data_readed.total_topic):
                    plan_done = True

                    self.lock.acquire()
                    self.database.update_interuptedPlan(inputs.user_id, inputs.category, inputs.program_level, inputs.plan_name)
                    self.lock.release()

                else:
                    # Get next topic
                    curr_topic_name = list(data_readed.total_topic.keys())[next_topic]

                # Update prev_score, path_status
                info = User(user_id = inputs.user_id ,user_mail = inputs.user_mail, subject = inputs.subject,           # depend on inputs
                            category=inputs.category, level = inputs.program_level, plan_name = inputs.plan_name, 
                            prev_score = inputs.score, total_masteries=total_masteries, topic_masteries = masteries_of_topic,     # depend on inputs
                            action_index = None, action_id = None, topic_name = data_readed.topic_name, init_score = None, 
                            flow_topic = None)
                
                self.lock.acquire()
                self.database.update_prev_score(Data_formater(info))
                self.lock.release()
            log_mssg += f'Category_{inputs.category} After checkdone {(time.time()-start):.3f}\n'

        if plan_done:
            log_mssg += f'Category_{inputs.category} Topic is done\n'
            return {inputs.category:"done"}, log_mssg
        

        # Get lasted masteries, topic (update topic_masteries from total_masteries)
        masteries_of_topic:dict = self.database.get_topic_masteries(user_id=inputs.user_id, subject=inputs.subject, category=inputs.category, level=inputs.program_level, 
                                                                    topic_name=curr_topic_name, total_masteries=total_masteries) # process from masteries_of_test
        if masteries_of_topic is None:
                return   {inputs.category:"error"}

        # Match raw_observ to standard_observ (equal to action_space)
        curr_observation:np.ndarray = rawObservation_to_standarObservation(list(masteries_of_topic.values()), curr_topic_name) # create action_shape
        curr_zero_list:list = [i for i in range(len(curr_observation)) if curr_observation[i] == 0.0]
        if curr_zero_list == []:
            logging.getLogger(SYSTEM_LOG).error(f'{inputs.user_mail}_{inputs.category} topic name{curr_topic_name} masteries {masteries_of_topic}')
        curr_topic_id:int = self.database.get_topic_id(inputs.subject, inputs.program_level, curr_topic_name) # process from curr_topic_name
        
        log_mssg += f'Category_{inputs.category} Before get action {(time.time()-start):.3f}\n'

        # Get action
        while True:
            action_index = self.predict_action(category=inputs.category, observation=curr_observation, topic_number=curr_topic_id, episode=episode, zero_list=curr_zero_list, prev_action=prev_action)

            # Update new action to prev_action
            prev_action = action_index
            
            log_mssg += f'Category_{inputs.category} Get action "{action_index}" {(time.time()-start):.3f}\n'

            # Select action until get right
            
            # Update negative action (action_value = 1) to relay buffer 
            if data_readed.prev_action is not None:
                self.update_step(inputs.category)
                item_relayBuffer = Item_relayBuffer(total_step= data_readed.total_step, observation=curr_observation, topic_id=curr_topic_id, 
                                                action_index=action_index, next_observation=curr_observation, num_items_inPool=data_readed.num_items_inPool) # score = None

                episode = self.update_relayBuffer(item_relayBuffer, category=inputs.category)

                log_mssg += f'Category_{inputs.category} Update buffer {(time.time()-start):.3f}\n'
            
            if self.is_suitable_action(curr_zero_list, action_index):
                break
        
        log_mssg += f'Category_{inputs.category} After get action {(time.time()-start):.3f}\n'

        action_id = self.database.get_lessonID_in_topic(action_index, subject=inputs.subject, category=inputs.category, level=inputs.program_level, topic_name=curr_topic_name) # process from action index
       
        # Update info to database
        info = User(user_id = inputs.user_id ,user_mail = inputs.user_mail, subject = inputs.subject,        # depend on inputs
                    category=inputs.category, level = inputs.program_level, plan_name = inputs.plan_name,      # depend on inputs
                    prev_score = prev_score, total_masteries=total_masteries, topic_masteries = masteries_of_topic, 
                    action_index = action_index, action_id = action_id, topic_name = curr_topic_name,  
                    init_score = init_score,flow_topic = flow_topic)
        

        self.database.write_to_DB(info)

        log_mssg += f'Category_{inputs.category} Done writeDB {(time.time()-start):.3f}\n'

        return {inputs.category:action_id}, log_mssg

    def update_step(self, category:str):
        '''
            Update step after generate a action
        '''
        if category in GRAMMAR:
            self.english_Grammar.step[0]+=1
        elif category in VOVABULARY:
            self.english_Vocabulary.step[0]+=1
        elif category in ALGEBRA:
            self.math_Algebra.step[0]+=1
        elif category in GEOMETRY:
            self.math_Geometry.step[0]+=1
        elif category in PROBABILITY:
            self.math_Probability.step[0]+=1
        elif category in ANALYSIS:
            self.math_Analysis.step[0]+=1


    def update_relayBuffer(self, item_relayBuffer:Item_relayBuffer, category:Item):
        # MATH categories                                        
        if category in ALGEBRA:
            category_model = self.math_Algebra
        elif category in GEOMETRY:
            category_model = self.math_Geometry         
        elif category in PROBABILITY:
            category_model = self.math_Probability   
        elif category in ANALYSIS:
            category_model = self.math_Analysis   

        # ENGLISH categories
        elif category in GRAMMAR:
            category_model = self.english_Grammar   
        elif category in VOVABULARY:
            category_model = self.english_Vocabulary   
        
        # Update buffer with thread-safety
        return self.process_update_buffer(category_model=category_model, item_relayBuffer=item_relayBuffer)

    def process_update_buffer(self, category_model:Subject_core, item_relayBuffer:Item_relayBuffer):
        
        reward, done = category_model.env.step_api(total_step=item_relayBuffer.total_step, action=item_relayBuffer.action_index, observation_=item_relayBuffer.observation,       
                                                             num_items_inPool=item_relayBuffer.num_items_inPool, score=item_relayBuffer.score) 
        self.lock.acquire()

        category_model.reward_per_episode+=reward                                    
        episode = category_model.episode[0]
        if done:
            episode += 1

            # Write total reward in a episode to tensorboard
            with category_model.learner.train_summary_writer.as_default():
                tf.summary.scalar('reward', category_model.reward_per_episode, step=episode)
            
            # Reset total reward after done episode
            category_model.reward_per_episode = 0

        category_model.episode.append(episode)
        
        category_model.replay_memory.append([item_relayBuffer.observation, item_relayBuffer.topic_id, item_relayBuffer.action_index, 
                                            reward, item_relayBuffer.next_observation, done])
        self.lock.release()

        return episode

    def is_suitable_action(self, zero_list:list, action_index:int):
            return True if action_index in zero_list else False


    @timer
    def is_done_program(self, user_id:str, subject:str, level:str, plan_name:str):
        '''
            Check status in database
        '''
        return self.database.get_plan_status(user_id, subject, level, plan_name)

    
    
    

    