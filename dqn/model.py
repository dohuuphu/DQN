import gc
import random
import logging
import threading
import numpy as np
import tensorflow as tf
import keras.backend as K

from os.path import join
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

        self.train_summary_writer = tf.summary.create_file_writer(join("logs", self.name, MODEL_SAVE))
    
    def train(self):
        
        batch_size = 128
        learning_rate = 0.7 # Learning rate
        discount_factor = 0.618
        # a = 0

        while True:
            # a+=1
            # if len(self.replay_memory) < MIN_REPLAY_SIZE and len(self.replay_memory) - temp_memory >=  batch_size:
            #     temp_memory = len(self.replay_memory)
            if self.episode[0] != 0 and self.episode[0] % NUM_EPISODE_TRAIN == 0:
                # if a % 1000000 == 0:
                    # print(self.name, self.replay_memory)
                if len(self.replay_memory) >  MIN_REPLAY_SIZE:
                    logging.getLogger(SYSTEM_LOG).info(f'{self.name} at {self.episode[0]} -> Start trainning')

                    
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
                        tf.summary.scalar('loss', his.history['loss'][0], step=self.episode[0])
                        tf.summary.scalar('acc', his.history['accuracy'][0], step=self.episode[0])
                    

                    # Update weight
                    if self.episode[0] % STEP_UPDATE_TARGETR_MODEL == 0:
                        # Log
                        logging.getLogger(SYSTEM_LOG).info(f'UPDATE weight {self.name} model at {self.episode[0]} episode !!!')
                        weight = self.model.get_weights()
                        self.target_model.set_weights(weight)

                        # Copy weight to Agent
                        self.event_copy_weight.clear()
                        self.agent.set_weights(weight)
                        self.event_copy_weight.set()
                    
                    if self.episode[0] % EPISODE_SAVE == 0:
                        # print("save model =======")
                        self.model.save(join("weight", self.name, MODEL_SAVE))
                        logging.getLogger(SYSTEM_LOG).info(f'SAVE weight {self.name} model at {self.episode[0]} episode !!!')

                    
                    K.clear_session()

class Subject_core():
    def __init__(self, name:str, learning_rate:float, item_cache:Item_cache) -> None:
        # Get cache value of relay_buffer and episode
        if not bool(item_cache): # cache is empty
            self.episode = deque(maxlen=1)
            self.episode.append(0)
            self.replay_memory = deque(maxlen=1_500)
        else:
            self.episode = item_cache.episode
            self.replay_memory = item_cache.relay_buffer

        self.env = SimStudent() 
        self.event_copy_weight = threading.Event()  
        self.model = Agent(STATE_ACTION_SPACE, NUM_TOPIC)
        self.learner = Learner(name, self.model, self.event_copy_weight, learning_rate, self.replay_memory, self.episode) 

class Recommend_core():
    def __init__(self, learning_rate=0.001):
        self.lock = Lock()

        self.database = MongoDb()

        threads = [] 
        self.english = Subject_core(name=ENGLISH, learning_rate=learning_rate, item_cache=load_pkl(ENGLISH_R_BUFFER))
        self.math_Algebra = Subject_core(name=ALGEBRA, learning_rate=learning_rate, item_cache=load_pkl(ALGEBRA_R_BUFFER))
        self.math_Geometry = Subject_core(name=GEOMETRY, learning_rate=learning_rate, item_cache=load_pkl(GEOMETRY_R_BUFFER))
        self.math_Probability = Subject_core(name=PROBABILITY, learning_rate=learning_rate, item_cache=load_pkl(PROBABILITY_R_BUFFER))
        self.math_Analysis = Subject_core(name=ANALYSIS, learning_rate=learning_rate, item_cache=load_pkl(ANALYSIS_R_BUFFER))


        threads.append(threading.Thread(target=self.english.learner.train))
        threads.append(threading.Thread(target=self.math_Algebra.learner.train))
        threads.append(threading.Thread(target=self.math_Geometry.learner.train))
        threads.append(threading.Thread(target=self.math_Probability.learner.train))
        threads.append(threading.Thread(target=self.math_Analysis.learner.train))
        
        for thread in threads:
            thread.start()
    
    def predict_action(self, category:str, observation:np.ndarray, topic_number:int, episode:int, zero_list:list, prev_action:int=None):
        max_epsilon = 1 # You can't explore more than 100% of the time
        min_epsilon = 0.01 # At a minimum, we'll always explore 1% of the time
        decay = 0.01
        action = None

        # Calculate epsilon
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)

        if np.random.rand() <= epsilon:
            # Explore
            if np.random.choice([1,0],p=[0.4, 0.6]):
                action = random.randint(0, len(observation) - 1)
            else:
                index = random.randint(0, len(zero_list) - 1)
                action = zero_list[index]
        else:
            # Exploit best known action
            encoded_reshaped = observation.reshape([1, observation.shape[0]])
            if category == ENGLISH:
                self.english.event_copy_weight.wait() # wait if learner is setting weight
                action = self.english.model((encoded_reshaped, np.array([topic_number])))
            elif category == ALGEBRA:
                self.math_Algebra.event_copy_weight.wait()
                action = self.math_Algebra.model((encoded_reshaped, np.array([topic_number])))
            elif category == GEOMETRY:
                self.math_Geometry.event_copy_weight.wait()
                action = self.math_Geometry.model((encoded_reshaped, np.array([topic_number])))
            elif category == PROBABILITY:
                self.math_Probability.event_copy_weight.wait()
                action = self.math_Probability.model((encoded_reshaped, np.array([topic_number])))
            elif category == ANALYSIS:
                self.math_Analysis.event_copy_weight.wait()
                action = self.math_Analysis.model((encoded_reshaped, np.array([topic_number])))

            # Random action after explore 
            if prev_action == action:
                action = random.choice(zero_list)
        
        K.clear_session()
            
        return action 

    def arrange_usage_category(self, item:Item)->list:
        '''
            Create new items corresponding to sub_subject in raw_masteries 
        '''
        result_items = []

        if item.subject == MATH:
            sub_subjects = {}
            # Get total LPDs in the categorys
            LPD_in_categorys:dict = self.database.get_LDP_in_category(item.subject, item.program_level)
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
            
        else:   # English 
            parsed_item = item.copy()
            parsed_item.category = item.subject
            result_items.append(parsed_item)  
            
        return result_items

    @timer
    def get_learning_point(self, input:Item): 
        # Arrange LPD to correct_category, Math only
        parsed_inputs:list = self.arrange_usage_category(item=input)

        # Run with multi-threading
        results = {}
        # with ThreadPoolExecutor(max_workers=10) as executor:
        #     for result in executor.map(self.process_learningPath, parsed_inputs):
        #         results.update(result)

        # Run with sequence
        for inputs in parsed_inputs:
            results.update(self.process_learningPath(inputs))
        
        return results

    # Interact with user_data => using category as subject
    # Interact with user_data => usingg category as category

    def process_learningPath(self, inputs:Item): 
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
                return {inputs.category:"done"}

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
            total_masteries = self.database.update_total_masteries(inputs.user_id, inputs.category, inputs.program_level, inputs.plan_name, inputs.masteries)
            self.lock.release()

            # Get current_observation 
            masteries_of_topic:dict = self.database.get_topic_masteries(user_id=inputs.user_id, subject=inputs.subject, level=inputs.program_level, 
                                                                    topic_name=data_readed.topic_name, total_masteries=total_masteries) # process from masteries_of_test
            if masteries_of_topic is None:
                return   {inputs.category:"error"}                                                            
            curr_observation:np.ndarray = rawObservation_to_standarObservation(list(masteries_of_topic.values()), data_readed.topic_name) # if done topic => curr_observation is full 1
            
            # Calculate reward for prev_observation
            curr_topic_name = data_readed.topic_name   
            prev_topic_id = self.database.get_topic_id(inputs.subject, inputs.program_level, curr_topic_name) # Process from curr_topic_name

            # Update replay_buffer
            # init_topic_observation:list = list(data_readed.total_topic[curr_topic_name].values())
            # raw_zerolist:list = [i for i in range(len(init_topic_observation)) if init_topic_observation[i] == 0.0] 

            item_relayBuffer = Item_relayBuffer(total_step= data_readed.total_step, observation=data_readed.observation, topic_id=prev_topic_id, 
                                                action_index=data_readed.prev_action, next_observation=curr_observation,
                                                score=inputs.score, num_items_inPool=data_readed.num_items_inPool)

            episode = self.update_relayBuffer(item_relayBuffer, category=inputs.category)
            
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

        if plan_done:
            return {inputs.category:"done"}
        

        # Get lasted masteries, topic (update topic_masteries from total_masteries)
        masteries_of_topic:dict = self.database.get_topic_masteries(user_id=inputs.user_id, subject=inputs.subject, level=inputs.program_level, 
                                                                    topic_name=curr_topic_name, total_masteries=total_masteries) # process from masteries_of_test
        if masteries_of_topic is None:
                return   {inputs.category:"error"} 
        curr_observation:np.ndarray = rawObservation_to_standarObservation(list(masteries_of_topic.values()), curr_topic_name)
        curr_zero_list:list = [i for i in range(len(curr_observation)) if curr_observation[i] == 0.0]
        curr_topic_id:int = self.database.get_topic_id(inputs.subject, inputs.program_level, curr_topic_name) # process from curr_topic_name
        
        # Get action
        while True:
            action_index = self.predict_action(category=inputs.category, observation=curr_observation, topic_number=curr_topic_id, episode=episode, zero_list=curr_zero_list, prev_action=prev_action)

            # Select action until get right
            if self.is_suitable_action(curr_zero_list, action_index):
                break
            else:
                # Update negative action (action_value = 1) to relay buffer 
                if data_readed.prev_action is not None:
                    # init_topic_observation:list = list(data_readed.total_topic[curr_topic_name].values())
                    # raw_zerolist:list = [i for i in range(len(init_topic_observation)) if init_topic_observation[i] == 0.0] 
                    item_relayBuffer = Item_relayBuffer(total_step= data_readed.total_step, observation=curr_observation, topic_id=curr_topic_id, 
                                                    action_index=action_index, next_observation=curr_observation, num_items_inPool=data_readed.num_items_inPool) # score = None

                    episode = self.update_relayBuffer(item_relayBuffer, category=inputs.category)


        action_id = self.database.get_lessonID_in_topic(action_index, subject=inputs.subject, category=inputs.category, level=inputs.program_level, topic_name=curr_topic_name) # process from action index
       
        # Update info to database
        info = User(user_id = inputs.user_id ,user_mail = inputs.user_mail, subject = inputs.subject,        # depend on inputs
                    category=inputs.category, level = inputs.program_level, plan_name = inputs.plan_name,      # depend on inputs
                    prev_score = prev_score, total_masteries=total_masteries, topic_masteries = masteries_of_topic, 
                    action_index = action_index, action_id = action_id, topic_name = curr_topic_name,  
                    init_score = init_score,flow_topic = flow_topic)
        
        self.lock.acquire()
        self.database.write_to_DB(info)
        self.lock.release()

        gc.collect()

        return {inputs.category:action_id}

    def update_relayBuffer(self, item_relayBuffer:Item_relayBuffer, category:Item):
        self.lock.acquire()
                                                    
        if category == ENGLISH:       # action:int, observation_:list, zero_list:list, score:int
            reward, done = self.english.env.step_api(total_step=item_relayBuffer.total_step, action=item_relayBuffer.action_index, observation_=item_relayBuffer.observation,       
                                                        num_items_inPool=item_relayBuffer.num_items_inPool, score=item_relayBuffer.score) 
            
            episode = self.english.episode[0]+1 if done else 0 
            self.english.episode.append(episode) 

            self.english.replay_memory.append([item_relayBuffer.observation, item_relayBuffer.topic_id, item_relayBuffer.action_index, 
                                                reward, item_relayBuffer.next_observation, done])

        elif category == ALGEBRA:
            reward, done = self.math_Algebra.env.step_api(total_step=item_relayBuffer.total_step, action=item_relayBuffer.action_index, observation_=item_relayBuffer.observation,       
                                                             num_items_inPool=item_relayBuffer.num_items_inPool, score=item_relayBuffer.score) 
                                                             
            episode = self.math_Algebra.episode + 1 if done else 0 
            self.math_Algebra.episode.append(episode)
            
            self.math_Algebra.replay_memory.append([item_relayBuffer.observation, item_relayBuffer.topic_id, item_relayBuffer.action_index, 
                                                reward, item_relayBuffer.next_observation, done])

        elif category == GEOMETRY:
            reward, done = self.math_Geometry.env.step_api(total_step=item_relayBuffer.total_step, action=item_relayBuffer.action_index, observation_=item_relayBuffer.observation,       
                                                            num_items_inPool=item_relayBuffer.num_items_inPool, score=item_relayBuffer.score) 
            
            episode = self.math_Geometry.episode + 1 if done else 0 
            self.math_Geometry.episode.append(episode)

            self.math_Geometry.replay_memory.append([item_relayBuffer.observation, item_relayBuffer.topic_id, item_relayBuffer.action_index, 
                                                reward, item_relayBuffer.next_observation, done])
          
        elif category == PROBABILITY:
            reward, done = self.math_Probability.env.step_api(total_step=item_relayBuffer.total_step, action=item_relayBuffer.action_index, observation_=item_relayBuffer.observation,       
                                                            num_items_inPool=item_relayBuffer.num_items_inPool, score=item_relayBuffer.score) 
            
            episode = self.math_Probability.episode + 1 if done else 0 
            self.math_Probability.episode.append(episode)

            self.math_Probability.replay_memory.append([item_relayBuffer.observation, item_relayBuffer.topic_id, item_relayBuffer.action_index, 
                                                reward, item_relayBuffer.next_observation, done])

        elif category == ANALYSIS:
            reward, done = self.math_Analysis.env.step_api(total_step=item_relayBuffer.total_step, action=item_relayBuffer.action_index, observation_=item_relayBuffer.observation,       
                                                            num_items_inPool=item_relayBuffer.num_items_inPool, score=item_relayBuffer.score) 
            
            episode = self.math_Analysis.episode + 1 if done else 0 
            self.math_Analysis.episode.append(episode)

            self.math_Analysis.replay_memory.append([item_relayBuffer.observation, item_relayBuffer.topic_id, item_relayBuffer.action_index, 
                                                reward, item_relayBuffer.next_observation, done])

        if done:
            print(f'done {category}')


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

    
    
    

    