import gc
import sys
import time
import atexit
import random
import logging
import threading
import numpy as np
import tensorflow as tf
import keras.backend as K

from tensorflow import keras
from collections import deque
from threading import Lock, Event
from multiexit import install, register
from os.path import join, dirname, abspath, isdir
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process, Manager, Queue, Value, Lock

from api.route import Item
from dqn.utils import *
from dqn.variables import *
from dqn.environment import SimStudent
from dqn.database import MongoDb, Format_reader, User, Data_formater


class Agent(keras.Model):
    def __init__(self, action_shape, embedding):
        super().__init__()
        init = tf.keras.initializers.HeUniform()

        self.dense1 = keras.layers.Dense(1024, activation=tf.nn.relu, kernel_initializer=init)
        self.dense2 = keras.layers.Dense(512, activation=tf.nn.relu, kernel_initializer=init)
        # self.dense3 = keras.layers.Dense(256, activation=tf.nn.tanh, kernel_initializer=init)
        self.dense4 = keras.layers.Dense(action_shape, activation='linear', kernel_initializer=init)
        self.topic_embedding = embedding

    def call(self, inputs):
        try:

            (observation, topic_ids) = inputs

            topic_f = self.topic_embedding(topic_ids)

            x = self.dense1(observation)
            x = keras.layers.Concatenate(axis=1)([x, topic_f])
            x = self.dense2(x)
            # x = self.dense3(x)
            output = self.dense4(x)

            return output
        except:
            pass

class Subject_core():
    def __init__(self, name:list, learning_rate:float, embedding) -> None:
        self.name = name[0]
        self.reward_per_episode = 0

        self.env = SimStudent() 
        self.items_shared = Item_shared()
        self.event_copy_weight = threading.Event()  
        self.event_copy_weight.set()
        self.embedding = embedding

        self.agent = None

        self.train_summary_writer = None
    


def get_cachePath(name):
        # MATH categories
        if name in ALGEBRA:
            cache_path = ALGEBRA_R_BUFFER
        elif name in GEOMETRY:
            cache_path = GEOMETRY_R_BUFFER
        elif name in PROBABILITY:
            cache_path = PROBABILITY_R_BUFFER
        elif name in ANALYSIS:
            cache_path = ANALYSIS_R_BUFFER

        # ENGLISH categories
        elif name in GRAMMAR:
            cache_path = GRAMMAR_R_BUFFER
        elif name in VOVABULARY:
            cache_path = VOCABULARY_R_BUFFER

        # MATHTEST categories
        elif name in TOAN_THPT:
            cache_path = TOTAL_R_BUFFER
        
        return cache_path 

def train(name, step_training:Value, episode:Value, observation_Q:Queue, topic_id_Q:Queue, action_index_Q:Queue, reward_Q:Queue, next_observation_Q:Queue, done_Q:Queue, weight_Q,embedding):
    
    # Init Learner model
    path_model = join("weight", name, MODEL_SAVE)

    if not isdir(path_model):
        model = Agent(STATE_ACTION_SPACE, embedding)
        target_model = Agent(STATE_ACTION_SPACE, embedding)

        model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
        target_model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
    else:
        logging.getLogger(SYSTEM_LOG).info(f'Load pretrained from {path_model}')
        model = keras.models.load_model(path_model)
        target_model = keras.models.load_model(path_model)
        
        weight_Q[-1] = model.get_weights()
    
    # Load & save cache_buffer
    cache_path = get_cachePath(name)

    def load_relayBuffer():
        cache = load_pkl(cache_path)
        if cache is not None:
            return cache
        else:
            return deque(maxlen=10000)

    replay_memory:deque = load_relayBuffer()
    
    @register
    def cache_relayBuffer():
        save_pkl(replay_memory, cache_path)


    # Logger of tensorboard
    train_summary_writer = tf.summary.create_file_writer(join("logs", name, MODEL_SAVE)) 

    # Hyperparameter
    learning_rate = 0.7 # Learning rate
    discount_factor = 0.618
    temp_step = 0
    temp_episode = 0
    temp_update_target = 0
    total_train_step = 0

    # Start Learning
    while True:
        # Get data from Queue of main process
        try:
            replay_memory.append([observation_Q.get(), topic_id_Q.get(), action_index_Q.get(), reward_Q.get(), next_observation_Q.get(), done_Q.get()])
        except:
            pass

        if step_training.value != 0:
            if step_training.value - temp_step >= STEP_TRAIN and len(replay_memory)  >  MIN_REPLAY_SIZE:
                temp_step +=  4#step_training.value
                total_train_step+=1
                logging.getLogger(SYSTEM_LOG).info(f'TRAINING {name}: AGENT_STEP {step_training.value} - LEARNER_STEP {total_train_step}')
                
                # Get random data and remove them in replay_memory
                mini_batch = random.sample(replay_memory, BATCH_SIZE)

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
                
                current_qs_list = model((current_states, np.array(current_topicID))).numpy()
                future_qs_list = target_model((new_states, np.array(current_topicID))).numpy()


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
                    
                his = model.fit((np.array(X),np.array(T)), np.array(Y), batch_size=BATCH_SIZE, verbose=0, shuffle=True, workers=1)

                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', his.history['loss'][0], step=episode.value)
                    tf.summary.scalar('acc', his.history['accuracy'][0], step=episode.value)
                
                # Copy weight from learner to Agent
                model_weight = model.get_weights()
                try:
                    weight_Q[-1] = (model_weight)
                except:
                    logging.getLogger(SYSTEM_LOG).error(f'LEARNER| Copy weight from Model to Agent error at step {total_train_step}')

                # Update weight            
                if step_training.value - temp_update_target >= STEP_UPDATE_TARGETR_MODEL:
                    try:
                        logging.getLogger(SYSTEM_LOG).info(f'LEARNER| Complete set weight TARGET_MODEL at step {total_train_step}')
                        target_model.set_weights(model_weight)
                    except :
                        logging.getLogger(SYSTEM_LOG).error(f'LEARNER| ERROR set weight TARGET_MODEL at step {total_train_step}')
                    temp_update_target = step_training.value 
                
                # Save weight
                if episode.value - temp_episode >= EPISODE_SAVE:
                    logging.getLogger(SYSTEM_LOG).info(f'LEARNER| SAVE weight {name} model at {episode.value} episode !!!')
                    model.save(path_model)
                    temp_episode = episode.value

class Recommend_core():
    def __init__(self, collection_user:str, learning_rate=0.001):

        install() # Init exits event

        self.lock = Lock()

        self.database = MongoDb(collection_user)    

        self.embedding = keras.layers.Embedding(NUM_TOPIC, 32, input_length=1, trainable=False) # need define topic_embeddings are separate from  category -> change hash topic_id

        self.english_Grammar = Subject_core(name=GRAMMAR, learning_rate=learning_rate, embedding=self.embedding)
        self.english_Vocabulary = Subject_core(name=VOVABULARY, learning_rate=learning_rate, embedding=self.embedding)
        self.math_Algebra = Subject_core(name=ALGEBRA, learning_rate=learning_rate, embedding=self.embedding)
        self.math_Geometry = Subject_core(name=GEOMETRY, learning_rate=learning_rate, embedding=self.embedding)
        self.math_Probability = Subject_core(name=PROBABILITY, learning_rate=learning_rate, embedding=self.embedding)
        self.math_Analysis = Subject_core(name=ANALYSIS, learning_rate=learning_rate, embedding=self.embedding)
        self.math_cerebry = Subject_core(name=TOAN_THPT, learning_rate=learning_rate, embedding=self.embedding)
        
        procs = [] 
        # procs.append(Process(target=train, args=(GRAMMAR[0], self.english_Grammar.items_shared.step,
        #                                                                         self.english_Grammar.items_shared.episode,
        #                                                                         self.english_Grammar.items_shared.observation,
        #                                                                         self.english_Grammar.items_shared.topic_id,
        #                                                                         self.english_Grammar.items_shared.action_index,
        #                                                                         self.english_Grammar.items_shared.reward,
        #                                                                         self.english_Grammar.items_shared.next_observation,
        #                                                                         self.english_Grammar.items_shared.done,
        #                                                                         self.english_Grammar.items_shared.weight,
        #                                                                         self.english_Grammar.embedding), daemon=True))
        # procs.append(Process(target=train, args=(VOVABULARY[0], self.english_Vocabulary.items_shared.step,
        #                                                                         self.english_Vocabulary.items_shared.episode,
        #                                                                         self.english_Vocabulary.items_shared.observation,
        #                                                                         self.english_Vocabulary.items_shared.topic_id,
        #                                                                         self.english_Vocabulary.items_shared.action_index,
        #                                                                         self.english_Vocabulary.items_shared.reward,
        #                                                                         self.english_Vocabulary.items_shared.next_observation,
        #                                                                         self.english_Vocabulary.items_shared.done,
        #                                                                         self.english_Vocabulary.items_shared.weight,
        #                                                                         self.english_Vocabulary.embedding), daemon=True))
        # procs.append(Process(target=train, args=(ALGEBRA[0], self.math_Algebra.items_shared.step,
        #                                                                         self.math_Algebra.items_shared.episode,
        #                                                                         self.math_Algebra.items_shared.observation,
        #                                                                         self.math_Algebra.items_shared.topic_id,
        #                                                                         self.math_Algebra.items_shared.action_index,
        #                                                                         self.math_Algebra.items_shared.reward,
        #                                                                         self.math_Algebra.items_shared.next_observation,
        #                                                                         self.math_Algebra.items_shared.done,
        #                                                                         self.math_Algebra.items_shared.weight,
        #                                                                         self.math_Algebra.embedding), daemon=True))
        # procs.append(Process(target=train, args=(GEOMETRY[0], self.math_Geometry.items_shared.step,
        #                                                                         self.math_Geometry.items_shared.episode,
        #                                                                         self.math_Geometry.items_shared.observation,
        #                                                                         self.math_Geometry.items_shared.topic_id,
        #                                                                         self.math_Geometry.items_shared.action_index,
        #                                                                         self.math_Geometry.items_shared.reward,
        #                                                                         self.math_Geometry.items_shared.next_observation,
        #                                                                         self.math_Geometry.items_shared.done,
        #                                                                         self.math_Geometry.items_shared.weight,
        #                                                                         self.math_Geometry.embedding), daemon=True))
        # procs.append(Process(target=train, args=(PROBABILITY[0], self.math_Probability.items_shared.step,
        #                                                                         self.math_Probability.items_shared.episode,
        #                                                                         self.math_Probability.items_shared.observation,
        #                                                                         self.math_Probability.items_shared.topic_id,
        #                                                                         self.math_Probability.items_shared.action_index,
        #                                                                         self.math_Probability.items_shared.reward,
        #                                                                         self.math_Probability.items_shared.next_observation,
        #                                                                         self.math_Probability.items_shared.done,
        #                                                                         self.math_Probability.items_shared.weight,
        #                                                                         self.math_Probability.embedding), daemon=True))
        # procs.append(Process(target=train, args=(ANALYSIS[0], self.math_Analysis.items_shared.step,
        #                                                                         self.math_Analysis.items_shared.episode,
        #                                                                         self.math_Analysis.items_shared.observation,
        #                                                                         self.math_Analysis.items_shared.topic_id,
        #                                                                         self.math_Analysis.items_shared.action_index,
        #                                                                         self.math_Analysis.items_shared.reward,
        #                                                                         self.math_Analysis.items_shared.next_observation,
        #                                                                         self.math_Analysis.items_shared.done,
        #                                                                         self.math_Analysis.items_shared.weight,
        #                                                                         self.math_Analysis.embedding), daemon=True))
        procs.append(Process(target=train, args=(TOAN_THPT[0], self.math_cerebry.items_shared.step,
                                                                                self.math_cerebry.items_shared.episode,
                                                                                self.math_cerebry.items_shared.observation,
                                                                                self.math_cerebry.items_shared.topic_id,
                                                                                self.math_cerebry.items_shared.action_index,
                                                                                self.math_cerebry.items_shared.reward,
                                                                                self.math_cerebry.items_shared.next_observation,
                                                                                self.math_cerebry.items_shared.done,
                                                                                self.math_cerebry.items_shared.weight,
                                                                                self.math_cerebry.embedding), daemon=True))
        for proc in procs:
            proc.start()

        
        # self.english_Grammar.init_Agent()
        # self.english_Vocabulary.init_Agent()
        # self.math_Algebra.init_Agent()
        # self.math_Geometry.init_Agent()
        # self.math_Probability.init_Agent()
        # self.math_Analysis.init_Agent()

        # Init and load model for agent
        try:
            self.english_Grammar.agent = keras.models.load_model(join("weight", self.english_Grammar.name, MODEL_SAVE))
        except:
            self.english_Grammar.agent = Agent(STATE_ACTION_SPACE, self.embedding) 

        self.english_Grammar.train_summary_writer = tf.summary.create_file_writer(join("logs", self.english_Grammar.name, MODEL_SAVE)) 
        
        try:
            self.english_Vocabulary.agent = keras.models.load_model(join("weight", self.english_Vocabulary.name, MODEL_SAVE))
        except:
            self.english_Vocabulary.agent = Agent(STATE_ACTION_SPACE, self.embedding) 
        self.english_Vocabulary.train_summary_writer = tf.summary.create_file_writer(join("logs", self.english_Vocabulary.name, MODEL_SAVE)) 
        
        try:
            self.math_Algebra.agent = keras.models.load_model(join("weight", self.math_Algebra.name, MODEL_SAVE))
        except:
            self.math_Algebra.agent = Agent(STATE_ACTION_SPACE, self.embedding) 
        self.math_Algebra.train_summary_writer = tf.summary.create_file_writer(join("logs", self.math_Algebra.name, MODEL_SAVE)) 
        
        try:
            self.math_Geometry.agent = keras.models.load_model(join("weight", self.math_Geometry.name, MODEL_SAVE))
        except:
            self.math_Geometry.agent = Agent(STATE_ACTION_SPACE, self.embedding) 
        self.math_Geometry.train_summary_writer = tf.summary.create_file_writer(join("logs", self.math_Geometry.name, MODEL_SAVE)) 
        
        try:
            self.math_Probability.agent = keras.models.load_model(join("weight", self.math_Probability.name, MODEL_SAVE))
        except:
            self.math_Probability.agent = Agent(STATE_ACTION_SPACE, self.embedding) 
        self.math_Probability.train_summary_writer = tf.summary.create_file_writer(join("logs", self.math_Probability.name, MODEL_SAVE)) 
        
        try:
            self.math_Analysis.agent = keras.models.load_model(join("weight", self.math_Analysis.name, MODEL_SAVE))
        except:
            self.math_Analysis.agent = Agent(STATE_ACTION_SPACE, self.embedding) 
        self.math_Analysis.train_summary_writer = tf.summary.create_file_writer(join("logs", self.math_Analysis.name, MODEL_SAVE)) 
        try:
            self.math_cerebry.agent = keras.models.load_model(join("weight", self.math_cerebry.name, MODEL_SAVE))
        except:
            self.math_cerebry.agent = Agent(STATE_ACTION_SPACE, self.embedding) 
        self.math_cerebry.train_summary_writer = tf.summary.create_file_writer(join("logs", self.math_cerebry.name, MODEL_SAVE)) 
    
        # Register a cleaner using a decorator
        @register
        def clean_main():
            for proc in procs:
                proc.terminate()
                proc.join()


    def predict_action(self, category:str, observation:list, topic_number:int, zero_list:list, prev_action:int=None):
        max_epsilon = 1 # You can't explore more than 100% of the time
        min_epsilon = 0.01 # At a minimum, we'll always explore 1% of the time
        decay = 0.01
        action = None
        model_predict = 'model'

        # Select model
        category_model:Subject_core = self.select_model(category)

        # Calculate epsilon
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * category_model.items_shared.episode.value)

        if np.random.rand() <= epsilon:
            # Explore
            model_predict = 'rand_explore'
            if np.random.choice([1,0],p=[0.7, 0.3]):    # Random value 1 or 0
                action = random.randint(0, len(observation) - 1)
            else:
                index = random.randint(0, len(zero_list) - 1)
                action = zero_list[index]
            
        else:
            # Exploit best known action
            topic_number = int(topic_number)
            observation:np.ndarray = np.array(observation)
            encoded_reshaped = observation.reshape([1, observation.shape[0]])

            self.update_weight(category_model)
            # category_model.event_copy_weight.wait()

            predicted = category_model.agent((encoded_reshaped, np.array([topic_number])))

            # Get action 
            action = np.argmax(predicted)

            # Random action after explore 
            if prev_action == action:
                model_predict = 'rand_exploit'
                action =  random.randint(0, len(observation) - 1)
        
        # K.clear_session()
            
        return action, model_predict

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

        results = {}
        mssgs = ''
        # Run with multi-threading
        # with ThreadPoolExecutor() as executor:
        #     for result, mssg in executor.map(self.process_learningPath, parsed_inputs):
        #         results.update(result)
        #         mssgs += mssg

        # Run with sequence
        for inputs in parsed_inputs:
            result, mssg = self.process_learningPath(inputs)
            results.update(result)
            mssgs += mssg

        return results, mssgs

    def process_learningPath(self, inputs:Item): 
        start = time.time()
        log_mssg = ''
        # if len(inputs.masteries)>1 and 0 not in list(inputs.masteries.values()):
        #     return {inputs.category:"Done"}, f'Category {inputs.category} is Done'
        # Processing input and store database
        data_readed:Format_reader = self.database.read_from_DB(inputs.user_id, inputs.category, str(inputs.program_level), inputs.plan_name)

        flow_topic = None
        init_score = None
        prev_score = None
        plan_done = False

        reward_per_user = data_readed.prev_reward

        # New plan
        if data_readed.prev_action is None :  
            #sort LPDs
            flow_topic, list_topicDone = self.database.prepare_flow_topic(subject=inputs.subject, level=inputs.program_level, total_masteries=inputs.masteries)

            if not bool(flow_topic):    # don't have flow_topic when inputs is all 1
                log_mssg += f'Category_{inputs.category} dont have flow_topic when inputs is all 1\n'

                # Update to db
                for topicName in list_topicDone:
                    info = User(user_id = inputs.user_id ,user_mail = inputs.user_mail, subject = inputs.subject,           # depend on inputs
                                category=inputs.category, level = inputs.program_level, plan_name = inputs.plan_name, 
                                prev_score = inputs.score, reward=reward_per_user, total_masteries=inputs.masteries, topic_masteries = None,     # depend on inputs
                                action_index = None, action_id = None, topic_name = topicName, init_score = None, 
                                flow_topic = list_topicDone)
                    self.database.write_to_DB(info)
                return {inputs.category:"done"}, log_mssg

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
            masteries_of_topic:dict = self.database.get_topic_masteries(user_id=inputs.user_id, subject=inputs.subject, category=inputs.category, level=inputs.program_level, 
                                                                    topic_name=data_readed.topic_name, total_masteries=total_masteries) # process from masteries_of_test
            if masteries_of_topic is None:
                log_mssg += f'Category_{inputs.category} get topic masteries was failed\n'
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

            reward = self.update_relayBuffer(item_relayBuffer, category=inputs.category, curr_reward=reward_per_user)
            # Update reward of user
            reward_per_user += reward

            # Topic is DONE
            if 0 not in curr_observation:
                # Render next topic
                next_topic = list(data_readed.total_topic.keys()).index(data_readed.topic_name) + 1 

                # Plan is DONE
                if next_topic >= len(data_readed.total_topic):
                    plan_done = True

                    # self.lock.acquire()
                    self.database.update_interuptedPlan(inputs.user_id, inputs.category, inputs.program_level, inputs.plan_name)
                    # self.lock.release()

                else:
                    # Get next topic
                    curr_topic_name = list(data_readed.total_topic.keys())[next_topic]

                # Update prev_score, path_status
                info = User(user_id = inputs.user_id ,user_mail = inputs.user_mail, subject = inputs.subject,           # depend on inputs
                            category=inputs.category, level = inputs.program_level, plan_name = inputs.plan_name, 
                            prev_score = inputs.score, reward=reward_per_user, total_masteries=total_masteries, topic_masteries = masteries_of_topic,     # depend on inputs
                            action_index = None, action_id = None, topic_name = data_readed.topic_name, init_score = None, 
                            flow_topic = None)
                
                # self.lock.acquire()
                self.database.update_prev_score_reward(Data_formater(info), update_lastest=True)
                # self.lock.release()

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

        # Get action
        while True:
            action_index, model_predict_flag = self.predict_action(category=inputs.category, observation=curr_observation, topic_number=curr_topic_id, zero_list=curr_zero_list, prev_action=prev_action)
            
            # Update new action to prev_action
            prev_action = action_index
            
            log_mssg += f'Category_{inputs.category} Get action "{action_index}" ({model_predict_flag}) {(time.time()-start):.3f}\n'

            # Select action until get right
            
            # Update negative action (action_value = 1) to relay buffer 
            if data_readed.prev_action is not None:
                self.update_step(inputs.category)
                item_relayBuffer = Item_relayBuffer(total_step= data_readed.total_step, observation=curr_observation, topic_id=curr_topic_id, 
                                                action_index=action_index, next_observation=curr_observation, num_items_inPool=data_readed.num_items_inPool) # score = None

                reward = self.update_relayBuffer(item_relayBuffer, category=inputs.category, curr_reward=reward_per_user)
                
                # Update reward of user
                reward_per_user += reward 

            
            if self.is_suitable_action(curr_zero_list, action_index):
                break

        action_id = self.database.get_lessonID_in_topic(action_index, subject=inputs.subject, category=inputs.category, level=inputs.program_level, topic_name=curr_topic_name) # process from action index
       
        # Update info to database
        info = User(user_id = inputs.user_id ,user_mail = inputs.user_mail, subject = inputs.subject,        # depend on inputs
                    category=inputs.category, level = inputs.program_level, plan_name = inputs.plan_name,      # depend on inputs
                    prev_score = prev_score, reward=reward_per_user, total_masteries=total_masteries, topic_masteries = masteries_of_topic, 
                    action_index = action_index, action_id = action_id, topic_name = curr_topic_name,  
                    init_score = init_score,flow_topic = flow_topic)
        
        self.database.write_to_DB(info)

        return {inputs.category:action_id}, log_mssg

    def update_weight(self, category_model:Subject_core):
        '''
            Update weight from LEARNER to AGENT in realtime by items_shared.weight
        '''
        try:     
            weight = category_model.items_shared.weight[-1]
            if len(category_model.agent.get_weights()) == len(weight):
                category_model.agent.set_weights(weight)
                # logging.getLogger(SYSTEM_LOG).info('DONE copy weight from LEARNER to AGENT')
            else:
                logging.getLogger(SYSTEM_LOG).error('None copy weight from LEARNER to AGENT')

        except:
            logging.getLogger(SYSTEM_LOG).error('FAILED copy weight from LEARNER to AGENT ')


    def update_step(self, category:str):
        '''
            Update step after generate a action
        '''
        # Select model
        category_model:Subject_core = self.select_model(category)

        # Update step
        # self.lock.acquire()
        category_model.items_shared.update_step()
        # self.lock.release()


    def update_relayBuffer(self, item_relayBuffer:Item_relayBuffer, category:str, curr_reward:int):
        # Select model
        category_model:Subject_core = self.select_model(category)

        # Get reward
        reward, done = category_model.env.step_api(total_step=item_relayBuffer.total_step, action=item_relayBuffer.action_index, observation_=item_relayBuffer.observation,       
                                                             num_items_inPool=item_relayBuffer.num_items_inPool, score=item_relayBuffer.score) 

        episode = category_model.items_shared.episode.value

        if done:
            episode += 1

            # Update new episode value
            category_model.items_shared.episode.value = episode

            # Write total reward in a episode to tensorboard
            with category_model.train_summary_writer.as_default():
                tf.summary.scalar('reward', (curr_reward+reward), step=episode)
            
            # Reset total reward after done episode
            category_model.reward_per_episode = 0

            # Update new episode value
            category_model.items_shared.udpate_episode(episode=episode)

        # Update relay_buffer
        category_model.items_shared.append_relayBuffer(item_relayBuffer.observation, item_relayBuffer.topic_id, item_relayBuffer.action_index, 
                                            reward, item_relayBuffer.next_observation, done)
        return reward

    def is_suitable_action(self, zero_list:list, action_index:int):
            return True if action_index in zero_list else False

    def select_model(self, category:str):
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

        # MATHTEST categories
        elif category in TOAN_THPT:
            category_model = self.math_cerebry
        
        return category_model

    @timer
    def is_done_program(self, user_id:str, subject:str, level:str, plan_name:str):
        '''
            Check status in database
        '''
        return self.database.get_plan_status(user_id, subject, level, plan_name)

    
    
    

    