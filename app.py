import os
import time
import atexit
import uvicorn
import logging

from fastapi import FastAPI, status, Request
from fastapi.middleware.cors import CORSMiddleware

from api.route import route_setup
from dqn.model import Recommend_core
from dqn.log import Logger 
from dqn.utils import save_pkl, Item_cache
from dqn.variables import *

os.environ['TZ'] = 'Asia/Bangkok'
time.tzset()

def set_up_app():
    app = FastAPI(name='kyonAI')
    origins = ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app

if __name__ == '__main__':
    
    Logger()
    app = set_up_app()
    recommender = Recommend_core()
    route_setup(app, recommender)

    # Cache relay_buffer
    def exit_handler():
        grammar_cache = Item_cache(recommender.english_Grammar.step, recommender.english_Grammar.episode, recommender.english_Grammar.replay_memory)
        save_pkl(grammar_cache, GRAMMAR_R_BUFFER)

        vocabulary_cache = Item_cache(recommender.english_Vocabulary.step, recommender.english_Vocabulary.episode, recommender.english_Vocabulary.replay_memory)
        save_pkl(vocabulary_cache, VOCABULARY_R_BUFFER)

        algebra_cache = Item_cache(recommender.math_Algebra.step, recommender.math_Algebra.episode, recommender.math_Algebra.replay_memory)
        save_pkl(algebra_cache, ALGEBRA_R_BUFFER)

        analysis_cache = Item_cache(recommender.math_Analysis.step, recommender.math_Analysis.episode, recommender.math_Analysis.replay_memory)
        save_pkl(analysis_cache, ANALYSIS_R_BUFFER)
        
        geometry_cache = Item_cache(recommender.math_Geometry.step, recommender.math_Geometry.episode, recommender.math_Geometry.replay_memory)
        save_pkl(geometry_cache, GEOMETRY_R_BUFFER)

        probability_cache = Item_cache(recommender.math_Probability.step, recommender.math_Probability.episode, recommender.math_Probability.replay_memory)
        save_pkl(probability_cache, PROBABILITY_R_BUFFER)

        logging.getLogger(SYSTEM_LOG).info('======= Save relay_buffer & deque =======')


    atexit.register(exit_handler)

    uvicorn.run(app, host='0.0.0.0', port=30616)