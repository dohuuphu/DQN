

import asyncio
import logging

from starlette.routing import Match
from fastapi import Request
from pydantic import BaseModel
from api.response import APIResponse
from concurrent.futures.thread import ThreadPoolExecutor

from dqn.variables import *
from dqn.database import MongoDb
from dqn.insert_data_to_db import DB_Backend
from dqn.hash_db import HashDB
from typing import Optional

from dqn.variables import CHECKDONE_LOG, RECOMMEND_LOG, SYSTEM_LOG, DONE, INPROCESS


class BodyLesson(BaseModel):
    program: str
    level: str

class Database_info(BaseModel):
    method:str
    url:str
    key:str
    value:str
    body: BodyLesson

class Item(BaseModel):
    user_id:str
    user_mail:str
    subject:str
    program_level:str
    plan_name:str
    masteries:dict = {}
    score:int = None
    category:str = None

def route_setup(app, RL_model):
    executor = ThreadPoolExecutor()

    def execute_api(item: Item):

        try:
            result, infer_time = RL_model.get_learning_point(item)
        except OSError as e:
            result = 'error'
        
        # Logging
        info = f'request_INFO: {item.user_mail}_{item.subject}_{item.program_level}|prev_score: {item.score}| masteries: {item.masteries}\n\t\t\t\t\t\t\t\tresult{result}\n\t\t\t\t\t\t\t\tprocess_time: {infer_time:.3f}s\n===============\n'
        logging.getLogger(RECOMMEND_LOG).info(info)


        return APIResponse.json_format(result)


    @app.post('/recommender')
    async def get_learningPoint(item: Item):

        return await asyncio.get_event_loop().run_in_executor(executor, execute_api, item)

    @app.get('/check_done_program')
    def check_done_program(item: Item):
        message, infer_time = RL_model.is_done_program(item.user_id, item.subject, item.program_level, item.plan_name)

        info = f'user_INFO: {item.user_mail}_{item.subject}_{str(item.program_level)}|{item.plan_name}: {message}|process_time: {infer_time:.3f}s'
        logging.getLogger(CHECKDONE_LOG).error(info)

        # # Logging
        # if message == '':
        #     info = f'user_INFO: {item.user_mail}_{item.subject}_{str(item.program_level)}|is_done: {is_done}|process_time: {infer_time:.3f}s'
        #     logging.getLogger(CHECKDONE_LOG).info(info)
        # else:
        #     info = f'user_INFO: {item.user_mail}_{item.subject}_{str(item.program_level)}|{item.plan_name}: {message}|process_time: {infer_time:.3f}s'
        #     logging.getLogger(CHECKDONE_LOG).error(info)

        return APIResponse.json_format(message)


def route_setup_database(app, database:MongoDb): 

    @app.post('/update_database')
    async def udpate_database(db_info:Database_info):
        # Get data from backend and preprocess
        lesson_from_api = DB_Backend(method=db_info.method, url=db_info.url, header={db_info.key:db_info.value}, json = db_info.body)
        lesson_from_api.modify_current_db()
        
        # Hash_topic_ID
        hash_db = HashDB()
        hash_db.add_hash_db()


    
    # @app.middleware("http")
    # async def log_requests(request: Request, call_next):
    #     logging.getLogger(SYSTEM_LOG).info(f"{request.method} {request.url}")
    #     routes = request.app.router.routes
    #     for route in routes:
    #         match, scope = route.matches(request)
    #         if match != Match.FULL:
    #             for name, value in scope["path_params"].items():
    #                 logging.getLogger(SYSTEM_LOG).debug(f"\t{name}: {value}")
    #     logging.getLogger(SYSTEM_LOG).debug("Headers:")
    #     for name, value in request.headers.items():
    #         logging.getLogger(SYSTEM_LOG).debug(f"\t{name}: {value}")
    #     # logging.getLogger(SYSTEM_LOG).info(f"Completed_in={formatted_process_time}ms status_code={response.status_code}")
    #     response = await call_next(request)
        
    #     return response


        


