

import asyncio
import aiohttp
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
from time import perf_counter

from dqn.variables import CHECKDONE_LOG, RECOMMEND_LOG, SYSTEM_LOG, DONE, INPROCESS


class BodyLesson(BaseModel):
    program: str
    level: str

class SandBox(BaseModel):
    callback_error: bool

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
    # score:int = None
    score:Optional[dict] = None 
    category:str = None
    sandbox: Optional[SandBox]


def route_setup(app, RL_model, url_callback):
    executor = ThreadPoolExecutor()

    def execute_api(item: Item):

        info = f'IN_request_INFO: {item.user_mail}_{item.subject}_{item.program_level}_{item.plan_name}|prev_score: {item.score}| masteries: {item.masteries}'
        logging.getLogger(RECOMMEND_LOG).info(info)
        try:
            (result, mssg), infer_time = RL_model.get_learning_point(item)
        except OSError as e:
            result = 'error'
        
        # Logging
        endline = f'='*80
        tab = f'\t'*8
        info = f"OUT_request_INFO: {item.user_mail}_{item.subject}_{item.program_level}_{item.plan_name}|prev_score: {item.score}| masteries: {item.masteries}\n{mssg}{tab}result {result}\n{tab}process_time: {infer_time:.3f}s\n{endline}\n"
        logging.getLogger(RECOMMEND_LOG).info(info)


        return APIResponse.json_format(result)

    async def call_another_api(item):
        info = f'IN_request_INFO: {item.user_mail}_{item.subject}_{item.program_level}_{item.plan_name}|prev_score: {item.score}| masteries: {item.masteries}'
        logging.getLogger(RECOMMEND_LOG).info(info)
        try:
            (result, mssg), infer_time = RL_model.get_learning_point(item)
        except OSError as e:
            result = 'error'
        print(result)
        input_json = {
            "user_id": item.user_id,
            "user_mail": item.user_mail,
            "subject": item.subject,
            "program_level": item.program_level,
            "plan_name": item.plan_name,
            "lesson_id": None if result['1'] == "error" else result['1']
        }
        print(input_json)
        # Logging
        endline = f'='*80
        tab = f'\t'*8
        info = f"OUT_request_INFO: {item.user_mail}_{item.subject}_{item.program_level}_{item.plan_name}|prev_score: {item.score}| masteries: {item.masteries}\n{mssg}{tab}result {result}\n{tab}process_time: {infer_time:.3f}s\n{endline}\n"
        logging.getLogger(RECOMMEND_LOG).info(info)
        url = url_callback
        async with aiohttp.ClientSession(headers={"X-Authenticated-User":"kyons-ai-api-key"}) as session:
            async with session.post(url, json=input_json) as response:
                result = await response.text()
                print(result)
                return result

    @app.post('/recommender')
    async def get_learningPoint(item: Item):
        try:
            asyncio.create_task(call_another_api(item)) # call another API asynchronously
            return {'status': 200}
        except Exception as e:
            logging.getLogger(ERROR_LOG).exception(e)
            return {'status': 403}            
        
    @app.post('/fake_api')
    async def call_back():
        try:
            print("Success")
            return 200
        except Exception as e:
            print("Failed")
            logging.getLogger(ERROR_LOG).exception(e)
            return  403          
    @app.get('/check_done_program')
    def check_done_program(item: Item):
        message, infer_time = RL_model.is_done_program(item.user_id, item.subject, item.program_level, item.plan_name)

        info = f'user_INFO: {item.user_mail}_{item.subject}_{str(item.program_level)}|{item.plan_name}: {message}|process_time: {infer_time:.3f}s'
        logging.getLogger(CHECKDONE_LOG).error(info)

        return APIResponse.json_format(message)


def route_setup_database(app): 

    @app.post('/update_database')
    async def udpate_database(db_info:Database_info):
        # Get data from backend and preprocess
        lesson_from_api = DB_Backend(method=db_info.method, url=db_info.url, header={db_info.key:db_info.value}, json = db_info.body)
        status = lesson_from_api.modify_current_db()
        
        # Hash_topic_ID
        hash_db = HashDB()
        hash_db.add_hash_db()
        return APIResponse.json_format(status)

    
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


        


