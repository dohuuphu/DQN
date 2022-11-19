import os
import time
import atexit
import uvicorn
import logging

from fastapi import FastAPI, status, Request
from fastapi.middleware.cors import CORSMiddleware

from api.route import route_setup_database
from dqn.model import Recommend_core
from dqn.log import Logger 
from dqn.database.core import MongoDb
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
    database = MongoDb()
    route_setup_database(app, database)

    uvicorn.run(app, host='0.0.0.0', port=30614)