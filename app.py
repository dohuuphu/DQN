import os
import time
import uvicorn

from fastapi import FastAPI, status, Request
from fastapi.middleware.cors import CORSMiddleware

from api.route import route_setup
from dqn.model import Recommend_core
from dqn.log import Logger 

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

    uvicorn.run(app, host='0.0.0.0', port=30616)