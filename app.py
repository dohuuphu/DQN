import os
import time
import uvicorn
import argparse

from fastapi import FastAPI
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

def set_mode(mode:str)->int:
    
    if mode == 'dev':
        collection_user =  "User_dev"
        port = 30615
    elif mode == "staging":
        collection_user = "User_staging"    
        port = 30617
    elif mode == "deploy":
        collection_user = "User"
        port = 30616
    elif mode == "uat":
        collection_user = "User_uat"
        port = 30619
        
    return collection_user, port





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, choices=["dev", "staging", "deploy", "uat"], required=True, help="dev/staging/deploy/uat")
    args = parser.parse_args()

    collection_user, port_number = set_mode(args.mode)

    
    Logger()
    app = set_up_app()
    recommender = Recommend_core(collection_user)
    route_setup(app, recommender)

    uvicorn.run(app, host='0.0.0.0', port=port_number)