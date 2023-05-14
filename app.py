import os
import time
import uvicorn
import argparse

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.route import route_setup, route_setup_database
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
        url_callback = " https://student-api-dev.kyons.vn/ai/update_learning_path_async"
    elif mode == "staging":
        collection_user = "User_staging"    
        port = 30617
        url_callback = "https://student-api-stg.kyons.vn/ai/update_learning_path_async"
    elif mode == "deploy":
        collection_user = "User"
        port = 30616
        url_callback = "https://student-api.kyons.vn/ai/update_learning_path_async"
    elif mode == "uat":
        collection_user = "User_uat"
        port = 30619
    elif mode == "test":
        collection_user = "test"
        port = 30614
        url_callback = "http://18.143.44.85:30614/fake_api"
        
    return collection_user, port, url_callback





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, choices=["dev", "staging", "deploy", "uat", "test"], required=True, help="dev/staging/deploy/uat")
    args = parser.parse_args()

    collection_user, port_number, url_callback = set_mode(args.mode)

    
    Logger()
    app = set_up_app()
    recommender = Recommend_core(collection_user)
    route_setup(app, recommender, url_callback)
    route_setup_database(app)

    uvicorn.run(app, host='0.0.0.0', port=port_number)