
import pandas as pd
import plotly.express as px
import sys
import os
sys.path.append('/home/ubuntu/DQN/')
from dqn.variables import *
from dqn.database import MongoDb
from dash import Dash, dcc, html, Input, Output

import random

app = Dash(__name__)
database = MongoDb()


TOPIC = []

# df_1 = pd.read_csv('/home/ubuntu/DQN/visualize/11/initial_masteries_analysis/1/topic_1.csv')
# df_2 = pd.read_csv('/home/ubuntu/DQN/visualize/11/initial_masteries_analysis/1/topic_2.csv')

# df_1.iloc[1]['value']
# print('aaa', df_1.iloc[1]['value'])
df_1 = pd.DataFrame({
    "lesson": [51, 51, 52],
    "amount": [1, 5, 1],
    "value": [5,6,10]
    }
    )
df_2 = pd.DataFrame({
    "lesson": [51, 51, 52],
    "amount": [11, 5, 1],
    "value": [10,6,10]
    }
    )


dict_df = {'1':df_1, '2':df_2}

app.layout = html.Div([
    html.Div([
        html.H1(children='User Informations Visualization',
            style={
                'textAlign': 'center'
            }),

        html.Div([
            html.Label(children='Graph_type'),
            dcc.Dropdown(
                V_GRAPH_TYPE,
                V_GRAPH_TYPE[0],
                id='graph_type',
            )
        ], style={'width': '10%', 'display': 'inline-block'}),
        
    
        html.Div([
            html.Label(children='Subject'),
            dcc.Dropdown(
                ['English', 'Math'],
                'English',
                id='subject',
            )
        ], style={'width': '10%', 'display': 'inline-block'}),

        html.Div([
            html.Label(children='Level'),
            dcc.Dropdown(
                [11,12],
                11,
                id='level',
            )
        ], style={'width': '10%', 'display': 'inline-block'}),

        html.Div([
            html.Label(children='Category'),
            dcc.Dropdown(
                [],
                '1',
                id='category',
            )
        ], style={'width': '10%', 'display': 'inline-block'}),
    

        html.Div([
            html.Label(children='Topic'),
            dcc.Dropdown(
                [11,12], # read from db
                11,
                id='topic',
            )
        ], style={'width': '10%', 'display': 'inline-block'}),


    ]),

    dcc.Graph(id='graphic')

])

@app.callback(
    Output('level', 'options'),
    Input('subject', 'value')
)
def get_level(subject):
    return list(database.content_data[subject].keys())

@app.callback(
    Output('category', 'options'),
    Input('subject', 'value'),
    Input('level', 'value'),
)
def get_category(subject_name, level):
    return list(database.content_data[subject_name][str(level)].keys())

@app.callback(
    Output('topic', 'options'),
    Output('topic', 'value'),
    Input('subject', 'value'),
    Input('level', 'value'),
    Input('category', 'value'),
)
def get_topic(subject_name, level, category):
    print(subject_name, level, category)
    topic_list = list(database.content_data[subject_name][str(level)][str(category)].keys())
    return topic_list, topic_list[0]


@app.callback(
    Output('graphic', 'figure'),
    Input('graph_type', 'value'),
    Input('subject', 'value'),
    Input('category', 'value'),
    Input('level', 'value'),
    Input('topic', 'value'),

)
def update_graph(graph_type, subject, category, level, topic):
    if graph_type == V_GRAPH_TYPE[1]:
        path_plot  = os.path.join('/home/ubuntu/DQN/visualize', str(level), str(graph_type), str(category), f'topic_{topic}.csv')
        data = pd.read_csv(path_plot, dtype = {'lesson': str, 'amount': str, 'value': str})
        fig = px.bar(data, x="lesson", y="amount", color="value", barmode="group")
    elif graph_type == V_GRAPH_TYPE[0]:
        path_plot  = os.path.join('/home/ubuntu/DQN/visualize', str(level), str(graph_type), str(category), topic,  f'lesson_1.0.csv')
        data = pd.read_csv(path_plot, dtype = {'lesson': int, 'score': int, 'value': int})
        fig = px.bar(data, y="value", x="score", barmode="group")
    # data = dict_df[str(category)]
    print('path', path_plot)
    # data = pd.read_csv(path_plot, dtype = {'lesson': str, 'amount': str, 'value': str})
    # print(data)

    


    return fig



if __name__ == '__main__':
    c = 0
    app.run_server(debug=True)
