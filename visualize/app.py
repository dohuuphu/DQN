import sys
sys.path.append('/home/ubuntu/DQN/')

from dqn.variables import *
from graph_template import *
from dqn.database import MongoDb


from dash import Dash, dcc, html, Input, Output


#======================== Init-value ========================
app = Dash(__name__)
database = MongoDb()

list_graphType = [attr for attr in dir(Graph) if not callable(getattr(Graph, attr)) and not attr.startswith("__")]
#======================== Front-end ========================

app.layout = html.Div([
    html.Div([
        html.H1(children='User Informations Visualization',
            style={
                'textAlign': 'center'
            }),

        html.Div([
            html.Label(children='Graph_type'),
            dcc.Dropdown(
                list_graphType,
                list_graphType[0],
                id='graph_type',
            )
        ], style={'width': '20%', 'display': 'inline-block'}),
        
    
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


        html.Div([
            html.Label(children='Lesson'),
            dcc.Dropdown(
                [], # read from db
                '1',
                id='lesson',
            )
        ], style={'width': '10%', 'display': 'inline-block'}),
    ]),

    dcc.Graph(id='graphic')

])

#======================== Call-back ========================
@app.callback(
    Output('level', 'options'),
    Input('subject', 'value')
)
def get_level(subject):
    '''
        Update list levels when change subject value
    '''
    return list(database.content_data[subject].keys())

@app.callback(
    Output('category', 'options'),
    Input('subject', 'value'),
    Input('level', 'value'),
)
def get_category(subject_name, level):
    '''
        Update list category when change subject or level
    '''
    return list(database.content_data[subject_name][str(level)].keys())

@app.callback(
    Output('topic', 'options'),
    Output('topic', 'value'),
    Input('subject', 'value'),
    Input('level', 'value'),
    Input('category', 'value'),
)
def get_topic(subject_name, level, category):
    '''
        Update list category when change subject or level or category
    '''
    topic_list = list(database.content_data[subject_name][str(level)][str(category)].keys())
    return topic_list, topic_list[0]


@app.callback(
    Output('lesson', 'options'),
    Output('lesson', 'value'),
    Input('subject', 'value'),
    Input('level', 'value'),
    Input('category', 'value'),
    Input('topic', 'value'),
)
def get_lesson(subject_name, level, category,topic):
    '''
        Update list category when change subject or level or category
    '''
    lesson_list = list(database.content_data[subject_name][str(level)][str(category)][str(topic)].keys())
    return lesson_list, lesson_list[0]


@app.callback(
    Output('graphic', 'figure'),
    Input('graph_type', 'value'),
    Input('subject', 'value'),
    Input('category', 'value'),
    Input('level', 'value'),
    Input('topic', 'value'),
    Input('lesson', 'value'),
)
def update_graph(*args):
    '''
        Update graph by collect information from settings and create a path of csv file
    '''
    graph_type = args[0]
    print('graph_type ',  graph_type , Graph.analysis_score_in_lesson)
    if graph_type == Graph.initial_masteries_analysis:
        fig = initial_masteries_analysis(*args)
    elif graph_type == Graph.analysis_score_in_lesson:
        fig = analysis_score_in_lesson(*args)
    elif graph_type == Graph.analysis_repeated_in_lesson:
        fig = analysis_repeated_in_lesson(*args)

    return fig



if __name__ == '__main__':
    c = 0
    app.run_server(debug=True)
