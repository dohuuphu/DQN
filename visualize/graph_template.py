import os

import pandas as pd
import plotly.express as px

def initial_masteries_analysis(graph_type, subject, category, level, topic,lesson):
    path_plot  = os.path.join('/home/ubuntu/DQN/visualize', str(level), str(graph_type), str(category), f'topic_{topic}.csv')
    print(path_plot)
    data = pd.read_csv(path_plot, dtype = {'lesson': str, 'amount': int, 'value': str})
    fig = px.bar(data, x="lesson", y="amount", color="value", barmode="group")
    
    return fig

def analysis_score_in_lesson(graph_type, subject, category, level, topic, lesson):
    path_plot  = os.path.join('/home/ubuntu/DQN/visualize', str(level), str(graph_type), str(category), topic,  f'lesson_{lesson}.0.csv')
    print(path_plot)
    data = pd.read_csv(path_plot, dtype = {'lesson': int, 'score': int, 'value': float})
    fig = px.bar(data, y="value", x="score", barmode="group")

    return fig

def analysis_repeated_in_lesson(graph_type, subject, category, level, topic, lesson):
    path_plot  = os.path.join('/home/ubuntu/DQN/visualize', str(level), str(graph_type), str(category), topic,  f'lesson_{lesson}.csv')
    print(path_plot)
    data = pd.read_csv(path_plot, dtype = {'lesson': int, 'repeat': int, 'n_repeat': float})
    fig = px.bar(data, y="n_repeat", x="repeat", barmode="group")

    return fig