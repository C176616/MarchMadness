import base64
import datetime
import io

import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html, dash_table

import pandas as pd
import json
import jsonpickle

import plotly.express as px

import plotly.graph_objects as go

#app 2 is for viewing all bracket results in a folder

app = dash.Dash(__name__)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

text_file = open('masterBracket.txt', 'r')
stringData = text_file.read()
# print(stringData)
text_file.close()

dict_pointValues = {"1": 1, "2": 2, "3": 4, "4": 8, "5": 16, "6": 32}
# print(dict_pointValues["4"])

jsonData = json.loads(stringData)

masterTourn = jsonpickle.decode(stringData)

# print(masterTourn)
a = 'R2W4'
# print(a[1])

df_masterTourn = pd.DataFrame(columns=['Actual Team', 'Points'])
myiter = iter(masterTourn)
for item in myiter:
    if item.value == 'W16':
        data = {
            "Actual Team": [item.winner],
            "Points": [1],
        }
    else:
        round = item.value[1]
        points = dict_pointValues[round]
        data = {
            "Actual Team": [item.winner],
            "Points": [points],
        }
    df_newRow = pd.DataFrame(data)
    df_masterTourn = pd.concat([df_masterTourn, df_newRow], ignore_index=True)
    # print(data)
print(df_masterTourn)

# print(item.winner.teamID)
submissionsDirectory = cwd + '//submissions'
# bracketsDirectory=cwd + '//BracketScoring'
# resultsPath = cwd + '//actualResults.csv'

# results = pd.read_csv(resultsPath)

# files = glob.glob(submissionsDirectory+'/*.csv')
# files

# df_resultsList = pd.DataFrame(columns=['Name', 'Result'])

# for fileName in files:
#     submission = pd.read_csv(fileName)
#     name = fileName.split("\\")[len(fileName.split("\\"))-1].split('.')[0].split("_")[1]
# #     print(submission)
#     df_compare = submission.merge(results)
# #     print(df_compare.head())
#     df_compare['Pred'] = df_compare['Pred'].replace(1,.99)
#     df_compare['Pred'] = df_compare['Pred'].replace(0,.01)
# #     print(df_compare.head())
#     n = df_compare.shape[0]
#     df_compare = df_compare.assign(LogLoss = lambda x: (x['Result'] * np.log(x['Pred']) + (1-x['Result'])*np.log(1-x['Pred'])))

#     df_append_data = {
#         'Name':[name],
#         'Result':[-(df_compare['LogLoss'].sum())/n]
#     }

#     df_append = pd.DataFrame(df_append_data)
# #     print(df_append)
# #     df_append.head()
#     df_resultsList = df_resultsList.append(df_append)
# df_resultsList.head()

# master = pd.read_csv(cwd + '//master_bracket.csv')

# files = glob.glob(bracketsDirectory+'/*.csv')
# files

# df_resultsList = pd.DataFrame(columns=['Name', 'Result'])

# for fileName in files:
#     submission = pd.read_csv(fileName)
#     name = fileName.split("\\")[len(fileName.split("\\"))-1].split('.')[0].split("_")[1]
#     df_compare = pd.merge(master, submission, on='Coordinates')
#     points = df_compare[df_compare['Actual Team'] == df_compare['Predicted Team']]['Points'].sum()

#     df_append_data = {
#         'Name':[name],
#         'Result':[points]
#     }
#     df_append = pd.DataFrame(df_append_data)
#     df_resultsList = df_resultsList.append(df_append)

# df_resultsList.head()
