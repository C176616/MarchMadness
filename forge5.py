import pandas as pd
import json
import jsonpickle

from src.matchPrediction import MatchPrediction

import os
import glob

cwd = os.getcwd()

text_file = open('masterBracket.txt', 'r')
stringData = text_file.read()
text_file.close()

jsonData = json.loads(stringData)

tourn = jsonpickle.decode(stringData)
df_masterStage1 = pd.DataFrame(columns=['ID', 'Result'])

myiter = iter(tourn)
dict_points = {"1": 1, "2": 2, "3": 4, "4": 8, "5": 16, "6": 32}
df_master_stage2 = pd.DataFrame(columns=["gameID", "actualWinner", "points"])
for item in myiter:
    gameID = item.value
    if ([*gameID][0] == "R"):
        points = dict_points[[*gameID][1]]
    else:
        points = 1
    df_newData = pd.DataFrame({
        'gameID': [item.value],
        'actualWinner': [item.winner.teamID],
        'points': points
    })
    df_master_stage2 = pd.concat([df_master_stage2, df_newData],
                                 ignore_index=True)

    if item.team1 == item.winner:
        result = 1
    else:
        result = 0

    df_newData = pd.DataFrame({
        'ID':
        ["2022_" + str(item.team1.teamID) + "_" + str(item.team2.teamID)],
        'Result': [result]
    })
    df_masterStage1 = pd.concat([df_masterStage1, df_newData],
                                ignore_index=True)

print(df_masterStage1)

submissionsDirectory = cwd + '//stage2Entries'

files = glob.glob(submissionsDirectory + '/*.txt')

#Stage 1
df_resultsList_stage1 = pd.DataFrame(columns=['Name', 'Result'])
for file in files:
    text_file = open(file, 'r')
    stringData = text_file.read()
    text_file.close()

    jsonData = json.loads(stringData)

    tourn = jsonpickle.decode(stringData)
    df_submission = pd.DataFrame(columns=['ID', 'Pred'])
    for prediction in tourn.predictionsList:
        df_newData = pd.DataFrame({
            'ID': [prediction.ID],
            'Pred': [prediction.pred]
        })
        df_submission = pd.concat([df_submission, df_newData],
                                  ignore_index=True)

    df_compare = df_submission.merge(df_masterStage1)
    df_compare['Pred'] = df_compare['Pred'].replace(1, 0.99)
    df_compare['Pred'] = df_compare['Pred'].replace(0, 0.01)

    import numpy as np
    n = df_compare.shape[0]
    df_compare = df_compare.assign(LogLoss=lambda x: (x['Result'] * np.log(x[
        'Pred']) + (1 - x['Result']) * np.log(1 - x['Pred'])))

    df_newData = pd.DataFrame({
        'Name': [tourn.author],
        'Result': [-(df_compare['LogLoss'].sum()) / n]
    })
    df_resultsList_stage1 = pd.concat([df_resultsList_stage1, df_newData],
                                      ignore_index=True)
print(df_resultsList_stage1)

df_resultsList_stage2 = pd.DataFrame(columns=['Name', 'Points'])
for file in files:
    text_file = open(file, 'r')
    stringData = text_file.read()
    text_file.close()

    jsonData = json.loads(stringData)

    tourn = jsonpickle.decode(stringData)

    myiter = iter(tourn)
    df_submission = pd.DataFrame(columns=["gameID", "predictedWinner"])
    for item in myiter:
        df_newData = pd.DataFrame({
            'gameID': [item.value],
            'predictedWinner': [item.winner.teamID]
        })
        df_submission = pd.concat([df_submission, df_newData],
                                  ignore_index=True)

    df_compare = pd.merge(df_master_stage2, df_submission, on='gameID')

    points = df_compare[df_compare['actualWinner'] ==
                        df_compare['predictedWinner']]['points'].sum()

    df_newData = pd.DataFrame({'Name': [tourn.author], 'Points': [points]})
    df_resultsList_stage2 = pd.concat([df_resultsList_stage2, df_newData],
                                      ignore_index=True)

print(df_resultsList_stage2)
