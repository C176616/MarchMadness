import pandas as pd
import os

import json
import jsonpickle

from sklearn import linear_model

import numpy as np

from sklearn.ensemble import RandomForestClassifier

cwd = os.getcwd()

df_training_set = pd.read_csv('training_set.csv')
df_training_set_stage2 = pd.read_csv("training_set_stage2.csv")
df_stage1Combinations = pd.read_csv(cwd +
                                    "/data_stage2/MSampleSubmissionStage2.csv")

print(df_stage1Combinations)

with open('tournamentLayout.json', 'r') as f:
    jsonData = json.load(f)

tourn = jsonpickle.decode(jsonData)
print(tourn)

tourn.populatePredictionsList(df_stage1Combinations)

print(tourn.predictionsList[0].pred)

df_corr = df_training_set.corr()
print(df_corr)

df_modelResults = pd.DataFrame(columns=['Season', 'Error', 'Accuracy (%)'])
df_corr = df_training_set[(df_training_set['Season'] >= 2010)
                          & (df_training_set['Season'] <= 2014)].corr()
print(df_corr)

cols = ['deltaSeed', 'deltaPointsAgainst']

X = df_training_set[cols][(df_training_set['Season'] >= 2010)
                          & (df_training_set['Season'] <= 2014)]

print(X)
y = df_training_set['Result'][(df_training_set['Season'] >= 2010)
                              & (df_training_set['Season'] <= 2014)]
print(y)

linearModel = linear_model.LinearRegression()

linearModel.fit(X, y)

X_pred = df_training_set_stage2[cols]

print(X_pred)

pred = linearModel.predict(X_pred)

print(pred)
df_stage1Combinations['Pred'] = np.round(pred, 2)
print(df_stage1Combinations)
print(tourn.root.winner)
tourn.populatePredictionsList(df_stage1Combinations)
# print(tourn.predictionsList)

tourn.simulateTournament("False")
print(tourn.root.winner)