import numpy as np
import pandas as pd
import os

cwd = os.getcwd()

currentSeason = 2023
currentSeasonStr = "2023"

df_seeds = pd.read_csv(cwd + "/data_stage2/MNCAATourneySeeds.csv")
df_seeds = df_seeds[df_seeds['Season'] == currentSeason]

teamList = df_seeds['TeamID']
print(teamList)

df_submission = pd.DataFrame(columns=['ID', 'Pred'])

for team1 in teamList:
    for team2 in teamList:
        print(team1)
        ID = currentSeasonStr + "_" + str(team1) + "_" + str(team2)
        pred = 0.5
        data = {'ID': [ID], 'Pred': [pred]}
        newRow = pd.DataFrame(data)
        df_submission = pd.concat([df_submission, newRow], ignore_index=True)

print(df_submission)
df_submission.to_csv("MSampleSubmissionStage2.csv", index=False)