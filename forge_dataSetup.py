import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pkg_resources

from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

cwd = os.getcwd()

currentYear = 2023
targetYear = 2003

# define functions

# Existing code from data manipulation section. Only run if needed.

# tourney_cresults = pd.read_csv(cwd + '/data/MNCAATourneyCompactResults.csv')
# seeds = pd.read_csv(cwd + '/data/MNCAATourneySeeds.csv')
seeds = pd.read_csv(cwd + '/data_stage2/MNCAATourneySeeds.csv')
# seeds = seeds[seeds['Season'] == currentYear]
seeds['Seed'] = pd.to_numeric(seeds['Seed'].str[1:3],
                              downcast='integer',
                              errors='coerce')
# print(seeds)
season_dresults = pd.read_csv(cwd +
                              '/data_stage2/MRegularSeasonDetailedResults.csv')

# tourney_cresults = tourney_cresults.loc[tourney_cresults['Season'] >= targetYear]

# training_set = pd.read_csv("training_set.csv")
record = pd.read_csv('record.csv')


def delta_seed(row):
    cond = (seeds['Season'] == row['Season'])
    return seeds[cond
                 & (seeds['TeamID'] == row['Team1'])]['Seed'].iloc[0] - seeds[
                     cond & (seeds['TeamID'] == row['Team2'])]['Seed'].iloc[0]

    # cond = (seeds['Season'] == row['Season'])
    # return seeds[cond
    #              & (seeds['TeamID'] == row['Team1'])]['Seed'].iloc[0] - seeds[
    #                  cond & (seeds['TeamID'] == row['Team2'])]
    # cond = (seeds['Season'] == row['Season'])
    # try:
    #     x = seeds[cond
    #               & (seeds['TeamID'] == row['Team1'])]['Seed'].iloc[0] - seeds[
    #                   cond & (seeds['TeamID'] == row['Team2'])]['Seed'].iloc[0]
    # except IndexError:
    #     x = 100
    # return x

    team1Seed = seeds[seeds['TeamID'] == row['Team1']]
    # return team1Seed
    # print(team1Seed)
    # team2Seed = seeds[seeds['TeamID'] == row['Team2']]['Seed']
    # diff = team1Seed - team2Seed
    # print(diff)

    # return diff
    # cond = (seeds['Season'] == row['Season'])
    # return seeds[cond
    #              & (seeds['TeamID'] == row['Team1'])]['Seed'].iloc[0] - seeds[
    #                  cond & (seeds['TeamID'] == row['Team2'])]['Seed'].iloc[0]


# function to, given a row, calculate what the difference between the two seeds was.
#Function to look up
def delta_winPct(row):
    cond1 = (record['Season'] == row['Season']) & (record['WTeamID']
                                                   == row['Team1'])
    cond2 = (record['Season'] == row['Season']) & (record['WTeamID']
                                                   == row['Team2'])
    return (record[cond1]['wins'] / record[cond1]['games']).mean() - (
        record[cond2]['wins'] / record[cond2]['games']).mean()


def get_points_against(row):
    wcond = (dfW['Season'] == row['Season']) & (dfW['WTeamID']
                                                == row['WTeamID'])
    fld1 = 'LScore'
    lcond = (dfL['Season'] == row['Season']) & (dfL['LTeamID']
                                                == row['WTeamID'])
    fld2 = 'WScore'
    retVal = dfW[wcond][fld1].sum()
    if len(dfL[lcond][fld2]) > 0:
        retVal = retVal + dfL[lcond][fld2].sum()
    return retVal


def get_points_for(row):
    wcond = (dfW['Season'] == row['Season']) & (dfW['WTeamID']
                                                == row['WTeamID'])
    fld1 = 'WScore'
    lcond = (dfL['Season'] == row['Season']) & (dfL['LTeamID']
                                                == row['WTeamID'])
    fld2 = 'LScore'
    retVal = dfW[wcond][fld1].sum()
    if len(dfL[lcond][fld2]) > 0:
        retVal = retVal + dfL[lcond][fld2].sum()
    return retVal


def get_remaining_stats(row, field):
    wcond = (dfW['Season'] == row['Season']) & (dfW['WTeamID']
                                                == row['WTeamID'])
    fld1 = 'W' + field
    lcond = (dfL['Season'] == row['Season']) & (dfL['LTeamID']
                                                == row['WTeamID'])
    fld2 = 'L' + field
    retVal = dfW[wcond][fld1].sum()
    if len(dfL[lcond][fld2]) > 0:
        retVal = retVal + dfL[lcond][fld2].sum()
    return retVal


def delta_stat(row, field):
    cond1 = (record['Season'] == row['Season']) & (record['WTeamID']
                                                   == row['Team1'])
    cond2 = (record['Season'] == row['Season']) & (record['WTeamID']
                                                   == row['Team2'])
    return (record[cond1][field] / record[cond1]['games']).mean() - (
        record[cond2][field] / record[cond2]['games']).mean()


sub = pd.read_csv(cwd + '/data_stage2/MSampleSubmissionStage2.csv')

sub = sub.replace(r'2022', '2023', regex=True)
# print(sub)

sub['Season'], sub['Team1'], sub['Team2'] = sub['ID'].str.split('_').str
sub[['Season', 'Team1', 'Team2']] = sub[['Season', 'Team1',
                                         'Team2']].apply(pd.to_numeric)
print(sub)

sub['deltaSeed'] = sub.apply(delta_seed, axis=1)

print(sub)
print(seeds)
# sub = sub[sub['deltaSeed'] != 100]

# sub['deltaWinPct'] = sub.apply(delta_winPct, axis=1)
# rawCols = [
#     'PointsFor', 'PointsAgainst', 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA',
#     'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF'
# ]

# for rawCol in rawCols:
#     print("Processing", rawCol)
#     sub['delta' + rawCol] = sub.apply(delta_stat, args=(rawCol, ), axis=1)

print(sub)

sub.to_csv("training_set_stage2_2.csv", index=False)