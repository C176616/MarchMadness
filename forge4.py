import pytest
import sys

from src.game import Game
from src.team import Team
from src.tournament import Tournament

import pandas as pd

root = Game('R6CH')
root.left = Game('R5WX')
root.left.parent = root
root.right = Game('R5YZ')
root.right.parent = root

root.left.left = Game('R4W1')
root.left.left.parent = root.left
root.left.right = Game('R4X1')
root.left.right.parent = root.left
root.right.left = Game('R4Y1')
root.right.left.parent = root.right
root.right.right = Game('R4Z1')
root.right.right.parent = root.right
root.left.left.left = Game('R1W1')
root.left.left.left.parent = root.left.left

tourn = Tournament(root)

df_info = pd.DataFrame([{
    'Slot': 'R1W1',
    'StrongSeed': 'W1',
    'WeakSeed': 'W16',
    'Team1ID': 1135,
    'Team1Name': 'Rose Hulman',
    'Team2ID': 1136,
    'Team2Name': 'Purdue'
}, {
    'Slot': 'R4W1',
    'StrongSeed': 'R1W1',
    'WeakSeed': 'W15',
    'Team2ID': 1143,
    'Team2Name': 'Valpo'
}, {
    'Slot': 'R4X1',
    'StrongSeed': 'X1',
    'WeakSeed': 'X16',
    'Team1ID': 1137,
    'Team1Name': 'ISU',
    'Team2ID': 1138,
    'Team2Name': 'Notre Dame'
}, {
    'Slot': 'R4Y1',
    'StrongSeed': 'Y1',
    'WeakSeed': 'Y16',
    'Team1ID': 1139,
    'Team1Name': 'USI',
    'Team2ID': 1140,
    'Team2Name': 'UE'
}, {
    'Slot': 'R4Z1',
    'StrongSeed': 'Z1',
    'WeakSeed': 'Z16',
    'Team1ID': 1141,
    'Team1Name': 'Ivy Tech',
    'Team2ID': 1142,
    'Team2Name': 'Ball State'
}])

print(df_info)

tourn.populateTeams(df_info)
tourn.getNode('R4W1').team2 = Team('R1W1', 1143, 'Valpo')

tourn.getNode('R4X1').team1 = Team('X1', 1137, 'ISU')
tourn.getNode('R4X1').team2 = Team('X16', 1138, 'ND')

tourn.getNode('R4Y1').team1 = Team('Y1', 1139, 'USI')
tourn.getNode('R4Y1').team2 = Team('Y16', 1140, 'UE')

tourn.getNode('R4Z1').team1 = Team('Z1', 1141, 'IVY Tech')
tourn.getNode('R4Z1').team2 = Team('Z16', 1142, 'Ball State')

df_stage1Combinations = pd.DataFrame(columns=['ID', 'Pred'])
for i in range(1134, 1144):
    for j in range(1134, 1144):
        data = {'ID': ['2022_' + str(i) + '_' + str(j)], 'Pred': [0.5]}
        df_newRow = pd.DataFrame(data)
        df_stage1Combinations = pd.concat([df_stage1Combinations, df_newRow],
                                          ignore_index=True)

tourn.populatePredictionsList(df_stage1Combinations)
print(tourn.predictionsList)

tourn.simulateTournament()