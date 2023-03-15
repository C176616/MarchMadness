from src.game import Game
from src.tournament import Tournament
import json
import jsonpickle
from itertools import cycle
import pandas as pd
import os
from src.team import Team


def initializeTournament():
    cwd = os.getcwd()

    season = 2023

    slotcsv = cwd + "\\data_stage2\\MNCAATourneySlots.csv"
    seedcsv = cwd + "\\data_stage2\\MNCAATourneySeeds.csv"
    namecsv = cwd + "\\data_stage2\\MTeams.csv"
    predictionscsv = cwd + "\\data_stage2\\MSampleSubmissionStage2.csv"

    df_slots = pd.read_csv(slotcsv)
    df_slots = df_slots[df_slots["Season"] == season]

    df_seeds = pd.read_csv(seedcsv)
    df_seeds = df_seeds[df_seeds["Season"] == season]

    df_names = pd.read_csv(namecsv)

    df_stage1Combinations = pd.read_csv(predictionscsv)
    # tourn.populatePredictionsList()
    df_comb = df_seeds.merge(df_names[['TeamID', 'TeamName']],
                             left_on='TeamID',
                             right_on='TeamID')[['Seed', 'TeamID', 'TeamName']]
    df_comb2 = df_slots.merge(df_comb, left_on="StrongSeed", right_on="Seed")[[
        'Slot',
        'StrongSeed',
        'WeakSeed',
        'TeamID',
        'TeamName',
    ]]
    df_comb2 = df_comb2.rename(columns={
        'TeamID': 'Team1ID',
        'TeamName': 'Team1Name'
    })
    df_comb3 = df_comb2.merge(df_comb,
                              how='left',
                              left_on="WeakSeed",
                              right_on="Seed")[[
                                  'Slot',
                                  'StrongSeed',
                                  'WeakSeed',
                                  'Team1ID',
                                  'Team1Name',
                                  'TeamID',
                                  'TeamName',
                              ]]
    df_info = df_comb3.rename(columns={
        'TeamID': 'Team2ID',
        'TeamName': 'Team2Name'
    })

    print(df_comb3)
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

    root.left.left.left = Game('R3W1')
    root.left.left.left.parent = root.left.left
    root.left.left.right = Game('R3W2')
    root.left.left.right.parent = root.left.left
    root.left.right.left = Game('R3X1')
    root.left.right.left.parent = root.left.right
    root.left.right.right = Game('R3X2')
    root.left.right.right.parent = root.left.right
    root.right.left.left = Game('R3Y1')
    root.right.left.left.parent = root.right.left
    root.right.left.right = Game('R3Y2')
    root.right.left.right.parent = root.right.left
    root.right.right.left = Game('R3Z1')
    root.right.right.left.parent = root.right.right
    root.right.right.right = Game('R3Z2')
    root.right.right.right.parent = root.right.right

    root.left.left.left.left = Game('R2W1')
    root.left.left.left.left.parent = root.left.left.left
    root.left.left.left.right = Game('R2W4')
    root.left.left.left.right.parent = root.left.left.left
    root.left.left.right.left = Game('R2W3')
    root.left.left.right.left.parent = root.left.left.right
    root.left.left.right.right = Game('R2W2')
    root.left.left.right.right.parent = root.left.left.right

    root.left.right.left.left = Game('R2X1')
    root.left.right.left.left.parent = root.left.right.left
    root.left.right.left.right = Game('R2X4')
    root.left.right.left.right.parent = root.left.right.left
    root.left.right.right.left = Game('R2X3')
    root.left.right.right.left.parent = root.left.right.right
    root.left.right.right.right = Game('R2X2')
    root.left.right.right.right.parent = root.left.right.right

    root.right.left.left.left = Game('R2Y1')
    root.right.left.left.left.parent = root.right.left.left
    root.right.left.left.right = Game('R2Y4')
    root.right.left.left.right.parent = root.right.left.left
    root.right.left.right.left = Game('R2Y3')
    root.right.left.right.left.parent = root.right.left.right
    root.right.left.right.right = Game('R2Y2')
    root.right.left.right.right.parent = root.right.left.right

    root.right.right.left.left = Game('R2Z1')
    root.right.right.left.left.parent = root.right.right.left
    root.right.right.left.right = Game('R2Z4')
    root.right.right.left.right.parent = root.right.right.left
    root.right.right.right.left = Game('R2Z3')
    root.right.right.right.left.parent = root.right.right.right
    root.right.right.right.right = Game('R2Z2')
    root.right.right.right.right.parent = root.right.right.right

    root.left.left.left.left.left = Game('R1W1')
    root.left.left.left.left.left.parent = root.left.left.left.left
    root.left.left.left.left.right = Game('R1W8')
    root.left.left.left.left.right.parent = root.left.left.left.left
    root.left.left.left.right.left = Game('R1W5')
    root.left.left.left.right.left.parent = root.left.left.left.right
    root.left.left.left.right.right = Game('R1W4')
    root.left.left.left.right.right.parent = root.left.left.left.right
    root.left.left.right.left.left = Game('R1W6')
    root.left.left.right.left.left.parent = root.left.left.right.left
    root.left.left.right.left.right = Game('R1W3')
    root.left.left.right.left.right.parent = root.left.left.right.left
    root.left.left.right.right.left = Game('R1W7')
    root.left.left.right.right.left.parent = root.left.left.right.right
    root.left.left.right.right.right = Game('R1W2')
    root.left.left.right.right.right.parent = root.left.left.right.right

    root.left.right.left.left.left = Game('R1X1')
    root.left.right.left.left.left.parent = root.left.right.left.left
    root.left.right.left.left.right = Game('R1X8')
    root.left.right.left.left.right.parent = root.left.right.left.left
    root.left.right.left.right.left = Game('R1X5')
    root.left.right.left.right.left.parent = root.left.right.left.right
    root.left.right.left.right.right = Game('R1X4')
    root.left.right.left.right.right.parent = root.left.right.left.right
    root.left.right.right.left.left = Game('R1X6')
    root.left.right.right.left.left.parent = root.left.right.right.left
    root.left.right.right.left.right = Game('R1X3')
    root.left.right.right.left.right.parent = root.left.right.right.left
    root.left.right.right.right.left = Game('R1X7')
    root.left.right.right.right.left.parent = root.left.right.right.right
    root.left.right.right.right.right = Game('R1X2')
    root.left.right.right.right.right.parent = root.left.right.right.right

    root.right.left.left.left.left = Game('R1Y1')
    root.right.left.left.left.left.parent = root.right.left.left.left
    root.right.left.left.left.right = Game('R1Y8')
    root.right.left.left.left.right.parent = root.right.left.left.left
    root.right.left.left.right.left = Game('R1Y5')
    root.right.left.left.right.left.parent = root.right.left.left.right
    root.right.left.left.right.right = Game('R1Y4')
    root.right.left.left.right.right.parent = root.right.left.left.right
    root.right.left.right.left.left = Game('R1Y6')
    root.right.left.right.left.left.parent = root.right.left.right.left
    root.right.left.right.left.right = Game('R1Y3')
    root.right.left.right.left.right.parent = root.right.left.right.left
    root.right.left.right.right.left = Game('R1Y7')
    root.right.left.right.right.left.parent = root.right.left.right.right
    root.right.left.right.right.right = Game('R1Y2')
    root.right.left.right.right.right.parent = root.right.left.right.right

    root.right.right.left.left.left = Game('R1Z1')
    root.right.right.left.left.left.parent = root.right.right.left.left
    root.right.right.left.left.right = Game('R1Z8')
    root.right.right.left.left.right.parent = root.right.right.left.left
    root.right.right.left.right.left = Game('R1X5')
    root.right.right.left.right.left.parent = root.right.right.left.right
    root.right.right.left.right.right = Game('R1Z4')
    root.right.right.left.right.right.parent = root.right.right.left.right
    root.right.right.right.left.left = Game('R1Z6')
    root.right.right.right.left.left.parent = root.right.right.right.left
    root.right.right.right.left.right = Game('R1Z3')
    root.right.right.right.left.right.parent = root.right.right.right.left
    root.right.right.right.right.left = Game('R1Z7')
    root.right.right.right.right.left.parent = root.right.right.right.right
    root.right.right.right.right.right = Game('R1Z2')
    root.right.right.right.right.right.parent = root.right.right.right.right

    tourn = Tournament(root)
    # tourn.reverseLevelOrder()

    tourn.getNode('R1W1').right = Game('W16')
    tourn.getNode('R1W1').right.parent = tourn.getNode('R1W1')
    tourn.getNode('R1X1').right = Game('X16')
    tourn.getNode('R1X1').right.parent = tourn.getNode('R1X1')
    tourn.getNode('R1Y6').right = Game('Y11')
    tourn.getNode('R1Y6').right.parent = tourn.getNode('R1Y6')
    tourn.getNode('R1Z6').right = Game('Z11')
    tourn.getNode('R1Z6').right.parent = tourn.getNode('R1Z6')

    # tourn.reverseLevelOrder()

    preRound1Slots = ['W16', 'X16', 'Y11', 'Z11']
    cycleList = cycle(preRound1Slots)

    for x in range(3):
        i = next(cycleList)
        slot = df_info[df_info['Slot'] == i]
        tourn.getNode(i).team1 = Team(slot['StrongSeed'].values[0],
                                      slot['Team1ID'].values[0],
                                      slot['Team1Name'].values[0])
        tourn.getNode(i).team2 = Team(slot['WeakSeed'].values[0],
                                      slot['Team2ID'].values[0],
                                      slot['Team2Name'].values[0])

    tourn.populateTeams(df_info)

    jsonData = jsonpickle.encode(tourn)
    print(jsonData)

    out_file = open('tournamentLayout.json', "w")
    json.dump(jsonData, out_file)
    out_file.close()


if __name__ == "__main__":
    initializeTournament()