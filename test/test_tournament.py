import pytest
import sys

from src.game import Game
from src.team import Team
from src.tournament import Tournament

import pandas as pd


@pytest.fixture
def setup():
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
    yield tourn


def test_iterator(setup):
    myiter = iter(setup)
    assert next(myiter).value == 'R1W1'
    assert next(myiter).value == 'R4W1'
    assert next(myiter).value == 'R4X1'
    assert next(myiter).value == 'R4Y1'
    assert next(myiter).value == 'R4Z1'
    assert next(myiter).value == 'R5WX'
    assert next(myiter).value == 'R5YZ'
    assert next(myiter).value == 'R6CH'


def test_iterator2(setup):
    myiter2 = iter(setup)
    list = []
    for item in myiter2:
        list.append(item.value)

    assert list == [
        'R1W1', 'R4W1', 'R4X1', 'R4Y1', 'R4Z1', 'R5WX', 'R5YZ', 'R6CH'
    ]


def test_getNode(setup):
    assert setup.getNode('R4Y1').value == 'R4Y1'


def test_populateTeams(setup):
    df_info = pd.DataFrame(
        {
            'Slot': ['R1W1'],
            'StrongSeed': ['W1'],
            'WeakSeed': ['W16'],
            'Team1ID': [1135],
            'Team1Name': ['Rose Hulman'],
            'Team2ID': [1136],
            'Team2Name': ['Purdue']
        }, {
            'Slot': ['R4X1'],
            'StrongSeed': ['X1'],
            'WeakSeed': ['X16'],
            'Team1ID': [1137],
            'Team1Name': ['ISU'],
            'Team2ID': [1138],
            'Team2Name': ['Notre Dame']
        })

    setup.populateTeams(df_info)
    assert (setup.getNode('R1W1').team1.teamID == 1135)
    assert (setup.getNode('R1W1').team1.teamName == 'Rose Hulman')
    assert (setup.getNode('R1W1').team1.teamSeed == 'W1')
    assert (setup.getNode('R1W1').team2.teamID == 1136)
    assert (setup.getNode('R1W1').team2.teamName == 'Purdue')
    assert (setup.getNode('R1W1').team2.teamSeed == 'W16')


def test_populatePredictionsList(setup):
    df_stage1Combinations = pd.DataFrame({
        'ID': ['2022_1135_1136'],
        'Pred': [0.95]
    })
    setup.populatePredictionsList(df_stage1Combinations)
    assert (setup.predictionsList[0].team1ID == '1135')
    assert (setup.predictionsList[0].team2ID == '1136')
    assert (setup.predictionsList[0].pred == 0.95)


def test_getMatchPrediction(setup):
    df_stage1Combinations = pd.DataFrame({
        'ID': ['2022_1135_1136'],
        'Pred': [0.95]
    })
    setup.populatePredictionsList(df_stage1Combinations)

    assert (setup.getMatchPrediction('1135', '1136', False) == (1, 0.95))
    assert (setup.getMatchPrediction('1136', '1135', False) == (0, 0.05))

    df_stage1Combinations = pd.DataFrame({
        'ID': ['2022_1136_1135'],
        'Pred': [0.05]
    })
    setup.populatePredictionsList(df_stage1Combinations)
    assert (setup.getMatchPrediction('1135', '1136', False) == (1, 0.95))
    assert (setup.getMatchPrediction('1136', '1135', False) == (0, 0.05))
    assert (setup.getMatchPrediction('1136', '1135', False) == (0, 0.05))
    assert (setup.getMatchPrediction('1135', '1136', False) == (1, 0.95))


def test_simulateTournament(setup):
    df_info = pd.DataFrame([{
        'Slot': 'R1W1',
        'StrongSeed': 'W1',
        'WeakSeed': 'W16',
        'Team1ID': 1135,
        'Team1Name': 'Rose Hulman',
        'Team2ID': 1136,
        'Team2Name': 'Purdue'
    }])
    print(df_info)

    setup.populateTeams(df_info)
    setup.getNode('R4W1').team2 = Team('R1W1', 1143, 'Valpo')

    setup.getNode('R4X1').team1 = Team('X1', 1137, 'ISU')
    setup.getNode('R4X1').team2 = Team('X16', 1138, 'ND')

    setup.getNode('R4Y1').team1 = Team('Y1', 1139, 'USI')
    setup.getNode('R4Y1').team2 = Team('Y16', 1140, 'UE')

    setup.getNode('R4Z1').team1 = Team('Z1', 1141, 'IVY Tech')
    setup.getNode('R4Z1').team2 = Team('Z16', 1142, 'Ball State')

    df_stage1Combinations = pd.DataFrame(columns=['ID', 'Pred'])
    for i in range(1134, 1144):
        for j in range(1134, 1144):
            data = {'ID': ['2022_' + str(i) + '_' + str(j)], 'Pred': [0.5]}
            df_newRow = pd.DataFrame(data)
            df_stage1Combinations = pd.concat(
                [df_stage1Combinations, df_newRow], ignore_index=True)

    setup.populatePredictionsList(df_stage1Combinations)
    print(setup.predictionsList)
    setup.simulateTournament()
    assert (setup.getNode('R6CH').winner != None)


# assert (setup.getNode('R4W1').team1.teamID == 1135)
# assert (setup.getNode('R1W1').team1.teamID == '1135')

# def test_setup(setup):
#     assert setup.root.left == setup.root.left.left.parent

# # def test_reverseLevelOrder(setup):
# #     setup.reverseLevelOrder() == []

# def test_printGivenLevel(setup):
#     pass

# def test_calculateHeight(setup):
#     assert setup.calculateHeight(setup.root) == 3

# def getNode(setup):
#     assert setup.getNode('R4Y1').value == 'R4Y1'

# def test_populateTeams(setup):
#     pass

# def populatePredictionsList(setup):
#     pass

# def test_getMatchPrediction(setup):
#     pass

# def test_simulateTournament(setup):
#     pass