import pytest
import sys

from src.game import Game
from src.team import Team
from src.tournament import Tournament


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

    tourn = Tournament(root)
    yield tourn


def test_iterator(setup):
    myiter = iter(setup)
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

    assert list == ['R4W1', 'R4X1', 'R4Y1', 'R4Z1', 'R5WX', 'R5YZ', 'R6CH']


def test_getNode(setup):
    assert setup.getNode('R4Y1').value == 'R4Y1'


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