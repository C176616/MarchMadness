import pytest
from src.team import Team
from src.game import Game
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

def test_setup(setup):
    assert setup.root.left == setup.root.left.left.parent

def test_reverseLevelOrder(setup):
    setup.reverseLevelOrder() == []

def test_getMatchPrediction(setup):
    pass