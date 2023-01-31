import pytest
from src.team import Team

@pytest.fixture
def setup():
    mTeam = Team('W13',1123,'RHIT')
    yield mTeam

def test_getID(setup):
   assert(setup.getID,1123)

def test_getString(setup):
    assert(setup.getString, 'RHIT W13')