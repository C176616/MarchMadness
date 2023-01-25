from src import team

class Game():
    def __init__(self, value, left=None, right=None, parent=None):
        self.value = value
        self.left = left
        self.right = right
        self.parent = parent
        self.winnerID = 0
        self.team1 = team.Team()
        self.team2 = team.Team()
        self.team1Seed = ""
        self.team1ID = 0
        self.team1Name = "" 
        self.team2Seed = ""
        self.team2ID = 0
        self.team2Name = ""

    def __str__(self):
        return self.value