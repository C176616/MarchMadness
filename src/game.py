class Game():
    def __init__(self, value, left=None, right=None, parent=None):
        self.value = value
        self.left = left
        self.right = right
        self.parent = parent
        self.winnerID = 0
        self.winner = None
        self.team1 = None
        self.team2 = None
        winPct = 0

    def __str__(self):
        return self.value