class Game():
    """ 
    This is a class that represents a game in a tournament tree. Each game acts
    as a node in the binary tree.
    
    Parameteres
    -----------
    value : str
        A representation of the node. Follows kaggle convention for a game e.g.
        'R2W4'
    left : :obj: 'Game', optional
        The left game of the node
    right : :obj: 'Game', optional
        The right game of the node
    parent : :obj: 'Game', optional
        The parent game of the node

    Attributes
    -----------
    value : str
        A representation of the node. Follows kaggle convention for a game e.g.
        'R2W4'
    left : :obj: 'Game'
        The left game of the node
    right : :obj: 'Game'
        The right game of the node
    parent : :obj: 'Game'
        The parent game of the node
    winner : :obj: 'Team'
        The team that wins this game node
    team1 : :obj: 'Team'
        The team designated as team 1 in this game node
    team2 : :obj: 'Team'
        The team designated as team 2 in this game node
    winPct : float
        A number 0-1 that represents the percent chance that team1 wins the game

    """

    def __init__(self, value, left=None, right=None, parent=None):
        self.value = value
        self.left = left
        self.right = right
        self.parent = parent
        self.winner = None
        self.team1 = None
        self.team2 = None
        self.winPct = 0

    def __str__(self):
        return self.value

    def getString(self):
        return self.value
