class Team():
    """ 
    This is a class that represents a team throughout the tournament.
    
    Parameteres
    -----------
    seed : str
        A representation of the seed. Follows kaggle convention for a seed e.g.
        'W4'
    ID : int
        The team's ID designated by kaggle
    name : string
        Human readable name of the team

    Attributes
    -----------
    seed : str
        A representation of the seed. Follows kaggle convention for a seed e.g.
        'W4'
    ID : int
        The team's ID designated by kaggle
    name : string
        Human readable name of the team

    """

    def __init__(self, seed, ID, name):
        self.teamSeed = seed
        self.teamID = ID
        self.teamName = name

    def __str__(self):
        return str(self.teamName)

    ''' Returns the ID of the team
    '''

    def getID(self):
        return self.teamName

    ''' Returns the string representation of the team to be displayed on the
        bracket image
    '''

    def getString(self):
        return (str(self.teamName) + " " + str(self.teamSeed))
