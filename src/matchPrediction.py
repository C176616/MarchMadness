import os
import pandas as pd


class MatchPrediction:
    """ 
    This is a class that represents a match prediction between two teams. It is
    mostly a utility class for interfacing with kaggle inputs and expected
    outputs 
    
    Parameteres
    -----------
    ID : str
        The ID of the matchup. Kaggle format, e.g. "2022_1105_1256" representing
        "year", "team1", "team2"

    Attributes
    -----------
    ID : str
        The ID of the matchup. Kaggle format, e.g. "2022_1105_1256" representing
        "year", "team1", "team2"
    IDArray : :obj:'list' of str
        The year of the matchup
    team1ID : str
        Team 1's ID
    team2ID : str
        Team 2's ID
    pred : float
        A decimal 0-1 that represents the percent chance for team 1 to win.
    """

    def __init__(self, ID, pred):
        self.ID = ID
        IDArray = ID.split("_")
        self.year = IDArray[0]
        self.team1ID = IDArray[1]
        self.team2ID = IDArray[2]
        self.pred = pred

    """ Returns the ID back in kaggle string format
    """

    def getID(self):
        print("2022" + "_" + str(self.team1ID) + "_" + str(self.team2ID))

    def __str__(self):
        return self.ID