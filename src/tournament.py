import plotly.graph_objects as go
import plotly.express as px
import numpy as np

from collections import deque

from src.matchPrediction import MatchPrediction
from src.team import Team


class Tournament:
    """ 
    This is a class that represents a tournament bracket. It is essesntially
    representing a tournament as a binary tree. 
    
    Parameteres
    -----------
    root : :obj: 'Game'
        A Game that represents the champions (root) of the tournament tree

    Attributes
    -----------
    root : :obj: 'Game'
        A Game that represents the champions (root) of the tournament tree
    nodeList :obj:'list' :obj:'game"
        A list in reverse level traversal order of the games in the tournament.
        Meant to be used until an iterator was developed. Deprecated in future
        versions
    predictionsList : 

    """

    def __init__(self, root):
        self.root = root
        self.nodeList = []
        self.predictionsList = []

    def __iter__(self):
        """
        An iterator that yields the tournament tree in reverse level order

        Example - For loop:
            myiter = iter(tourn)
            for item in myiter:
                print(item)
        Example - next
            myiter = iter(tourn)
            item = next(myiter)
        """
        queue = deque()
        queue.append(self.root)
        stack = deque()

        while queue:
            curr = queue.popleft()
            stack.append(curr)

            if curr.right:
                queue.append(curr.right)

            if curr.left:
                queue.append(curr.left)

        while stack:
            yield stack.pop()

    def getNode(self, nodeValue):
        """ 
        Parameters
        ----------
        nodeValue : str
            a string of the value of the node to find

        Returns
        -------
        :obj: game the game node object that has the value of the given
            nodeValue
        """
        myiter = iter(self)
        for item in myiter:
            if item.value == nodeValue:
                return item

    def populateTeams(self, df_info):
        """This method runs through the first round of the tournament and
        populates the team IDs and names in each slot based on seed.

        Parameters
        ----------
        df_info : pandas dataframe
        ['Slot','StrongSeed','WeakSeed','Team1ID','Team1Name','Team2ID','Team2Name']
            A pandas dataframe that has tournamnet information of each team in
            each game node
        """
        mIter = iter(self)
        for game in mIter:
            # if the 2nd array position is a 1 e.g. 'R1W5'
            if [*game.value][1] == '1':
                slot = df_info[df_info['Slot'] == game.value]
                print(slot['StrongSeed'].values[0])
                if slot.empty != True:
                    game.team1 = Team(slot['StrongSeed'].values[0],
                                      slot['Team1ID'].values[0],
                                      slot['Team1Name'].values[0])
                    game.team2 = Team(slot['WeakSeed'].values[0],
                                      slot['Team2ID'].values[0],
                                      slot['Team2Name'].values[0])

    def populatePredictionsList(self, df_stage1Combinations):
        """This method imports the kaggle sample submission and populates the
        tournaments predictions with matchup predictions

        Parameters
        ----------
        df_stage1Combinations : pandas dataframe ['ID', 'Pred']
            a pandas dataframe that is used to interface with the kaggle
            competition
        """
        self.predictionsList = []
        for index, row in df_stage1Combinations.iterrows():
            self.predictionsList.append(MatchPrediction(
                row['ID'], row['Pred']))

    def getMatchPrediction(self, team1ID, team2ID, upsets):
        """_summary_

        Parameters
        ----------
        team1ID : int
            team 1 ID of the prediction
        team2ID : int
            team 2 ID of the prediction
        upsets : bool
            True or False if upsets are enabled or not

        Returns
        -------
        'obj' list[int, float]
            the result, 1 if team 1 wins, 0 if team 2 wins. And the predicted
            value by which team 1 wins (between 0 and 1 expressed as a percent)
        """
        gameID1 = '2022_' + str(int(team1ID)) + '_' + str(int(team2ID))
        gameID2 = '2022_' + str(int(team2ID)) + '_' + str(int(team1ID))

        if upsets == True:
            randResult = np.random.random()

            for i in self.predictionsList:
                if i.ID == gameID1:
                    if i.pred > randResult:
                        result = 0
                        return (result, i.pred)
                    else:
                        result = 1
                        return (result, i.pred)

            for i in self.predictionsList:
                if i.ID == gameID2:
                    if i.pred < randResult:
                        result = 1
                        return (result, i.pred)
                    else:
                        result = 0
                        return (result, i.pred)

        else:
            for i in self.predictionsList:
                if i.ID == gameID1:
                    if i.pred > 0.5:
                        result = 1
                        return (result, i.pred)
                    else:
                        result = 0
                        return (result, i.pred)

                elif i.ID == gameID2:
                    if i.pred > 0.5:
                        result = 0
                        return (result, round(1 - i.pred, 2))
                    else:
                        result = 1
                        return (result, round(1 - i.pred, 2))

    def simulateTournament(self):
        """travels each game node, calculates a predicted winner and elevates
        that winner to the game node's parent game node based on if it is
        either a left or right branch.
        """
        mIter = iter(self)
        for game in mIter:
            print(game)
            result = self.getMatchPrediction(game.team1.teamID,
                                             game.team2.teamID, False)
            game.winPct = result[1]

            # if this is the final game
            if (game.parent == None):
                if (result[0] == 1):
                    print(1)
                    game.winner = game.team1
                elif (result[0] == 0):
                    print(2)
                    game.winner = game.team2

            # if this game is a left branch, it becomes the next game's team 1
            elif (game == game.parent.left):
                if (result[0] == 1):
                    print(3)
                    game.winner = game.team1
                    game.parent.team1 = game.team1
                elif (result[0] == 0):
                    game.winner = game.team2
                    game.parent.team1 = game.team2

            # if this game is a right branch, it becomes the next game's team 2
            elif (game == game.parent.right):
                if (result[0] == 1):
                    game.winner = game.team2
                    game.parent.team2 = game.team1
                elif (result[0] == 0):
                    game.winner = game.team1
                    game.parent.team2 = game.team2
            else:
                pass
            # self.reverseLevelOrder()
