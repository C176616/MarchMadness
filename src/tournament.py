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

    def __iter__(self):
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

    """Generates
    """

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
        for game in self.nodeList:
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
        self.predictionsList = []
        for index, row in df_stage1Combinations.iterrows():
            self.predictionsList.append(MatchPrediction(
                row['ID'], row['Pred']))

    def getMatchPrediction(self, team1ID, team2ID):
        gameID1 = '2022_' + str(int(team1ID)) + '_' + str((team2ID))
        gameID2 = '2022_' + str(int(team2ID)) + '_' + str(int(team1ID))
        # print("teamID",gameID1)

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
                    result = 0
                    return (result, i.pred)
                else:
                    result = 1
                    return (result, i.pred)

    def simulateTournament(self):
        for game in self.nodeList:
            # print("game:", game.value, game.team1, game.team2)
            result = self.getMatchPrediction(int(game.team1.teamID),
                                             int(game.team2.teamID))
            # print("result: ",result[0], "chance:", result[1])
            game.winPct = result[1]
            if (game.parent == None):
                # print("no parents")
                if (result[0] == 1):
                    print(1)
                    game.winner = game.team1
                elif (result[0] == 0):
                    print(2)
                    game.winner = game.team2
            elif (game == game.parent.left):
                # print('left')
                if (result[0] == 1):
                    # print(1)
                    game.winner = game.team1
                    game.parent.team1 = game.team1

                elif (result[0] == 0):
                    game.winner = game.team2
                    game.parent.team1 = game.team2

            elif (game == game.parent.right):
                if (result[0] == 1):
                    game.winner = game.team2
                    game.parent.team2 = game.team1

                elif (result[0] == 0):
                    game.winner = game.team1
                    game.parent.team2 = game.team2

            else:
                pass

            self.reverseLevelOrder()
