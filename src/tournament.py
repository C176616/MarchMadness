import plotly.graph_objects as go
import plotly.express as px
from src.matchPrediction import MatchPrediction
import numpy as np
from src.team import Team

class Tournament:
    def __init__(self,root):
        self.root = root
        self.nodeList = [] 
        self.predictionsList = []
        
    def reverseLevelOrder(self):
        self.nodeList = []
        h = self.calculateHeight(self.root)
        for i in reversed(range(1, h+1)):
            self.printGivenLevel(self.root,i)
            
    def printGivenLevel(self, root, level): 
        if root is None:
            return
        if level ==1:
            self.nodeList.append(root)

        elif level>1:
            self.printGivenLevel(root.left, level-1)
            self.printGivenLevel(root.right, level-1)

    def calculateHeight(self, node):
        if node is None:
            return 0
        else:
            # Compute the height of each subtree
            lheight = self.calculateHeight(node.left)
            rheight = self.calculateHeight(node.right)

            # Use the larger one
            if lheight > rheight :
                return lheight + 1
            else:
                return rheight + 1
            
    def getNode(self, node):
        for i in self.nodeList:
            if i.value==node:
                return i        

    def populateTeams(self, df_info):
        for game in self.nodeList:
            if [*game.value][1] == '1':
                slot = df_info[df_info['Slot']==game.value]
                print(slot['StrongSeed'].values[0])                          
                if slot.empty != True:
                    game.team1 = Team(slot['StrongSeed'].values[0],slot['Team1ID'].values[0],slot['Team1Name'].values[0])
                    game.team2 = Team(slot['WeakSeed'].values[0],slot['Team2ID'].values[0],slot['Team2Name'].values[0])
    
    # def populateTeams(self, df_slots, df_seeds, df_names):
    #     for game in self.nodeList:
    #         game.team1Seed = df_slots[df_slots['Slot']==game.value]['StrongSeed'].values[0]
    #         game.team2Seed = df_slots[df_slots['Slot']==game.value]['WeakSeed'].values[0]
    #         if [*game.value][1] == '1':
    #             a = df_seeds[df_seeds['Seed']==game.team1Seed]
    #             b = df_seeds[df_seeds['Seed']==game.team2Seed]
    #             if a.empty != True:
    #                 game.team1ID = a['TeamID'].values[0]
    #             if b.empty != True:
    #                 game.team2ID = b['TeamID'].values[0]

    #             a = df_names[df_names['TeamID']==game.team1ID]
    #             b = df_names[df_names['TeamID']==game.team2ID]
    #             if a.empty != True:
    #                 game.team1Name = a['TeamName'].values[0]
    #             if b.empty != True:
    #                 game.team2Name = b['TeamName'].values[0]
                
    def flatten(self):
        pass

    def populatePredictionsList(self, df_stage1Combinations):
        self.predictionsList = []
        for index, row in df_stage1Combinations.iterrows():
            self.predictionsList.append(MatchPrediction(row['ID'], row['Pred']))
    
    def getMatchPrediction(self, team1ID, team2ID):        
        gameID1 = '2022_' + str(int(team1ID))+ '_' +str((team2ID))
        gameID2 = '2022_' + str(int(team2ID))+ '_' +str(int(team1ID))    
        print("teamID",gameID1)

        randResult = np.random.random()

        for i in self.predictionsList:
            if i.ID == gameID1:
                if i.pred > randResult:
                    result = 0
                    return(result,i.pred)
                else:
                    result = 1
                    return(result,i.pred)

        for i in self.predictionsList:
            if i.ID == gameID2:
                if i.pred < randResult:
                    result = 0
                    return (result, i.pred)
                else:
                    result = 1
                    return(result,i.pred)

