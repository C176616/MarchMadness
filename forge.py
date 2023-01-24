import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
import os

import numpy as np

import plotly.io as pio

pio.renderers.deafult = "notebook"

cwd = os.getcwd()

class Team():
    def __init__(self, teamSeed):
        self.TeamName
        self.TeamSeed
        self.TeamID
    
    def __str__(self):
        return self.TeamName

class Node2():
    def __init__(self, value, left=None, right=None, parent=None):
        self.value = value
        self.left = left
        self.right = right
        self.parent = parent
        self.winnerID = 0
        self.team1 = None
        self.team2 = None
        self.team1Seed = ""
        self.team1ID = 0
        self.team1Name = "" 
        self.team2Seed = ""
        self.team2ID = 0
        self.team2Name = ""

    def __str__(self):
        return self.value

class MatchPrediction:
    def __init__(self,ID,pred):
        self.ID = ID
        IDArray = ID.split("_")
        self.year = IDArray[0]
        self.team1ID=IDArray[1]
        self.team2ID=IDArray[2]
        self.pred=pred
    
    def getID(self):
        print("2022"+"_"+str(self.team1ID)+"_"+str(self.team2ID))
        
    def __str__(self):
        return self.ID

class Tournament:
    def __init__(self,root):
        self.root = root
        self.nodeList = [] 
        self.predictionsList = []
        
    def reverseLevelOrder(self):
        self.nodeList = []
        h = self.calculateHeight(root)
        for i in reversed(range(1, h+1)):
            self.printGivenLevel(root,i)
            
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
#             print(i)
            if i.value==node:
                return i
        
    def populateSlots(self,csv,season):
        tourneySlots = pd.read_csv(csv)
        tourneySlots = tourneySlots[tourneySlots["Season"]==season]
        
        for node in self.nodeList:
            a = tourneySlots[tourneySlots['Slot']==node.value]
            node.team1Seed = a['StrongSeed'].values[0]
            node.team2Seed = a['WeakSeed'].values[0]
    
    def populateTeamIDs(self,csv,season):
        tourneySeeds = pd.read_csv(csv)
        tourneySeeds = tourneySeeds[tourneySeeds['Season']==season]
        tourneySeeds.head(30)
        
#         for i in range(0,31):
        for node in self.nodeList:
#             print(node.team1Seed)
#             if [*tourn.nodeList[5].value][1] != '2':
#             print(node.value)
            if [*node.value][1] != '2' and [*node.value][1] != '3' and [*node.value][1] != '4' and [*node.value][1] != '5' and [*node.value][1] != '6':
#                 print(node.value)
#                 print(node.team1Seed)
#                 print(node.team2Seed)
                a = tourneySeeds[tourneySeeds['Seed']==node.team1Seed]
                b = tourneySeeds[tourneySeeds['Seed']==node.team2Seed]
#                 print(b)
                if(a.empty):
                    print('a empty')
                if(b.empty):
                    print('b empty')
#                 print(a['TeamID'].values[0])
#                 print(b['TeamID'].values[0])
                if a.empty != True:
                    node.team1ID = a['TeamID'].values[0]
                if b.empty != True:
                    node.team2ID = b['TeamID'].values[0]
                    
    def populateMatchPairs(self, csv):
        pass
    
    def populateTeamNames(self, csv):
        teamNames = pd.read_csv(csv)
        
        for node in self.nodeList:
            if [*node.value][1] != '2' and [*node.value][1] != '3' and [*node.value][1] != '4' and [*node.value][1] != '5' and [*node.value][1] != '6':
                a = teamNames[teamNames['TeamID']==node.team1ID]
                b = teamNames[teamNames['TeamID']==node.team2ID]
                if a.empty != True:
                    node.team1Name = a['TeamName'].values[0]
                if b.empty != True:
                    node.team2Name = b['TeamName'].values[0]
    
#     def populateStage1Combinations(self, csv):
#         stage1Combinations = pd.read_csv(cwd+csv)

    def populatePredictionsList(self, csv):
        stage1Combinations = pd.read_csv(csv)
        self.predictionsList = []
        for index, row in stage1Combinations.iterrows():
            self.predictionsList.append(MatchPrediction(row['ID'], row['Pred']))
            
    def getMatchPrediction(self, team1ID, team2ID):
        gameID1 = '2022_' + str(team1ID)+ '_' +str(team2ID)
        gameID2 = '2022_' + str(team2ID)+ '_' +str(team1ID)    
    
        randResult = np.random.random()

        for i in self.predictionsList:
            if i.ID == gameID1:
                if i.pred > randResult:
                    result = 0
                    return(result,i.pred)
                else:
                    result = 1
                    return(result,i.pred)
    #     print(game.parent)

        for i in self.predictionsList:
            if i.ID == gameID2:
                if i.pred < randResult:
                    result = 0
                    return (result, i.pred)
                else:
                    result = 1
                    return(result,i.pred)

root = Node2('R6CH')
root.left = Node2('R5WX')
root.left.parent = root
root.right = Node2('R5YZ')
root.right.parent = root

root.left.left = Node2('R4W1')
root.left.left.parent = root.left
root.left.right = Node2('R4X1')
root.left.right.parent = root.left
root.right.left = Node2('R4Y1')
root.right.left.parent = root.right
root.right.right = Node2('R4Z1')
root.right.right.parent = root.right

root.left.left.left = Node2('R3W1')
root.left.left.left.parent = root.left.left
root.left.left.right = Node2('R3W2')
root.left.left.right.parent = root.left.left
root.left.right.left = Node2('R3X1')
root.left.right.left.parent = root.left.right
root.left.right.right = Node2('R3X2')
root.left.right.right.parent = root.left.right
root.right.left.left = Node2('R3Y1')
root.right.left.left.parent = root.right.left
root.right.left.right = Node2('R3Y2')
root.right.left.right.parent = root.right.left
root.right.right.left = Node2('R3Z1')
root.right.right.left.parent = root.right.right
root.right.right.right = Node2('R3Z2')
root.right.right.right.parent = root.right.right

root.left.left.left.left = Node2('R2W1')
root.left.left.left.left.parent = root.left.left.left
root.left.left.left.right = Node2('R2W2')
root.left.left.left.right.parent = root.left.left.left
root.left.left.right.left = Node2('R2W3')
root.left.left.right.left.parent = root.left.left.right
root.left.left.right.right = Node2('R2W4')
root.left.left.right.right.parent = root.left.left.right

root.left.right.left.left = Node2('R2X1')
root.left.right.left.left.parent = root.left.right.left
root.left.right.left.right = Node2('R2X2')
root.left.right.left.right.parent = root.left.right.left
root.left.right.right.left = Node2('R2X3')
root.left.right.right.left.parent = root.left.right.right
root.left.right.right.right = Node2('R2X4')
root.left.right.right.right.parent = root.left.right.right

root.right.left.left.left = Node2('R2Y1')
root.right.left.left.left.parent = root.right.left.left
root.right.left.left.right = Node2('R2Y2')
root.right.left.left.right.parent = root.right.left.left
root.right.left.right.left = Node2('R2Y3')
root.right.left.right.left.parent = root.right.left.right
root.right.left.right.right = Node2('R2Y4')
root.right.left.right.right.parent = root.right.left.right

root.right.right.left.left = Node2('R2Z1')
root.right.right.left.left.parent = root.right.right.left
root.right.right.left.right = Node2('R2Z2')
root.right.right.left.right.parent = root.right.right.left
root.right.right.right.left = Node2('R2Z3')
root.right.right.right.left.parent = root.right.right.right
root.right.right.right.right = Node2('R2Z4')
root.right.right.right.right.parent = root.right.right.right

root.left.left.left.left.left = Node2('R1W1')
root.left.left.left.left.left.parent = root.left.left.left.left
root.left.left.left.left.right = Node2('R1W2')
root.left.left.left.left.right.parent = root.left.left.left.left
root.left.left.left.right.left = Node2('R1W3')
root.left.left.left.right.left.parent = root.left.left.left.right
root.left.left.left.right.right = Node2('R1W4')
root.left.left.left.right.right.parent = root.left.left.left.right
root.left.left.right.left.left = Node2('R1W5')
root.left.left.right.left.left.parent = root.left.left.right.left
root.left.left.right.left.right = Node2('R1W6')
root.left.left.right.left.right.parent = root.left.left.right.left
root.left.left.right.right.left = Node2('R1W7')
root.left.left.right.right.left.parent = root.left.left.right.right
root.left.left.right.right.right = Node2('R1W8')
root.left.left.right.right.right.parent = root.left.left.right.right

root.left.right.left.left.left = Node2('R1X1')
root.left.right.left.left.left.parent = root.left.right.left.left
root.left.right.left.left.right = Node2('R1X2')
root.left.right.left.left.right.parent = root.left.right.left.left
root.left.right.left.right.left = Node2('R1X3')
root.left.right.left.right.left.parent = root.left.right.left.right
root.left.right.left.right.right= Node2('R1X4')
root.left.right.left.right.right.parent = root.left.right.left.right
root.left.right.right.left.left = Node2('R1X5')
root.left.right.right.left.left.parent = root.left.right.right.left
root.left.right.right.left.right= Node2('R1X6')
root.left.right.right.left.right.parent = root.left.right.right.left
root.left.right.right.right.left = Node2('R1X7')
root.left.right.right.right.left.parent = root.left.right.right.right
root.left.right.right.right.right = Node2('R1X8')
root.left.right.right.right.right.parent = root.left.right.right.right

root.right.left.left.left.left = Node2('R1Y1')
root.right.left.left.left.left.parent = root.right.left.left.left
root.right.left.left.left.right = Node2('R1Y2')
root.right.left.left.left.right.parent = root.right.left.left.left
root.right.left.left.right.left = Node2('R1Y3')
root.right.left.left.right.left.parent = root.right.left.left.right
root.right.left.left.right.right = Node2('R1Y4')
root.right.left.left.right.right.parent = root.right.left.left.right
root.right.left.right.left.left = Node2('R1Y5')
root.right.left.right.left.left.parent = root.right.left.right.left
root.right.left.right.left.right = Node2('R1Y6')
root.right.left.right.left.right.parent = root.right.left.right.left
root.right.left.right.right.left = Node2('R1Y7')
root.right.left.right.right.left.parent = root.right.left.right.right
root.right.left.right.right.right = Node2('R1Y8')
root.right.left.right.right.right.parent = root.right.left.right.right

root.right.right.left.left.left = Node2('R1Z1')
root.right.right.left.left.left.parent = root.right.right.left.left
root.right.right.left.left.right = Node2('R1Z2')
root.right.right.left.left.right.parent = root.right.right.left.left
root.right.right.left.right.left = Node2('R1X3')
root.right.right.left.right.left.parent = root.right.right.left.right
root.right.right.left.right.right = Node2('R1Z4')
root.right.right.left.right.right.parent = root.right.right.left.right
root.right.right.right.left.left = Node2('R1Z5')
root.right.right.right.left.left.parent = root.right.right.right.left
root.right.right.right.left.right = Node2('R1Z6')
root.right.right.right.left.right.parent = root.right.right.right.left
root.right.right.right.right.left = Node2('R1Z7')
root.right.right.right.right.left.parent = root.right.right.right.right
root.right.right.right.right.right = Node2('R1Z8')
root.right.right.right.right.right.parent = root.right.right.right.right

tourn = Tournament(root)   
tourn.reverseLevelOrder()

tourn.getNode('R1W5').right = Node2('W12')
tourn.getNode('R1W5').right.parent = tourn.getNode('R1W5')
tourn.getNode('R1X6').right = Node2('X11')
tourn.getNode('R1X6').right.parent = tourn.getNode('R1X6')
tourn.getNode('R1Y1').right = Node2('Y16')
tourn.getNode('R1Y1').right.parent = tourn.getNode('R1Y1')
tourn.getNode('R1Z1').right = Node2('Z16')
tourn.getNode('R1Z1').right.parent = tourn.getNode('R1Z1')

tourn.reverseLevelOrder()

tourn.getNode('W12').team1Seed = 'W12a'
tourn.getNode('W12').team1Seed = 'W12b'

tourn.getNode('X11').team1Seed = 'X11a'
tourn.getNode('X11').team1Seed = 'X11b'

tourn.getNode('Y16').team1Seed = 'Y16a'
tourn.getNode('Y16').team1Seed = 'Y16b'

tourn.getNode('Z16').team1Seed = 'Z16a'
tourn.getNode('Z16').team1Seed = 'Z16b'

tourn.reverseLevelOrder()

tourn.populateSlots(cwd+"\\data_stage2\\MNCAATourneySlots.csv",2022)
tourn.populateTeamIDs(cwd+"\\data_stage2\\MNCAATourneySeeds.csv",2022)
tourn.populateTeamNames(cwd+"\\data_stage2\\MTeams.csv")
tourn.populatePredictionsList(cwd+"\\data_stage2\\MSampleSubmissionStage2.csv")

for game in tourn.nodeList:
    print("game:", game.value, game.team1Seed, game.team2Seed)
#     print("parent",game.parent)
#     print(game.value, game.team1ID, game.team2ID)
    result = tourn.getMatchPrediction(game.team1ID,game.team2ID)
    print("result: ",result[0])
    if(game.parent==None):
        print("no parents")
        if (result[0] == 1):
            print(1)
            game.winnerID = game.team1ID   
        elif(result[0]== 0):
            print(2)
            game.winnerID = game.team2ID        
    elif (game==game.parent.left):
        print('left')
        if (result[0] == 1):
            print(1)
            game.winnerID = game.team1ID
            game.parent.team1Seed = game.team1Seed    
            game.parent.team1ID = game.team1ID  
        elif(result[0]== 0):
            print(2)
            game.winnerID = game.team2ID        
            game.parent.team1Seed = game.team2Seed
            game.parent.team1ID = game.team2ID
    elif (game==game.parent.right):
        print('right')
        if (result[0] == 1):
            print(3)
            game.winnerID = game.team2ID
            game.parent.team2Seed = game.team1Seed
            game.parent.team2ID = game.team1ID
        elif(result[0]== 0):
            print(4)
            game.winnerID = game.team1ID
            game.parent.team2Seed = game.team2Seed
            game.parent.team2ID = game.team2ID
    else:
        print("no parents :(")

fig = go.Figure()
fig.update_layout(width=1600, height=2600)

tourn.reverseLevelOrder()
gamesList = tourn.nodeList.copy()

## pre round


width_dist = 10
depth_dist = 10
levels = 7


def bintree_level(node, levels, x, y, width):
    segments = []
    xl = x - depth_dist
    yl = y - width / 2
    xr = x - depth_dist
    yr = y + width / 2
#     print('x')
#     yr = y 
#     segments.append([[x, y], [xl, y]])
#     segments.append([[x, y], [xr, y]])
    
    fig.add_trace(go.Scatter(
    x=[x, xl],
    y=[yl, yl],
    mode="lines+text",
    line_color="black",
    name="Lines and Text",
    text=[node.team1Name],
#     text
    textposition="top left",
    textfont=dict(
        family="sans serif",
        size=18,
        color="black"
    )    
    )
    )
    
#     print("a")
    fig.add_trace(go.Scatter(
    x=[x, xr],
    y=[yr, yr],
    mode="lines+text",
    line_color="black",
    name="Lines and Text",
    textposition="top left",
    text=[node.team2Name],
#     text=['team2'],
    textfont=dict(
        family="sans serif",
        size=18,
        color="black"
    )
    )
    )
    
    fig.add_trace(go.Scatter(
    x=[x,x],
    y=[yl, yr],
    mode="lines",
    line_color="black",
    ))
    
    if levels > 2:
#         j = j+1
        bintree_level(node.left, levels - 1, xl, yl, width / 2)
        bintree_level(node.right, levels - 1, xr, yr, width  / 2)
        
    if levels == 1:
        print("yes")

        #print final
fig.add_trace(go.Scatter(
    x=[0,0+width_dist],
    y=[0, 0],
    mode="lines+text",
    line_color="black",
    name="Lines and Text",
    text=[tourn.root.winnerID],
#     text
    textposition="top right",
    textfont=dict(
        family="sans serif",
        size=18,
        color="black"
    )    
    )
    )       

j = 0
node = tourn.root
bintree_level(node,levels, 0, 0, width_dist)
fig.update_layout(paper_bgcolor="grey", plot_bgcolor="white")
fig.show()