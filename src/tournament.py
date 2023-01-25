import plotly.graph_objects as go
import plotly.express as px
from src.matchPrediction import MatchPrediction
import numpy as np

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
#             print(i)
            if i.value==node:
                return i
        

    def populateTeams(self, df_slots, df_seeds, df_names):
        for game in self.nodeList:
            game.team1Seed = df_slots[df_slots['Slot']==game.value]['StrongSeed'].values[0]
            game.team2Seed = df_slots[df_slots['Slot']==game.value]['WeakSeed'].values[0]
            if [*game.value][1] == '1':
                a = df_seeds[df_seeds['Seed']==game.team1Seed]
                b = df_seeds[df_seeds['Seed']==game.team2Seed]
                if a.empty != True:
                    game.team1ID = a['TeamID'].values[0]
                if b.empty != True:
                    game.team2ID = b['TeamID'].values[0]

                a = df_names[df_names['TeamID']==game.team1ID]
                b = df_names[df_names['TeamID']==game.team2ID]
                if a.empty != True:
                    game.team1Name = a['TeamName'].values[0]
                if b.empty != True:
                    game.team2Name = b['TeamName'].values[0]
                

    def populatePredictionsList(self, df_stage1Combinations):
        self.predictionsList = []
        for index, row in df_stage1Combinations.iterrows():
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


    # def plot(self, fig, node, levels, x, y, width, side): 

    #     fig.update_layout(width=1600, height=2600)    
    #     width = 10   
    #     width_dist = 10
    #     depth_dist = 10
    #     x = 0
    #     y = 0
    #     # levels = 6
    #     segments = []
    #     if side=='left':
    #         xl = x - depth_dist
    #         xr = x - depth_dist
    #     elif side=='right':
    #         xl = x + depth_dist
    #         xr = x + depth_dist

    #     yr = y + width / 2
    #     yl = y - width / 2

    #     fig.add_trace(go.Scatter(
    #     x=[x, xl],
    #     y=[yl, yl],
    #     mode="lines+text",
    #     line_color="black",
    #     name="Lines and Text",
    #     # text=[node.value],
    #     textposition="top left",
    #     textfont=dict(
    #         family="sans serif",
    #         size=18,
    #         color="black"
    #     )    
    #     )
    #     )
        
    #     fig.add_trace(go.Scatter(
    #     x=[x, xr],
    #     y=[yr, yr],
    #     mode="lines+text",
    #     line_color="black",
    #     name="Lines and Text",
    #     textposition="top left",
    #     # text=[node.value],
    #     textfont=dict(
    #         family="sans serif",
    #         size=18,
    #         color="black"
    #     )))
        
    #     fig.add_trace(go.Scatter(
    #     x=[x,x],
    #     y=[yl, yr],
    #     mode="lines",
    #     line_color="black",
    #     ))
        
    #     if levels > 2:
    #         self.plot(fig, node.left, levels - 1, xl, yl, width / 2, side)
    #         self.plot(fig, node.right, levels - 1, xr, yr, width  / 2, side)
            
    #     if levels == 1:
    #         print("yes")
    #         # return fig

        # #print final
        # fig.add_trace(go.Scatter(
        # x=[0,0+width_dist],
        # y=[0, 0],
        # mode="lines+text",
        # line_color="black",
        # name="Lines and Text",
        # # text=[self.root.winnerID],
        # textposition="top right",
        # textfont=dict(
        #     family="sans serif",
        #     size=18,
        #     color="black"
        # )    
        # )
        # )       
        # fig.update_layout(paper_bgcolor="grey", plot_bgcolor="white")
        
    
