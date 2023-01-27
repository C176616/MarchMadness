from dash import Dash, dcc, html, Input, Output, dash_table
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn import linear_model
import numpy as np
import plotly.graph_objects as go

# from src.team import Team
from src.tournament import Tournament
from src.matchPrediction import MatchPrediction
import os


app = Dash(__name__)
cwd = os.getcwd()

df_training_set = pd.read_csv('training_set.csv')
df_training_set_stage2 = pd.read_csv("training_set_stage2.csv")
df_corr = df_training_set.corr()
cols = ['deltaSeed', 'deltaWinPct', 'deltaPointsFor', 'deltaFGM', 'deltaAst', 'deltaBlk']
# X = df[cols]
y = df_training_set['Result']
testModelType = 'nope'

df_stage1Combinations = pd.read_csv(cwd+"\\data_stage2\\MSampleSubmissionStage2.csv")

predictionsList = []


## from jupyter
# 

##      
for index, row in df_stage1Combinations.iterrows():
    predictionsList.append(MatchPrediction(row['ID'], row['Pred']))

def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

# fig = px.bar(df, x="deltaFGM", y="deltaBlk", barmode="group")
# fig = sns.heatmap(correlation, vmax=.8, square=True)
fig1 = go.Figure()
fig1.add_trace(
    go.Heatmap(
        x = ['Result'],
        y = df_corr.index,
        z = np.array(df_corr),
        text=df_corr.values,
        texttemplate='%{text:.2f}',
    )
)
# fig1.update_layout(titlewidth=400, height=1000, yaxis_nticks=len(df_corr))

############### from Forge
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
import numpy as np
import plotly.io as pio
from itertools import cycle

from src.game import Game
from src.team import Team
from src.tournament import Tournament
# from src import tournament

pio.renderers.deafult = "notebook"

cwd = os.getcwd()

season = 2022

slotcsv = cwd+"\\data_stage2\\MNCAATourneySlots.csv"
seedcsv = cwd+"\\data_stage2\\MNCAATourneySeeds.csv"
namecsv = cwd+"\\data_stage2\\MTeams.csv"
predictionscsv = cwd+"\\data_stage2\\MSampleSubmissionStage2.csv"

df_slots = pd.read_csv(slotcsv)
df_slots = df_slots[df_slots["Season"]==season]

df_seeds = pd.read_csv(seedcsv)
df_seeds = df_seeds[df_seeds["Season"]==season]

df_names = pd.read_csv(namecsv)

df_stage1Combinations = pd.read_csv(predictionscsv)
# tourn.populatePredictionsList()
df_comb = df_seeds.merge(df_names[['TeamID','TeamName']], left_on = 'TeamID', right_on='TeamID')[['Seed','TeamID','TeamName']]
df_comb2 = df_slots.merge(df_comb, left_on="StrongSeed", right_on="Seed")[['Slot','StrongSeed','WeakSeed','TeamID','TeamName',]]
df_comb2 = df_comb2.rename(columns={'TeamID':'Team1ID', 'TeamName':'Team1Name'})
df_comb3 = df_comb2.merge(df_comb, how='left',left_on="WeakSeed", right_on="Seed")[['Slot','StrongSeed','WeakSeed','Team1ID','Team1Name','TeamID','TeamName',]]
df_info = df_comb3.rename(columns={'TeamID':'Team2ID', 'TeamName':'Team2Name'})

print(df_info)


## Initialize the tournament structure

root = Game('R6CH')
root.left = Game('R5WX')
root.left.parent = root
root.right = Game('R5YZ')
root.right.parent = root

root.left.left = Game('R4W1')
root.left.left.parent = root.left
root.left.right = Game('R4X1')
root.left.right.parent = root.left
root.right.left = Game('R4Y1')
root.right.left.parent = root.right
root.right.right = Game('R4Z1')
root.right.right.parent = root.right

root.left.left.left = Game('R3W1')
root.left.left.left.parent = root.left.left
root.left.left.right = Game('R3W2')
root.left.left.right.parent = root.left.left
root.left.right.left = Game('R3X1')
root.left.right.left.parent = root.left.right
root.left.right.right = Game('R3X2')
root.left.right.right.parent = root.left.right
root.right.left.left = Game('R3Y1')
root.right.left.left.parent = root.right.left
root.right.left.right = Game('R3Y2')
root.right.left.right.parent = root.right.left
root.right.right.left = Game('R3Z1')
root.right.right.left.parent = root.right.right
root.right.right.right = Game('R3Z2')
root.right.right.right.parent = root.right.right

root.left.left.left.left = Game('R2W1')
root.left.left.left.left.parent = root.left.left.left
root.left.left.left.right = Game('R2W2')
root.left.left.left.right.parent = root.left.left.left
root.left.left.right.left = Game('R2W3')
root.left.left.right.left.parent = root.left.left.right
root.left.left.right.right = Game('R2W4')
root.left.left.right.right.parent = root.left.left.right

root.left.right.left.left = Game('R2X1')
root.left.right.left.left.parent = root.left.right.left
root.left.right.left.right = Game('R2X2')
root.left.right.left.right.parent = root.left.right.left
root.left.right.right.left = Game('R2X3')
root.left.right.right.left.parent = root.left.right.right
root.left.right.right.right = Game('R2X4')
root.left.right.right.right.parent = root.left.right.right

root.right.left.left.left = Game('R2Y1')
root.right.left.left.left.parent = root.right.left.left
root.right.left.left.right = Game('R2Y2')
root.right.left.left.right.parent = root.right.left.left
root.right.left.right.left = Game('R2Y3')
root.right.left.right.left.parent = root.right.left.right
root.right.left.right.right = Game('R2Y4')
root.right.left.right.right.parent = root.right.left.right

root.right.right.left.left = Game('R2Z1')
root.right.right.left.left.parent = root.right.right.left
root.right.right.left.right = Game('R2Z2')
root.right.right.left.right.parent = root.right.right.left
root.right.right.right.left = Game('R2Z3')
root.right.right.right.left.parent = root.right.right.right
root.right.right.right.right = Game('R2Z4')
root.right.right.right.right.parent = root.right.right.right

root.left.left.left.left.left = Game('R1W1')
root.left.left.left.left.left.parent = root.left.left.left.left
root.left.left.left.left.right = Game('R1W2')
root.left.left.left.left.right.parent = root.left.left.left.left
root.left.left.left.right.left = Game('R1W3')
root.left.left.left.right.left.parent = root.left.left.left.right
root.left.left.left.right.right = Game('R1W4')
root.left.left.left.right.right.parent = root.left.left.left.right
root.left.left.right.left.left = Game('R1W5')
root.left.left.right.left.left.parent = root.left.left.right.left
root.left.left.right.left.right = Game('R1W6')
root.left.left.right.left.right.parent = root.left.left.right.left
root.left.left.right.right.left = Game('R1W7')
root.left.left.right.right.left.parent = root.left.left.right.right
root.left.left.right.right.right = Game('R1W8')
root.left.left.right.right.right.parent = root.left.left.right.right

root.left.right.left.left.left = Game('R1X1')
root.left.right.left.left.left.parent = root.left.right.left.left
root.left.right.left.left.right = Game('R1X2')
root.left.right.left.left.right.parent = root.left.right.left.left
root.left.right.left.right.left = Game('R1X3')
root.left.right.left.right.left.parent = root.left.right.left.right
root.left.right.left.right.right= Game('R1X4')
root.left.right.left.right.right.parent = root.left.right.left.right
root.left.right.right.left.left = Game('R1X5')
root.left.right.right.left.left.parent = root.left.right.right.left
root.left.right.right.left.right= Game('R1X6')
root.left.right.right.left.right.parent = root.left.right.right.left
root.left.right.right.right.left = Game('R1X7')
root.left.right.right.right.left.parent = root.left.right.right.right
root.left.right.right.right.right = Game('R1X8')
root.left.right.right.right.right.parent = root.left.right.right.right

root.right.left.left.left.left = Game('R1Y1')
root.right.left.left.left.left.parent = root.right.left.left.left
root.right.left.left.left.right = Game('R1Y2')
root.right.left.left.left.right.parent = root.right.left.left.left
root.right.left.left.right.left = Game('R1Y3')
root.right.left.left.right.left.parent = root.right.left.left.right
root.right.left.left.right.right = Game('R1Y4')
root.right.left.left.right.right.parent = root.right.left.left.right
root.right.left.right.left.left = Game('R1Y5')
root.right.left.right.left.left.parent = root.right.left.right.left
root.right.left.right.left.right = Game('R1Y6')
root.right.left.right.left.right.parent = root.right.left.right.left
root.right.left.right.right.left = Game('R1Y7')
root.right.left.right.right.left.parent = root.right.left.right.right
root.right.left.right.right.right = Game('R1Y8')
root.right.left.right.right.right.parent = root.right.left.right.right

root.right.right.left.left.left = Game('R1Z1')
root.right.right.left.left.left.parent = root.right.right.left.left
root.right.right.left.left.right = Game('R1Z2')
root.right.right.left.left.right.parent = root.right.right.left.left
root.right.right.left.right.left = Game('R1X3')
root.right.right.left.right.left.parent = root.right.right.left.right
root.right.right.left.right.right = Game('R1Z4')
root.right.right.left.right.right.parent = root.right.right.left.right
root.right.right.right.left.left = Game('R1Z5')
root.right.right.right.left.left.parent = root.right.right.right.left
root.right.right.right.left.right = Game('R1Z6')
root.right.right.right.left.right.parent = root.right.right.right.left
root.right.right.right.right.left = Game('R1Z7')
root.right.right.right.right.left.parent = root.right.right.right.right
root.right.right.right.right.right = Game('R1Z8')
root.right.right.right.right.right.parent = root.right.right.right.right

tourn = Tournament(root)   
tourn.reverseLevelOrder()

tourn.getNode('R1W5').right = Game('W12')
tourn.getNode('R1W5').right.parent = tourn.getNode('R1W5')
tourn.getNode('R1X6').right = Game('X11')
tourn.getNode('R1X6').right.parent = tourn.getNode('R1X6')
tourn.getNode('R1Y1').right = Game('Y16')
tourn.getNode('R1Y1').right.parent = tourn.getNode('R1Y1')
tourn.getNode('R1Z1').right = Game('Z16')
tourn.getNode('R1Z1').right.parent = tourn.getNode('R1Z1')

tourn.reverseLevelOrder()

preRound1Slots = ['W12','X11','Y16','Z16']
cycleList = cycle(preRound1Slots)

for x in range(3):
    i = next(cycleList)
    slot = df_info[df_info['Slot']==i]
    tourn.getNode(i).team1 = Team(slot['StrongSeed'].values[0],slot['Team1ID'].values[0],slot['Team1Name'].values[0])
    tourn.getNode(i).team2 = Team(slot['WeakSeed'].values[0],slot['Team2ID'].values[0],slot['Team2Name'].values[0])

tourn.populateTeams(df_info)

tourn.populatePredictionsList(df_stage1Combinations)

for game in tourn.nodeList:
    print("game:", game.value, game.team1, game.team2)
#     print("parent",game.parent)
#     print(game.value, game.team1ID, game.team2ID)
    result = tourn.getMatchPrediction(int(game.team1.teamID),int(game.team2.teamID))
    print("result: ",result[0], "chance:", result[1])
    game.winPct = result[1]
    if(game.parent==None):
        print("no parents")
        if (result[0] == 1):
            print(1)
            game.winner = game.team1  
        elif(result[0]== 0):
            print(2)
            game.winner = game.team2        
    elif (game==game.parent.left):
        print('left')
        if (result[0] == 1):
            print(1)
            game.winner = game.team1
            game.parent.team1 = game.team1   

        elif(result[0]== 0):
            print(2)
            game.winner = game.team2       
            game.parent.team1 = game.team2

    elif (game==game.parent.right):
        print('right')
        if (result[0] == 1):
            print(3)
            game.winner = game.team2
            game.parent.team2 = game.team1

        elif(result[0]== 0):
            print(4)
            game.winner = game.team1
            game.parent.team2 = game.team2

    else:
        print("no parents :(")


tourn.reverseLevelOrder()

fig = go.Figure()
fig.update_layout(width=1600, height=1000)

tourn.reverseLevelOrder()
gamesList = tourn.nodeList.copy()

## pre round


width_dist = 10
depth_dist = 10
levels = 6


def bintree_level(node, levels, x, y, width, side):
    segments = []
    if side=='left':
        xl = x - depth_dist
        xr = x - depth_dist
        textPosition = "top left"
    elif side=='right':
        xl = x + depth_dist
        xr = x + depth_dist
        textPosition = "top right"

    yr = y + width / 2
    yl = y - width / 2
    
    # print team1 
    fig.add_trace(go.Scatter(
    x=[x, xl],
    y=[yl, yl],
    mode="lines+text",
    line_color="black",
    name="Lines and Text",
    text=[node.value + " " + str(node.winPct) + " " + node.team1.getString()],
#     text
    textposition=textPosition,
    textfont=dict(
        family="sans serif",
        size=10,
        color="black"
    )    
    )
    )
    
    # print team2
    fig.add_trace(go.Scatter(
    x=[x, xr],
    y=[yr, yr],
    mode="lines+text",
    line_color="black",
    name="Lines and Text",
    textposition=textPosition,
    text=[node.value + " " + str(node.winPct) + " " + node.team2.getString()],
#     text=['team2'],
    textfont=dict(
        family="sans serif",
        size=10,
        color="black"
    )
    )
    )
    
    # print line connecting team1 and team 2
    fig.add_trace(go.Scatter(
    x=[x,x],
    y=[yl, yr],
    mode="lines",
    line_color="black",
    ))
    
    #recursively call this function
    if levels > 2:
        print(levels)
        bintree_level(node.left, levels - 1, xl, yl, width / 2, side)
        bintree_level(node.right, levels - 1, xr, yr, width  / 2, side)
        
    # recursion base condition
    if levels == 1:
        print("yes")


        #print final
    

j = 0
node1 = tourn.root.left
node2 = tourn.root.right
bintree_level(node1,levels,-10,0,width_dist,'left')
bintree_level(node2,levels,10,0,width_dist,'right')

print('ready for final')
#final right
fig.add_trace(go.Scatter(
x=[0, 10],
y=[-4, -4],
mode="lines+text",
line_color="black",
name="Lines and Text",
text=[tourn.root.left.value, tourn.root.team1.getString()],
#     text
textposition="top right",
textfont=dict(
    family="sans serif",
    size=18,
    color="black"
)    
)
)  

#final left
fig.add_trace(go.Scatter(
x=[-10, 0],
y=[4, 4],
mode="lines+text",
line_color="black",
name="Lines and Text",
text=[tourn.root.right.value, tourn.root.team2.getString()],
#     text
textposition="top left",
textfont=dict(
    family="sans serif",
    size=18,
    color="black"
)    
)
) 

fig.add_trace(go.Scatter(
x=[-8, 8],
y=[0, 0],
mode="lines+text",
line_color="black",
name="Lines and Text",
text=[tourn.root.winner.getString()],
#     text
textposition="top right",
textfont=dict(
    family="sans serif",
    size=18,
    color="black"
)    
)
) 

fig.update_layout(paper_bgcolor="grey", plot_bgcolor="white")

#################
app.layout = html.Div([
    html.H1("March Madness Machine Learning"),
    # Heatmap Graph
    html.Div([
        html.H2("Explore"),
            html.Div([       
                dcc.Graph(
                    id='heatmap',
                ),
            ], style={'width': '50%', 'vertical-align': 'middle', 'display': 'inline-block'}),

            html.Div([
                html.H3("Tournament Seasons to Include in Data:"),
                dcc.RangeSlider(
                    df_training_set['Season'].min(),
                    df_training_set['Season'].max(),
                    step=None,
                    value=[df_training_set['Season'].min(), df_training_set['Season'].max()],
                    marks={str(Season): str(Season) for Season in df_training_set['Season'].unique()},
                    id='i-season-range'            
                )
            ], style={'width': '100%', 'horizontal-align': 'middle'}),

    ], style={'width': '30%', 'display':'inline-block'}),
    


    html.Div([
        html.H2("Build"),
        html.Div([
            dcc.Dropdown(df_training_set.columns[4:], multi=True, id='i-model-features', placeholder="Select Features to Include"),
        ]),
        html.Br(),
        html.Div([
            dcc.RadioItems(['Linear', 'Logistic', 'Random Tree', 'Random Forest', 'Neural Net'], 'Random Forest', id='i-model-type'),
        ]),

    ],style={'width': '30%', 'display':'inline-block', 'vertical-align':'top'}),   

    #model creation 

    #Features
    

    html.Div([
        html.H2("Test"),
        html.H3("Train/Test Split (%)"),
        dcc.Slider(min=0,max=100,step=10, id='split-slider', value=70, marks=None, tooltip={"placement": "bottom", "always_visible": True}),
        html.Div(id='results-table'),
        
    ],style={'width': '33%', 'display':'inline-block', 'vertical-align':'top'}),
    

    html.Div([
        html.H2("Predict"),
        html.Div([
            dcc.Graph(
                id='predicted-bracket'
            )
        ])  
        ],style={'width': '100%', 'display':'inline-block', 'vertical-align':'top'})
    ])





@app.callback(
    Output('heatmap', 'figure'),
    Output('results-table', 'children'),
    Output('predicted-bracket','figure'),
    Input('i-season-range', 'value'),
    Input('i-model-type', 'value'),
    Input('i-model-features','value'))    
def update_output(value, modelType, modelFeatures):
    a = 0
    cols = modelFeatures
    df_modelResults = pd.DataFrame(columns=['Season','Error','Accuracy'])
    # Linear
    heatmap_figure = go.Figure()
    bracket_figure = go.Figure()
    df_corr = df_training_set[(df_training_set['Season']>=value[0]) & (df_training_set['Season']<=value[1])].corr()
    heatmap_figure.add_trace(
    go.Heatmap(
        x = ['Result'],
        y = df_corr.index,
        z = np.array(df_corr),
        text=df_corr.values,
        texttemplate='%{text:.2f}'
    ))
    heatmap_figure.update_layout(height=600,yaxis_nticks=len(df_corr))

## Logistic
    if modelType == 'Logistic':
        X = df_training_set[cols][(df_training_set['Season']>=value[0]) & (df_training_set['Season']<=value[1])]
        y = df_training_set['Result'][(df_training_set['Season']>=value[0]) & (df_training_set['Season']<=value[1])]

        model = linear_model.LogisticRegression(solver='lbfgs')

        model.fit(X,y)

        X_pred = df_training_set_stage2[cols]
        pred = model.predict_proba(X_pred)[:,1]
        df_stage1Combinations['Pred'] = np.round(pred,2)
        tourn.populatePredictionsList(df_stage1Combinations)

        for year in range(value[0],value[1]):
            if year != 2020:
                X_test = df_training_set[df_training_set['Season'] == year][cols]
                y_test = df_training_set[df_training_set['Season'] == year]['Result']

                df_results = X_test
                df_results['Prediction'] = model.predict_proba(X_test)[:,1]
                df_results['Result'] = y_test
                
                correct = df_results.loc[(df_results['Result']==0) & (df_results['Prediction']<0.5)].shape[0]
                correct = correct + df_results.loc[(df_results['Result']==1) & (df_results['Prediction']>0.5)].shape[0]

                total = df_results.shape[0]

                accuracy = correct/total

                df_results.loc[df_results['Prediction'] > 0.9, 'Prediction']=0.99
                df_results.loc[df_results['Prediction'] < 0.1, 'Prediction']=0.01

                error = -np.log(1-df_results.loc[df_results['Result'] == 0]['Prediction']).mean()
                data = {'Season':year,'Error':error, 'Accuracy': accuracy}
                print(data)
                
                df_modelResults = df_modelResults.append(data, ignore_index=True)

    if modelType == 'Linear':
        X = df_training_set[cols][(df_training_set['Season']>=value[0]) & (df_training_set['Season']<=value[1])]
        y = df_training_set['Result'][(df_training_set['Season']>=value[0]) & (df_training_set['Season']<=value[1])]
        linearModel = linear_model.LinearRegression()
        linearModel.fit(X,y)
        a = linearModel.score(X,y)
        #generate preditions list
        
        X_pred = df_training_set_stage2[cols]
        pred = linearModel.predict(X_pred)
        df_stage1Combinations['Pred'] = np.round(pred,2)
        tourn.populatePredictionsList(df_stage1Combinations)

        #test the model
        for year in range(value[0],value[1]):
            if year != 2020:
                X_test = df_training_set[df_training_set['Season'] == year][cols]
                y_test = df_training_set[df_training_set['Season'] == year]['Result']

                df_results = X_test
                df_results['Prediction'] = linearModel.predict(X_test)
                df_results['Result'] = y_test
                
                correct = df_results.loc[(df_results['Result']==0) & (df_results['Prediction']<0.5)].shape[0]
                correct = correct + df_results.loc[(df_results['Result']==1) & (df_results['Prediction']>0.5)].shape[0]

                total = df_results.shape[0]

                accuracy = correct/total

                error = -np.log(1-df_results.loc[df_results['Result'] == 0]['Prediction']).mean()
                data = {'Season':year,'Error':error, 'Accuracy': accuracy}
                print(data)
                
                df_modelResults = df_modelResults.append(data, ignore_index=True)

        tourn.populatePredictionsList(df_stage1Combinations)

        for game in tourn.nodeList:
            print("game:", game.value, game.team1, game.team2)
        #     print("parent",game.parent)
        #     print(game.value, game.team1ID, game.team2ID)
            result = tourn.getMatchPrediction(int(game.team1.teamID),int(game.team2.teamID))
            print("result: ",result[0], "chance:", result[1])
            game.winPct = result[1]
            if(game.parent==None):
                print("no parents")
                if (result[0] == 1):
                    print(1)
                    game.winner = game.team1  
                elif(result[0]== 0):
                    print(2)
                    game.winner = game.team2        
            elif (game==game.parent.left):
                print('left')
                if (result[0] == 1):
                    print(1)
                    game.winner = game.team1
                    game.parent.team1 = game.team1   

                elif(result[0]== 0):
                    print(2)
                    game.winner = game.team2       
                    game.parent.team1 = game.team2

            elif (game==game.parent.right):
                print('right')
                if (result[0] == 1):
                    print(3)
                    game.winner = game.team2
                    game.parent.team2 = game.team1

                elif(result[0]== 0):
                    print(4)
                    game.winner = game.team1
                    game.parent.team2 = game.team2

            else:
                print("no parents :(")

        tourn.reverseLevelOrder()

        
        bracket_figure.update_layout(width=1600, height=1000)

        tourn.reverseLevelOrder()
        gamesList = tourn.nodeList.copy()

## pre round


        width_dist = 10
        depth_dist = 10
        levels = 6


        def bintree_level(node, levels, x, y, width, side):
            segments = []
            if side=='left':
                xl = x - depth_dist
                xr = x - depth_dist
                textPosition = "top left"
            elif side=='right':
                xl = x + depth_dist
                xr = x + depth_dist
                textPosition = "top right"

            yr = y + width / 2
            yl = y - width / 2
            
            # print team1 
            bracket_figure.add_trace(go.Scatter(
            x=[x, xl],
            y=[yl, yl],
            mode="lines+text",
            line_color="black",
            name="Lines and Text",
            text=[node.value + " " + str(node.winPct) + " " + node.team1.getString()],
        #     text
            textposition=textPosition,
            textfont=dict(
                family="sans serif",
                size=10,
                color="black"
            )    
            )
            )
            
            # print team2
            bracket_figure.add_trace(go.Scatter(
            x=[x, xr],
            y=[yr, yr],
            mode="lines+text",
            line_color="black",
            name="Lines and Text",
            textposition=textPosition,
            text=[node.value + " " + str(node.winPct) + " " + node.team2.getString()],
        #     text=['team2'],
            textfont=dict(
                family="sans serif",
                size=10,
                color="black"
            )
            )
            )
            
            # print line connecting team1 and team 2
            bracket_figure.add_trace(go.Scatter(
            x=[x,x],
            y=[yl, yr],
            mode="lines",
            line_color="black",
            ))
            
            #recursively call this function
            if levels > 2:
                print(levels)
                bintree_level(node.left, levels - 1, xl, yl, width / 2, side)
                bintree_level(node.right, levels - 1, xr, yr, width  / 2, side)
                
            # recursion base condition
            if levels == 1:
                print("yes")


                #print final
            

        j = 0
        node1 = tourn.root.left
        node2 = tourn.root.right
        bintree_level(node1,levels,-10,0,width_dist,'left')
        bintree_level(node2,levels,10,0,width_dist,'right')

        print('ready for final')
        #final right
        bracket_figure.add_trace(go.Scatter(
        x=[0, 10],
        y=[-4, -4],
        mode="lines+text",
        line_color="black",
        name="Lines and Text",
        text=[tourn.root.left.value, tourn.root.team1.getString()],
        #     text
        textposition="top right",
        textfont=dict(
            family="sans serif",
            size=18,
            color="black"
        )    
        )
        )  

        #final left
        bracket_figure.add_trace(go.Scatter(
        x=[-10, 0],
        y=[4, 4],
        mode="lines+text",
        line_color="black",
        name="Lines and Text",
        text=[tourn.root.right.value, tourn.root.team2.getString()],
        #     text
        textposition="top left",
        textfont=dict(
            family="sans serif",
            size=18,
            color="black"
        )    
        )
        ) 

        bracket_figure.add_trace(go.Scatter(
        x=[-8, 8],
        y=[0, 0],
        mode="lines+text",
        line_color="black",
        name="Lines and Text",
        text=[tourn.root.winner.getString()],
        #     text
        textposition="top right",
        textfont=dict(
            family="sans serif",
            size=18,
            color="black"
        )    
        )
        ) 
# paper_bgcolor="grey"
        bracket_figure.update_layout(plot_bgcolor="white", showlegend=False)
        bracket_figure.update_xaxes(showticklabels=False)
        bracket_figure.update_yaxes(showticklabels=False)            
    # Logistic
    # Random Tree
    # Random Forest
    # Neural Net

    else:
        pass
    
    print(df_modelResults)
    resultsTable = generate_table(df_modelResults)
    returnTable = dash_table.DataTable(
        data=df_modelResults.to_dict('records'), 
        columns=[{"name": i, "id": i} for i in df_modelResults.columns]
    )
    return heatmap_figure, returnTable, bracket_figure

if __name__ == '__main__':
    app.run_server(debug=True)
