import base64
import datetime
import io

import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html, dash_table

import pandas as pd
import json
import jsonpickle

import plotly.express as px

import plotly.graph_objects as go

import os

## app 3 is for generating the master bracket for use with app 2
cwd = os.getcwd()

app = dash.Dash(__name__)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

text_file = open('brackettext.txt', 'r')
stringData = text_file.read()
print(stringData)
text_file.close()

## Import required data
df_training_set = pd.read_csv('training_set.csv')
df_training_set_stage2 = pd.read_csv("training_set_stage2.csv")
df_stage1Combinations = pd.read_csv(cwd +
                                    "/data_stage2/MSampleSubmissionStage2.csv")

# Open the pre-populated tournament layout JSON file
with open('tournamentLayout.json', 'r') as f:
    jsonData = json.load(f)

tourn = jsonpickle.decode(jsonData)

print(jsonData)
# with open('brackettext.txt', 'r') as f:
#     jsonData = json.loads(f)

bracket_figure = go.Figure()
bracket_figure = px.scatter()
bracket_figure.update_layout(plot_bgcolor="#3c3c3c",
                             showlegend=False,
                             template='plotly_dark',
                             clickmode='event+select')
bracket_figure.update_xaxes(showticklabels=False,
                            showgrid=False,
                            zeroline=False)
bracket_figure.update_yaxes(showticklabels=False,
                            showgrid=False,
                            zeroline=False)

width_dist = 10
depth_dist = 10
levels = 6

# print team1

app.layout = html.Div([
    # dcc.Graph(id='o-predicted-bracket', figure=bracket_figure),
    dcc.Graph(id='o-predicted-bracket'),
    html.Div([
        html.H2(
            "asdf",
            id='click-data',
        ),
    ], ),
    html.Div([
        html.Button("Download Bracket", id="btn-download"),
        dcc.Download(id="download-text")
    ]),
])


@app.callback(Output('download-text', 'data'),
              Input("btn-download", "n_clicks"),
              prevent_initial_call=True)
def download_function(n_clicks):
    jsonData = jsonpickle.encode(tourn)
    return dict(content=jsonData, filename="masterBracket.txt")


@app.callback(Output('o-predicted-bracket', 'figure'),
              Input('o-predicted-bracket', 'clickData'))
def clickInput(clickData):
    width_dist = 10
    depth_dist = 10
    levels = 6

    coordinateDict = {}
    bracket_figure = go.Figure()
    bracket_figure = px.scatter()
    bracket_figure.update_layout(plot_bgcolor="#3c3c3c",
                                 showlegend=False,
                                 template='plotly_dark',
                                 clickmode='event+select')
    bracket_figure.update_xaxes(showticklabels=False,
                                showgrid=False,
                                zeroline=False)
    bracket_figure.update_yaxes(showticklabels=False,
                                showgrid=False,
                                zeroline=False)

    def bintree_level(node, levels, x, y, width, side):
        segments = []
        if side == 'left':
            xl = x - depth_dist
            xr = x - depth_dist
            textPosition = "top left"
        elif side == 'right':
            xl = x + depth_dist
            xr = x + depth_dist
            textPosition = "top right"

        yr = y + width / 2
        yl = y - width / 2

        # newDict = {xl: {yl: node.team1.teamName if node.team1 else " "}}
        thisKey = str(xl) + "_" + str(yl)
        thisValue = [node.value, node.team1]
        # thisValue = str(
        #     node.value) + "_" + str(node.team1.teamName if node.team1 else " ")
        newDict = {thisKey: thisValue}
        coordinateDict.update(newDict)
        thisKey = str(xr) + "_" + str(yr)
        thisValue = [node.value, node.team2]
        # thisValue = str(
        #     node.value) + "_" + str(node.team1.teamName if node.team1 else " ")
        newDict = {thisKey: thisValue}
        coordinateDict.update(newDict)

        bracket_figure.add_trace(
            go.Scatter(
                x=[x, xl],
                y=[yl, yl],
                mode="lines+text",
                line_color="white",
                name="testname",
                text=[node.team1.teamName if node.team1 else " "],
                #     text
                textposition=textPosition,
                customdata=["testing"],
                textfont=dict(family="sans serif", size=10, color="white")))

        # print team2
        bracket_figure.add_trace(
            go.Scatter(
                x=[x, xr],
                y=[yr, yr],
                mode="lines+text",
                line_color="white",
                name="Lines and Text",
                textposition=textPosition,
                text=[node.team2.teamName if node.team2 else " "],
                #     text=['team2'],
                textfont=dict(family="sans serif", size=10, color="white")))

        # print line connecting team1 and team 2
        bracket_figure.add_trace(
            go.Scatter(
                x=[x, x],
                y=[yl, yr],
                mode="lines",
                line_color="white",
            ))

        #recursively call this function
        if levels > 2:
            # print(levels)
            bintree_level(node.left, levels - 1, xl, yl, width / 2, side)
            bintree_level(node.right, levels - 1, xr, yr, width / 2, side)

        # recursion base condition
        if levels == 1:
            # print("yes")
            pass

    node1 = tourn.root.left
    node2 = tourn.root.right
    bintree_level(node1, levels, -10, 0, width_dist, 'left')
    bintree_level(node2, levels, 10, 0, width_dist, 'right')

    #print final right
    bracket_figure.add_trace(
        go.Scatter(
            x=[0, 10],
            y=[-4, -4],
            mode="lines+text",
            line_color="white",
            name="Lines and Text",
            text=[tourn.root.right.value],
            #     text
            textposition="top right",
            textfont=dict(family="sans serif", size=10, color="white")))

    #print final left
    bracket_figure.add_trace(
        go.Scatter(
            x=[-10, 0],
            y=[4, 4],
            mode="lines+text",
            line_color="white",
            name="Lines and Text",
            text=[tourn.root.left.value],
            #     text
            textposition="top right",
            textfont=dict(family="sans serif", size=10, color="white")))

    # print final winner
    bracket_figure.add_trace(
        go.Scatter(
            x=[-8, 8],
            y=[0, 0],
            mode="lines+text",
            line_color="white",
            name="Lines and Text",
            text=[tourn.root.value],
            #     text
            textposition="top right",
            textfont=dict(family="sans serif", size=10, color="white")))

    # paper_bgcolor="grey"

    styles = {'pre': {'border': 'thin lightgrey solid', 'overflowX': 'scroll'}}

    bracket_figure.update_layout(plot_bgcolor="#3c3c3c",
                                 showlegend=False,
                                 template='plotly_dark')
    bracket_figure.update_xaxes(showticklabels=False,
                                showgrid=False,
                                zeroline=False)
    bracket_figure.update_yaxes(showticklabels=False,
                                showgrid=False,
                                zeroline=False)
    bracket_figure.update_layout(width=1600, height=1000)

    # jsonData = json.load(clickData)
    if (clickData):
        # print(coordinateDict)
        target = str(clickData['points'][0]['x']) + "_" + str(
            clickData['points'][0]['y'])
        print(target)
        info = coordinateDict[target]
        nodeValue = info[0]
        team = info[1]
        node = tourn.getNode(nodeValue)
        node.winner = team
        # node.parent.team1 = team

        if node == node.parent.left:
            node.parent.team1 = team
        if node == node.parent.right:
            node.parent.team2 = team

        # print(teamName)
        # currentNode = tourn.getNodeByTeamName(teamName)
        # print(currentNode.parent)
        # if currentNode == currentNode.parent.left:
        #     print("left")
        #     if teamName == currentNode.team1.teamName:
        #         currentNode.parent.team1 = currentNode.team1
        #     elif teamName == currentNode.team2.teamName:
        #         currentNode.parent.team1 = currentNode.team2

        # elif currentNode == currentNode.parent.right:
        #     print("right")
        #     if teamName == currentNode.team1.teamName:
        #         currentNode.parent.team2 = currentNode.team1
        #     elif teamName == currentNode.team2.teamName:
        #         currentNode.parent.team2 = currentNode.team2

        # if teamName == currentNode.team1:
        #     currentNode.winner = currentNode.team1
        # if teamName == currentNode.team2:
        #     currentNode.winner = currentNode.team2

        # print(currentNode.parent.team1.teamName)
        node1 = tourn.root.left
        node2 = tourn.root.right
        bintree_level(node1, levels, -10, 0, width_dist, 'left')
        bintree_level(node2, levels, 10, 0, width_dist, 'right')

    return bracket_figure


if __name__ == '__main__':
    app.run_server(debug=True)
