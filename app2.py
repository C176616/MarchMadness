import base64
import datetime
import io

import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html, dash_table

import pandas as pd
import json
import jsonpickle

import plotly.graph_objects as go

app = dash.Dash(__name__)

text_file = open('brackettext.txt', 'r')
stringData = text_file.read()
print(stringData)
text_file.close()

jsonData = json.loads(stringData)

print(jsonData)
# with open('brackettext.txt', 'r') as f:
#     jsonData = json.loads(f)

tourn = jsonpickle.decode(stringData)

bracket_figure = go.Figure()
bracket_figure.update_layout(plot_bgcolor="#3c3c3c",
                             showlegend=False,
                             template='plotly_dark')
bracket_figure.update_xaxes(showticklabels=False,
                            showgrid=False,
                            zeroline=False)
bracket_figure.update_yaxes(showticklabels=False,
                            showgrid=False,
                            zeroline=False)

width_dist = 10
depth_dist = 10
levels = 6


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

    # print team1
    bracket_figure.add_trace(
        go.Scatter(
            x=[x, xl],
            y=[yl, yl],
            mode="lines+text",
            line_color="white",
            name="Lines and Text",
            text=[
                node.value + " " + str(node.winPct) + " " +
                node.team1.getString()
            ],
            #     text
            textposition=textPosition,
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
            text=[node.value + " " + node.team2.getString()],
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
        text=[
            tourn.root.left.value + " " + str(tourn.root.left.winPct) + " " +
            tourn.root.team2.getString()
        ],
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
        text=[
            tourn.root.left.value + " " + str(tourn.root.left.winPct) + " " +
            tourn.root.team1.getString()
        ],
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
        text=[
            tourn.root.left.value + " " + str(tourn.root.winPct) + " " +
            tourn.root.winner.getString()
        ],
        #     text
        textposition="top right",
        textfont=dict(family="sans serif", size=10, color="white")))
# paper_bgcolor="grey"
bracket_figure.update_layout(plot_bgcolor="#3c3c3c",
                             showlegend=False,
                             template='plotly_dark')
bracket_figure.update_xaxes(showticklabels=False,
                            showgrid=False,
                            zeroline=False)
bracket_figure.update_yaxes(showticklabels=False,
                            showgrid=False,
                            zeroline=False)

app.layout = html.Div(
    [dcc.Graph(id='o-predicted-bracket', figure=bracket_figure)])


@app.callback()
def update_output():

    pass


if __name__ == '__main__':
    app.run_server(debug=True)
