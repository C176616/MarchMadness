import os
from itertools import cycle
import json
import jsonpickle

from dash import Dash, dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import numpy as np

from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier

# from src.team import Team
from src.tournament import Tournament
from src.matchPrediction import MatchPrediction
from src.game import Game
from src.team import Team
"""_summary_
This app requires 
training_set.csv
training_set_stage2.csv
/data_stage2/MSampleSubmissionStage2.csv

a pre-loaded tournamentLayout.json file

Returns
-------
_type_
    _description_
    """
app = Dash(external_stylesheets=[dbc.themes.DARKLY])
cwd = os.getcwd()

pio.renderers.deafult = "notebook"

## Import required data
df_training_set = pd.read_csv('training_set.csv')
df_training_set_stage2 = pd.read_csv("training_set_stage2.csv")
df_stage1Combinations = pd.read_csv(cwd +
                                    "/data_stage2/MSampleSubmissionStage2.csv")

# Open the pre-populated tournament layout JSON file
with open('tournamentLayout.json', 'r') as f:
    jsonData = json.load(f)

tourn = jsonpickle.decode(jsonData)

# populate the tournament predictions list
tourn.populatePredictionsList(df_stage1Combinations)

# simluate the tournament
tourn.simulateTournament()

# Create the initial heatmap figure
df_corr = df_training_set.corr()
heatmap_figure = go.Figure()
heatmap_figure.add_trace(
    go.Heatmap(
        x=['Result'],
        y=df_corr.index,
        z=np.array(df_corr),
        text=df_corr.values,
        texttemplate='%{text:.2f}',
    ))

heatmap_figure.update_layout(template='plotly_dark')

# Create the initial bracket figure
bracket_figure = go.Figure()
bracket_figure.update_layout(plot_bgcolor="#3c3c3c",
                             showlegend=False,
                             template='plotly_dark')
bracket_figure.update_xaxes(showticklabels=False)
bracket_figure.update_yaxes(showticklabels=False)

# initialize a pandas dataframe to display the results of testing
data = {'Season': [2022], 'Error': [0], 'Accuracy': [0]}
df_modelResults = pd.DataFrame(data)

load_figure_template("darkly")
################# app.layout
app.layout = html.Div([
    # Header
    html.H1("March Madness Machine Learning"),
    # Heatmap Graph
    html.Div([
        html.H2("Explore"),
        html.Div(
            [
                dcc.Graph(
                    id='o-heatmap-figure',
                    figure=heatmap_figure,
                ),
            ],
            style={
                'width': '80%',
                'vertical-align': 'middle',
                'display': 'inline-block',
                'padding': '50px'
            }),
        html.Div([
            html.H3("Tournament Seasons to Include in Data:"),
            dcc.RangeSlider(
                df_training_set['Season'].min(),
                df_training_set['Season'].max(),
                step=None,
                value=[
                    df_training_set['Season'].min(),
                    df_training_set['Season'].max()
                ],
                marks={
                    str(Season): str(Season)
                    for Season in df_training_set['Season'].unique()
                },
                id='i-season-range')
        ],
                 style={
                     'width': '100%',
                     'horizontal-align': 'middle'
                 }),
    ],
             style={
                 'width': '30%',
                 'display': 'inline-block'
             }),
    html.Div([
        html.H2("Build"),
        html.Div([
            dcc.Dropdown(df_training_set.columns[4:],
                         multi=True,
                         id='i-model-features',
                         placeholder="Select Features to Include"),
        ],
                 style={
                     'width': '50%',
                     'color': 'rgb(50,50,50)'
                 }),
        html.Br(),
        html.Div([
            dcc.RadioItems(
                ['Linear', 'Logistic', 'Random Forest', 'Neural Net'],
                'Linear',
                id='i-model-type'),
        ]),
    ],
             style={
                 'width': '30%',
                 'display': 'inline-block',
                 'vertical-align': 'top'
             }),

    #model creation
    #Features
    html.Div([
        html.H2("Test"),
        dash_table.DataTable(
            data=df_modelResults.to_dict('records'),
            id='o-results-table',
            style_header={
                'backgroundColor': 'rgb(30, 30, 30)',
                'color': 'white'
            },
            style_data={
                'backgroundColor': 'rgb(50, 50, 50)',
                'color': 'white'
            },
        ),
    ],
             style={
                 'width': '25%',
                 'display': 'inline-block',
                 'vertical-align': 'top'
             }),
    html.Div([
        html.H2("Predict"),
        html.Div([dcc.Graph(id='o-predicted-bracket', figure=bracket_figure)])
    ],
             style={
                 'width': '100%',
                 'display': 'inline-block',
                 'vertical-align': 'top'
             })
])


@app.callback(Output('o-heatmap-figure', 'figure'),
              Output('o-results-table', 'data'),
              Output('o-predicted-bracket', 'figure'),
              Input('i-season-range', 'value'), Input('i-model-type', 'value'),
              Input('i-model-features', 'value'))
def update_output(seasonRange, modelType, modelFeatures):
    """_summary_

    Parameters
    ----------
    seasonRange : _type_
        _description_
    modelType : _type_
        _description_
    modelFeatures : _type_
        _description_
    """
    cols = modelFeatures

    #clear the existing return variables
    df_modelResults = pd.DataFrame(columns=['Season', 'Error', 'Accuracy'])
    heatmap_figure = go.Figure()
    bracket_figure = go.Figure()

    df_corr = df_training_set[(df_training_set['Season'] >= seasonRange[0]) & (
        df_training_set['Season'] <= seasonRange[1])].corr()

    heatmap_figure.add_trace(
        go.Heatmap(x=['Result'],
                   y=df_corr.index,
                   z=np.array(df_corr),
                   text=df_corr.values,
                   texttemplate='%{text:.2f}'))
    heatmap_figure.update_layout(height=600, yaxis_nticks=len(df_corr))
    bracket_figure.update_layout(width=1600, height=1000)

    X = df_training_set[cols][(df_training_set['Season'] >= seasonRange[0])
                              & (df_training_set['Season'] <= seasonRange[1])]
    y = df_training_set['Result'][
        (df_training_set['Season'] >= seasonRange[0])
        & (df_training_set['Season'] <= seasonRange[1])]

    if modelType == 'Random Forest':
        print('random forest')
        RFClassifier = RandomForestClassifier(n_estimators=1)
        RFClassifier.fit(X, y)

        X_pred = df_training_set_stage2[cols]
        pred = RFClassifier.predict_proba(X_pred)[:, 1]
        df_stage1Combinations['Pred'] = np.round(pred, 2)
        tourn.populatePredictionsList(df_stage1Combinations)

        for year in range(seasonRange[0], seasonRange[1]):
            if year != 2020:
                X_test = df_training_set[df_training_set['Season'] ==
                                         year][cols]
                y_test = df_training_set[df_training_set['Season'] ==
                                         year]['Result']

                #not sure if this should be :,1 or :,0
                df_results = X_test
                df_results['Prediction'] = RFClassifier.predict_proba(
                    X_test)[:, 1]
                df_results['Result'] = y_test

                # df_results

                df_results.loc[df_results['Prediction'] > 0.9,
                               'Prediction'] = 0.99
                df_results.loc[df_results['Prediction'] < 0.1,
                               'Prediction'] = 0.01

                correct = df_results.loc[(df_results['Result'] == 0) & (
                    df_results['Prediction'] < 0.5)].shape[0]
                correct = correct + df_results.loc[
                    (df_results['Result'] == 1)
                    & (df_results['Prediction'] > 0.5)].shape[0]

                # print("correct", correct)
                total = df_results.shape[0]
                # print("total", total)
                # print("accuracy" + total)
                # print(correct)

                accuracy = round(correct / total, 2)
                # print("accuracy", accuracy)

                error = round(
                    -np.log(1 - df_results.loc[df_results['Result'] == 0]
                            ['Prediction']).mean(), 2)
                # print("error", error)
                data = {
                    'Season': [year],
                    'Error': [error],
                    'Accuracy': [accuracy]
                }
                # print(data)

                # df_modelResults = df_modelResults.append(data, ignore_index=True)

                df_newRow = pd.DataFrame(data)
                # print(df_newRow)
                df_modelResults = pd.concat([df_modelResults, df_newRow],
                                            ignore_index=True)


## Logistic
    elif modelType == 'Logistic':
        print('logistic')
        model = linear_model.LogisticRegression(solver='lbfgs')
        model.fit(X, y)

        X_pred = df_training_set_stage2[cols]
        pred = model.predict_proba(X_pred)[:, 1]
        df_stage1Combinations['Pred'] = np.round(pred, 2)
        tourn.populatePredictionsList(df_stage1Combinations)

        for year in range(seasonRange[0], seasonRange[1]):
            if year != 2020:
                X_test = df_training_set[df_training_set['Season'] ==
                                         year][cols]
                y_test = df_training_set[df_training_set['Season'] ==
                                         year]['Result']

                df_results = X_test
                df_results['Prediction'] = model.predict_proba(X_test)[:, 1]
                df_results['Result'] = y_test

                correct = df_results.loc[(df_results['Result'] == 0) & (
                    df_results['Prediction'] < 0.5)].shape[0]
                correct = correct + df_results.loc[
                    (df_results['Result'] == 1)
                    & (df_results['Prediction'] > 0.5)].shape[0]

                total = df_results.shape[0]

                accuracy = round(correct / total, 2)

                df_results.loc[df_results['Prediction'] > 0.9,
                               'Prediction'] = 0.99
                df_results.loc[df_results['Prediction'] < 0.1,
                               'Prediction'] = 0.01

                error = round(
                    -np.log(1 - df_results.loc[df_results['Result'] == 0]
                            ['Prediction']).mean(), 2)
                data = {
                    'Season': [year],
                    'Error': [error],
                    'Accuracy': [accuracy]
                }
                print(data)
                df_newRow = pd.DataFrame(data)
                df_modelResults = pd.concat([df_modelResults, df_newRow],
                                            ignore_index=True)

    elif modelType == 'Linear':
        print('linear')
        linearModel = linear_model.LinearRegression()
        linearModel.fit(X, y)

        X_pred = df_training_set_stage2[cols]
        pred = linearModel.predict(X_pred)
        df_stage1Combinations['Pred'] = np.round(pred, 2)
        # print(df_stage1Combinations)
        tourn.populatePredictionsList(df_stage1Combinations)

        #test the model
        for year in range(seasonRange[0], seasonRange[1]):
            if year != 2020:
                X_test = df_training_set[df_training_set['Season'] ==
                                         year][cols]
                y_test = df_training_set[df_training_set['Season'] ==
                                         year]['Result']

                df_results = X_test
                df_results['Prediction'] = linearModel.predict(X_test)
                df_results['Result'] = y_test

                correct = df_results.loc[(df_results['Result'] == 0) & (
                    df_results['Prediction'] < 0.5)].shape[0]
                correct = correct + df_results.loc[
                    (df_results['Result'] == 1)
                    & (df_results['Prediction'] > 0.5)].shape[0]

                total = df_results.shape[0]

                accuracy = round(correct / total, 2)
                # print("correct", correct)
                # print("total", total)
                # print("accuracy", accuracy)

                error = round(
                    -np.log(1 - df_results.loc[df_results['Result'] == 0]
                            ['Prediction']).mean(), 2)
                data = {
                    'Season': [year],
                    'Error': [error],
                    'Accuracy': [accuracy]
                }
                # print(data)

                # df_modelResults = df_modelResults.append(data, ignore_index=True)

                df_newRow = pd.DataFrame(data)
                df_modelResults = pd.concat([df_modelResults, df_newRow],
                                            ignore_index=True)

    print("made it here1")

    tourn.simulateTournament()

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
                tourn.root.left.value + " " + str(tourn.root.left.winPct) +
                " " + tourn.root.team2.getString()
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
                tourn.root.left.value + " " + str(tourn.root.left.winPct) +
                " " + tourn.root.team1.getString()
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
    bracket_figure.update_layout(plot_bgcolor="#3c3c3c", showlegend=False)
    bracket_figure.update_xaxes(showticklabels=False,
                                showgrid=False,
                                zeroline=False)
    bracket_figure.update_yaxes(showticklabels=False,
                                showgrid=False,
                                zeroline=False)

    # store the results to the df_modelResults which is tied to the table
    resultsData = df_modelResults.to_dict('records')

    return heatmap_figure, resultsData, bracket_figure

if __name__ == '__main__':
    app.run_server(debug=True)
