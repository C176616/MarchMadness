from dash import Dash, dcc, html, Input, Output, dash_table
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn import linear_model
import numpy as np
import plotly.graph_objects as go


app = Dash(__name__)

df = pd.read_csv('training_set.csv')
df_corr = df.corr()
cols = ['deltaSeed', 'deltaWinPct', 'deltaPointsFor', 'deltaFGM', 'deltaAst', 'deltaBlk']
# X = df[cols]
y = df['Result']
testModelType = 'nope'

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
fig = go.Figure()
fig.add_trace(
    go.Heatmap(
        x = ['Result'],
        y = df_corr.index,
        z = np.array(df_corr),
        text=df_corr.values,
        texttemplate='%{text:.2f}'
    )
)
app.layout = html.Div([
    html.H1("March Madness Machine Learning"),
    dcc.Graph(
        id='example-graph',
        figure=fig
    ),
    dcc.Dropdown(df.columns[4:], multi=True, id='i-model-features'),
    # dcc.Dropdown(['Linear', 'Logistic', 'Random Tree', 'Random Forest', 'Neural Net'], 
    #     'Random Forest', 
    #     id='i-model-type'),
    
    html.Br(),
    html.Div(id='my-output'),
    html.Br(),
    
    dcc.RadioItems(['Linear', 'Logistic', 'Random Tree', 'Random Forest', 'Neural Net'], 'Random Forest', id='i-model-type'),
    dcc.RangeSlider(
        df['Season'].min(),
        df['Season'].max(),
        step=None,
        value=[df['Season'].min(), df['Season'].max()],
        marks={str(Season): str(Season) for Season in df['Season'].unique()},
        id='i-season-range'
    ),
    html.Div(id='output-container-range-slider'),

    dcc.Slider(min=0,max=100,step=10, id='split-slider', value=70, marks=None, tooltip={"placement": "bottom", "always_visible": True}),
    html.Div(id='output2'),
    html.Div(id='output3'),
    
])


@app.callback(
    Output('output-container-range-slider', 'children'),
    Output('output2', 'children'),
    Output('output3', 'children'),
    Input('i-season-range', 'value'),
    Input('i-model-type', 'value'),
    Input('i-model-features','value'))
def update_output(value, modelType, modelFeatures):
    a = 0
    cols = modelFeatures
    df_modelResults = pd.DataFrame(columns=['Season','Error','Accuracy'])
    # Linear
    if modelType == 'Linear':
        testModelType = 'yes thats it'
        X = df[cols][(df['Season']>=value[0]) & (df['Season']<=value[1])]
        y = df['Result'][(df['Season']>=value[0]) & (df['Season']<=value[1])]
        linearModel = linear_model.LinearRegression()
        linearModel.fit(X,y)
        a = linearModel.score(X,y)

        for year in range(value[0],value[1]):
            if year != 2020:
                X_test = df[df['Season'] == year][cols]
                y_test = df[df['Season'] == year]['Result']

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
            
    # Logistic
    # Random Tree
    # Random Forest
    # Neural Net

    else:
        testModelType = 'yeah thats wrong'
    
    print(df_modelResults)
    resultsTable = generate_table(df_modelResults)
    returnTable = dash_table.DataTable(
        data=df_modelResults.to_dict('records'), 
        columns=[{"name": i, "id": i} for i in df_modelResults.columns]
    )
    return 'You have selected "{}"'.format(a), testModelType, returnTable

if __name__ == '__main__':
    app.run_server(debug=True)
