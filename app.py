from dash import Dash, dcc, html, Input, Output
import pandas as pd

app = Dash(__name__)

df = pd.read_csv('training_set.csv')

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

app.layout = html.Div([
    html.H6("Change the value in the text box to see callbacks in action!"),
    generate_table(df),
    html.Div([
        "Input: ",
        dcc.Input(id='my-input', value='initial value', type='text')
    ]),
    html.Label('Dropdown'),
        dcc.Dropdown(['New York City', 'Montréal', 'San Francisco']),
    html.Br(),
    html.Div(id='my-output'),
    html.Br(),
    dcc.Dropdown(['New York City', 'Montréal', 'San Francisco'],
                ['Montréal', 'San Francisco'],
                multi=True),
    dcc.RadioItems(['New York City', 'Montréal', 'San Francisco'], 'Montréal'),
    dcc.RangeSlider(
        df['Season'].min(),
        df['Season'].max(),
        step=None,
        value=[df['Season'].min(), df['Season'].max()],
        marks={str(Season): str(Season) for Season in df['Season'].unique()},
        id='my-range-slider'
    ),
    html.Div(id='output-container-range-slider')
    
])


@app.callback(
    Output('output-container-range-slider', 'children'),
    [Input('my-range-slider', 'value')])
def update_output(value):
    return 'You have selected "{}"'.format(value)




if __name__ == '__main__':
    app.run_server(debug=True)
