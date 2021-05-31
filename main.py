import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output
import json
import base64
from tensorflow.keras.datasets import mnist


(train_X, train_y), (test_X, test_y) = mnist.load_data()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

umap_data = pd.read_csv('umap.csv')
umap_data['Value'] = [str(int(i)) for i in umap_data['Value']]

umap_fig = px.scatter_3d(umap_data, x='x', y='y', z='z', color='Value')

sample_fig = px.imshow(test_X[0])


widgets = html.Div(
    [
        dbc.Row(
            dbc.Col(
                dbc.Card(
                    [dbc.CardHeader("Title"),
                     dbc.CardBody(
                        [dcc.Graph(id='umap_plot', figure=umap_fig)
                         ]
                    )
                    ]
                )
            ), style={"padding-bottom": "25px"}),

        dbc.Row(
            dbc.Col(
                dbc.Card(
                    [dbc.CardHeader("Title"),
                     dbc.CardBody(
                        [dcc.Graph(id="img_view", figure=sample_fig)
                         ]
                    )
                    ]
                )
            ), style={"padding-bottom": "25px"}),

    ]
)


app.layout = html.Div(children=[
    dbc.Jumbotron([
        html.H1(children='Hello Dash'),

        html.Div(children='''
            Dash: A web application framework for Python.
        '''),
    ]
    ),
    widgets,
    dbc.Toast(
        [html.P("This is the content of the toast", className="mb-0")],
        id="auto-toast",
        header="Info",
        icon="info",
        duration=4000,
        style={"position": "fixed", "top": 66, "right": 10, "width": 350},

    )

])


@app.callback([Output('img_view', 'figure')], [Input("umap_plot", "clickData")]
              )
def show_clicked_point(click_data):
    if click_data:
        sample_number = int(click_data['points'][0]['pointNumber'])
        return [px.imshow(test_X[sample_number])]
    else:
        return [px.imshow(test_X[0])]


if __name__ == '__main__':
    app.run_server(debug=True, threaded=True)
