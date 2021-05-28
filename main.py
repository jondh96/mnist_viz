import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

gap = px.data.gapminder()

fig_gap = px.scatter(gap.query("year==2007"), x="gdpPercap", y="lifeExp",
                     size="pop", color="continent",
                     hover_name="country", log_x=True, size_max=60)

fig_gap.update_layout(template='plotly_white')

fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

fig.update_layout(template='plotly_white')

N = 100000
fig_scatter = go.Figure(data=go.Scattergl(
    x=np.random.randn(N),
    y=np.random.randn(N),
    mode='markers',
    marker=dict(
        color=np.random.randn(N),
        colorscale='Viridis',
        line_width=1
    )
))

fig_scatter.update_layout(template='plotly_white')


fig_map = go.Figure(go.Scattermapbox(
    fill="toself",
    lon=[-74, -70, -70, -74], lat=[47, 47, 45, 45],
    marker={'size': 10, 'color': "orange"}))

fig_map.update_layout(
    mapbox={
        'style': "stamen-terrain",
        'center': {'lon': -73, 'lat': 46},
        'zoom': 5},
    showlegend=False)


widgets = html.Div(
    [
        dbc.Row(
            dbc.Col(
                dbc.Card(
                    [dbc.CardHeader("Title"),
                     dbc.CardBody(
                        [dcc.Graph(id='graph-with-slider', figure=fig_gap),
                         dcc.Slider(
                            id='year-slider',
                            min=gap['year'].min(),
                            max=gap['year'].max(),
                            value=gap['year'].min(),
                            marks={str(year): str(year)
                                   for year in gap['year'].unique()},
                            step=None
                        )
                        ]
                    )
                    ]
                )
            ), style={"padding-bottom": "25px"}),

        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Title"),
                            dbc.CardBody(
                                dcc.Graph(id='example_graph1', figure=fig))
                        ]
                    )
                ),

                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Title"),
                            dbc.CardBody(
                                dcc.Graph(id='example_graph2', figure=fig_scatter))
                        ]
                    )
                ),

                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Title"),
                            dbc.CardBody(
                                dcc.Graph(id='example_graph3', figure=fig_map))
                        ]
                    )
                ),
            ]
        ),
    ],
    style={"padding": "25px"}
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
    dbc.Spinner(color="danger")

])


@app.callback(
    Output('graph-with-slider', 'figure'),
    Input('year-slider', 'value'))
def update_figure(selected_year):
    filtered_df = gap[gap.year == selected_year]

    fig = px.scatter(filtered_df, x="gdpPercap", y="lifeExp",
                     size="pop", color="continent", hover_name="country",
                     log_x=True, size_max=55)

    fig.update_layout(transition_duration=500, template="plotly_white")

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
