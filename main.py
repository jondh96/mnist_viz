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


card = dbc.Card(
    [
        dbc.CardHeader(
            dbc.Tabs(
                [
                    dbc.Tab(label="Tab 1", tab_id="tab-1"),
                    dbc.Tab(label="Tab 2", tab_id="tab-2"),
                    dbc.Tab(label="Tab 3", tab_id="tab-3"),
                ],
                id="card-tabs",
                card=True,
                active_tab="tab-1",
            )
        ),
        dbc.CardBody(html.Div(id="card-content", className="card-text")),
    ]
)


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

        dbc.Row(
            dbc.Col(
                card
            )

        )
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
    dbc.Spinner(color="danger"),
    dbc.Toast(
        [html.P("This is the content of the toast", className="mb-0")],
        id="auto-toast",
        header="Info",
        icon="info",
        duration=4000,
        style={"position": "fixed", "top": 66, "right": 10, "width": 350},

    )

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


@app.callback(
    [Output("auto-toast", "children"), Output("auto-toast",
                                              "is_open")], [Input("card-tabs", "active_tab")]
)
def tab_content(active_tab):
    if active_tab == "tab-2":
        return html.P("The plot may take a moment to render"), True
    else:
        return html.P(""), False


@app.callback(
    Output("card-content", "children"), [Input("card-tabs", "active_tab")]
)
def tab_content(active_tab):
    if active_tab == "tab-1":
        return dcc.Graph(id='example_graph4', figure=fig)
    elif active_tab == "tab-2":
        return dcc.Graph(id="example_graph5", figure=fig_scatter)
    elif active_tab == "tab-3":
        return dcc.Graph(id='example_graph6', figure=fig_map)


if __name__ == '__main__':
    app.run_server(debug=True, threaded=True)
