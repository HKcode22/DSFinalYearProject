from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px
from google.cloud import storage
import os
from io import StringIO

# Function to fetch CSV data from Google Cloud Storage
def get_csv_from_gcs(bucket_name, source_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    data = blob.download_as_text()
    return pd.read_csv(StringIO(data))

# Placeholder for real data source
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv')

# Initialize Dash app
app = Dash()
server = app.server  # Required for Google App Engine deployment

# Define Navbar
def get_navbar():
    return html.Nav([
        html.A("Home", href="/"),
        html.A("Project Objective", href="/objective"),
        html.A("Analytical Methods", href="/methods"),
        html.A("Major Findings", href="/findings")
    ], className="navbar")

# Define Layouts
home_layout = html.Div([
    get_navbar(),
    html.H1("Welcome to the Project Dashboard"),
    html.P("This dashboard provides insights into our data analysis project."),
    dcc.RadioItems(options=['pop', 'lifeExp', 'gdpPercap'], value='lifeExp', id='controls-and-radio-item'),
    dash_table.DataTable(data=df.to_dict('records'), page_size=6),
    dcc.Graph(figure={}, id='controls-and-graph')
])

objective_layout = html.Div([
    get_navbar(),
    html.H2("Project Objective"),
    html.P("Describe project goals, data sources, and motivation here.")
])

methods_layout = html.Div([
    get_navbar(),
    html.H2("Analytical Methods"),
    html.P("Explain major techniques and methodologies used.")
])

findings_layout = html.Div([
    get_navbar(),
    html.H2("Major Findings"),
    html.P("Summarize key insights and results."),
    dcc.Graph(id='findings-graph', figure={})
])

# Define App Routes
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/objective':
        return objective_layout
    elif pathname == '/methods':
        return methods_layout
    elif pathname == '/findings':
        return findings_layout
    else:
        return home_layout

@app.callback(
    Output('controls-and-graph', 'figure'),
    Input('controls-and-radio-item', 'value')
)
def update_graph(col_chosen):
    fig = px.histogram(df, x='continent', y=col_chosen, histfunc='avg')
    return fig

if __name__ == '__main__':
    app.run(debug=True)
