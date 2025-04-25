from dash import Dash, html, dcc, callback, Output, Input, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from google.cloud import storage
from io import StringIO
import os

# Initialize Google Cloud Storage client
def get_gcs_data(bucket_name, blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    data = blob.download_as_text()
    return pd.read_csv(StringIO(data))

# Sample data (replace with actual GCS calls)
def load_sample_data():
    stages = ['Pre-Seed', 'Seed', 'Series A', 'Series B', 'Series C', 'Series D+']
    counts = [50, 347, 313, 223, 150, 205]
    success_rates = [0.0, 0.6, 20.0, 93.7, 96.6, 100.0]
    return pd.DataFrame({
        'Stage': stages,
        'Count': counts,
        'SuccessRate': success_rates
    })

df = load_sample_data()

# Initialize the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Startup Analytics Platform"
server = app.server

# Navbar component (fixed the parenthesis issue here)
def get_navbar():
    return dbc.Navbar(
        dbc.Container([
            html.A(
                dbc.Row([
                    dbc.Col(html.Img(src="/assets/logo.png", height="30px")),
                    dbc.Col(dbc.NavbarBrand("Startup Analytics", className="ms-2")),
                ], align="center", className="g-0"),
                href="/",
                style={"textDecoration": "none"},
            ),
            dbc.Nav([
                dbc.NavItem(dbc.NavLink("Home", href="/")),
                dbc.NavItem(dbc.NavLink("Project Objective", href="/objective")),
                dbc.NavItem(dbc.NavLink("Methods", href="/methods")),
                dbc.NavItem(dbc.NavLink("Findings", href="/findings")),
                dbc.NavItem(dbc.NavLink("Data Explorer", href="/explorer")),
            ], className="ms-auto", navbar=True)
        ]),
        color="primary",
        dark=True,
        sticky="top"
    )


# Corrected Home/Landing Page
home_layout = html.Div([
    get_navbar(),
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Startup Funding Analytics", className="text-center my-4"),
                html.P("Data-driven insights for startup investment decisions", className="lead text-center mb-5"),
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                dcc.Graph(
                    figure=px.bar(df, x='Stage', y='Count', 
                                title="Startup Distribution by Funding Stage",
                                color='Stage',
                                color_discrete_sequence=px.colors.sequential.Plasma_r)
                    .update_layout(showlegend=False),
                ),
                html.P("Our dataset includes 1,748 startups across various funding stages, with Seed and Series A being most common.", 
                      className="text-center mt-2")
            ], md=6),
            
            dbc.Col([
                dcc.Graph(
                    figure=px.line(df, x='Stage', y='SuccessRate',
                                 title="Success Rate by Funding Stage",
                                 markers=True)
                    .update_yaxes(title="Success Rate (%)")
                    .update_traces(line_color='#2ca02c'),
                ),
                html.P("Success rates increase dramatically after Series B, reaching 100% for Series D and beyond.", 
                      className="text-center mt-2")
            ], md=6)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H3("About This Project", className="mt-5"),
                    html.P("This platform provides data-driven insights into startup funding patterns, success predictors, and anomaly detection. Our analysis of 1,748 startups reveals key patterns that can inform investment decisions and startup strategy."),
                    
                    html.H4("Key Features:", className="mt-4"),
                    html.Ul([
                        html.Li("Funding stage prediction with 82.8% accuracy"),
                        html.Li("Anomaly detection with 99.4% accuracy"),
                        html.Li("Success/failure risk assessment"),
                        html.Li("Interactive data exploration")
                    ])
                ], className="p-4 bg-light rounded")
            ], width=12)
        ])
    ], className="my-4")
])

# Project Objective Page
objective_layout = html.Div([
    get_navbar(),
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Project Objective", className="text-center my-4"),
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H2("Goals"),
                    html.P("The primary objectives of this project are:"),
                    html.Ul([
                        html.Li("Develop predictive models for startup funding stages"),
                        html.Li("Identify anomalies in funding patterns"),
                        html.Li("Predict startup success probabilities"),
                        html.Li("Provide actionable insights for investors and founders")
                    ]),
                    
                    html.H2("Data Sources", className="mt-5"),
                    html.P("Our analysis leverages multiple data sources:"),
                    html.Ul([
                        html.Li("1,748 startup records from Crunchbase and PitchBook"),
                        html.Li("Historical funding data (2010-2023)"),
                        html.Li("Employee growth metrics"),
                        html.Li("Industry classification data")
                    ]),
                    html.P("All data is securely stored in Google Cloud Storage.", className="mt-2 text-muted"),
                    
                    html.H2("Target Audience", className="mt-5"),
                    html.Ul([
                        html.Li("Venture capital and angel investors"),
                        html.Li("Startup founders and executives"),
                        html.Li("Accelerators and incubators"),
                        html.Li("Economic researchers")
                    ])
                ], className="p-4")
            ], md=8),
            
            dbc.Col([
                html.Div([
                    html.H4("Data Overview", className="mb-3"),
                    html.P("Key statistics about our dataset:"),
                    html.Ul([
                        html.Li("1,748 startups analyzed"),
                        html.Li("16 funding stages classified"),
                        html.Li("17 engineered features"),
                        html.Li("4 key predictive factors identified")
                    ]),
                    html.Img(src="/assets/data-flow.png", className="img-fluid mt-3")
                ], className="p-4 border rounded")
            ], md=4)
        ])
    ], className="my-4")
])

# Analytical Methods Page
methods_layout = html.Div([
    get_navbar(),
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Analytical Methods", className="text-center my-4"),
                html.P("Technical details of our modeling approach", className="text-center text-muted mb-5"),
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H2("Funding Stage Prediction"),
                    html.P("We employed several machine learning techniques to predict startup funding stages:"),
                    
                    html.H4("XGBoost Classifier", className="mt-3"),
                    html.Ul([
                        html.Li("82.8% accuracy on test set"),
                        html.Li("17 engineered features"),
                        html.Li("4 key predictors: funding amount, employee count, previous rounds, funding velocity")
                    ]),
                    
                    html.H4("Feature Engineering", className="mt-4"),
                    html.P("Key transformations applied to the raw data:"),
                    html.Ul([
                        html.Li("Log transformation of funding amounts"),
                        html.Li("Temporal features for funding velocity"),
                        html.Li("Industry-normalized metrics"),
                        html.Li("Employee growth rates")
                    ]),
                    
                    html.H4("Model Validation", className="mt-4"),
                    html.Ul([
                        html.Li("10-fold cross validation"),
                        html.Li("Stratified train-test split (80/20)"),
                        html.Li("Confusion matrix analysis"),
                        html.Li("Feature importance evaluation")
                    ])
                ], className="p-4 border rounded mb-4"),
                
                html.Div([
                    html.H2("Anomaly Detection"),
                    html.P("We used Isolation Forest for identifying unusual funding patterns:"),
                    
                    html.H4("Algorithm Details", className="mt-3"),
                    html.Ul([
                        html.Li("Contamination parameter: 0.01"),
                        html.Li("99.4% accuracy on test set"),
                        html.Li("Perfect precision (1.0)"),
                        html.Li("Multiple detection methods combined")
                    ]),
                    
                    html.H4("Technical References", className="mt-4"),
                    html.Ul([
                        html.Li(html.A("XGBoost Documentation", href="https://xgboost.readthedocs.io/", target="_blank")),
                        html.Li(html.A("Isolation Forest Paper", href="https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf", target="_blank")),
                        html.Li(html.A("Scikit-learn Documentation", href="https://scikit-learn.org/", target="_blank"))
                    ])
                ], className="p-4 border rounded")
            ], md=8),
            
            dbc.Col([
                html.Div([
                    html.H4("Model Architecture", className="mb-3"),
                    html.Img(src="/assets/model-architecture.png", className="img-fluid mb-3"),
                    html.P("Our ensemble approach combines multiple models for robust predictions.", className="text-muted"),
                    
                    html.H4("Performance Metrics", className="mt-4"),
                    dcc.Graph(
                        figure=go.Figure(
                            data=[go.Bar(
                                x=['XGBoost', 'Random Forest', 'Logistic Regression'],
                                y=[82.8, 82.0, 75.2],
                                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c']
                            )],
                            layout=dict(
                                title="Model Accuracy Comparison (%)",
                                yaxis=dict(range=[70, 85])
                            )
                        )
                    )
                ], className="p-4 border rounded")
            ], md=4)
        ])
    ], className="my-4")
])

# Major Findings Page
findings_layout = html.Div([
    get_navbar(),
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Major Findings", className="text-center my-4"),
                html.P("Key insights from our analysis", className="text-center text-muted mb-5"),
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                dcc.Graph(
                    figure=px.scatter(
                        df, x='Count', y='SuccessRate', 
                        color='Stage', size='Count',
                        title="Success Rate vs. Prevalence by Stage",
                        labels={'Count': 'Number of Startups', 'SuccessRate': 'Success Rate (%)'}
                    ).update_traces(marker=dict(opacity=0.8)),
                    className="mb-4"
                )
            ], md=6),
            
            dbc.Col([
                dcc.Graph(
                    figure=go.Figure(
                        data=[go.Pie(
                            labels=['Seed', 'Series A', 'Series B', 'Series C+'],
                            values=[347, 313, 223, 355],
                            hole=0.4,
                            marker_colors=px.colors.sequential.Plasma_r
                        )],
                        layout=dict(title="Funding Stage Distribution")
                    ),
                    className="mb-4"
                )
            ], md=6)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H3("Key Insights", className="mb-3"),
                    html.Ul([
                        html.Li("Series B appears to be the critical inflection point for startup success"),
                        html.Li("Seed-stage startups have less than 1% success rate in our dataset"),
                        html.Li("Only 1.09% of funding patterns were identified as anomalies"),
                        html.Li("Employee count and funding velocity are the strongest predictors")
                    ])
                ], className="p-4 bg-light rounded mb-4")
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.H3("Interactive Exploration", className="mt-4 mb-3"),
                html.P("Select a funding stage to see detailed statistics:", className="mb-2"),
                
                dcc.Dropdown(
                    id='stage-selector',
                    options=[{'label': stage, 'value': stage} for stage in df['Stage'].unique()],
                    value='Series B',
                    clearable=False,
                    className="mb-3"
                ),
                
                html.Div(id='stage-stats', className="p-3 border rounded")
            ], md=6),
            
            dbc.Col([
                html.H3("Success Probability Calculator", className="mt-4 mb-3"),
                html.P("Adjust parameters to estimate success probability:", className="mb-2"),
                
                dbc.Label("Funding Stage", className="mt-2"),
                dcc.Dropdown(
                    id='calc-stage',
                    options=[{'label': s, 'value': i} for i, s in enumerate(df['Stage'])],
                    value=3,  # Default to Series B
                    className="mb-2"
                ),
                
                dbc.Label("Employee Count", className="mt-2"),
                dcc.Slider(id='calc-employees', min=1, max=1000, step=10, value=50,
                          marks={1: '1', 250: '250', 500: '500', 750: '750', 1000: '1000'}),
                
                dbc.Label("Funding Amount ($M)", className="mt-2"),
                dcc.Slider(id='calc-funding', min=0.1, max=100, step=0.5, value=10,
                          marks={0.1: '0.1', 25: '25', 50: '50', 75: '75', 100: '100'}),
                
                html.Div(id='success-prob', className="mt-4 p-3 bg-info text-white rounded")
            ], md=6)
        ])
    ], className="my-4")
])

# Data Explorer Page (with interactive table)
explorer_layout = html.Div([
    get_navbar(),
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Data Explorer", className="text-center my-4"),
                html.P("Explore the underlying dataset", className="text-center text-muted mb-4"),
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.P("Filter by funding stage:"),
                    dcc.Dropdown(
                        id='stage-filter',
                        options=[{'label': 'All Stages', 'value': 'all'}] + 
                                [{'label': s, 'value': s} for s in df['Stage'].unique()],
                        value='all',
                        className="mb-3"
                    ),
                    
                    dash_table.DataTable(
                        id='data-table',
                        columns=[{"name": i, "id": i} for i in df.columns],
                        data=df.to_dict('records'),
                        page_size=10,
                        style_table={'overflowX': 'auto'},
                        style_cell={
                            'textAlign': 'left',
                            'padding': '8px'
                        }
                    )
                ], className="p-4 border rounded")
            ], width=12)
        ])
    ], className="my-4")
])

# App layout with routing
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# Callbacks for interactive components
@callback(
    Output('stage-stats', 'children'),
    Input('stage-selector', 'value')
)
def update_stage_stats(selected_stage):
    stage_data = df[df['Stage'] == selected_stage].iloc[0]
    return [
        html.H4(f"{selected_stage} Statistics"),
        html.P(f"Number of Startups: {stage_data['Count']}"),
        html.P(f"Success Rate: {stage_data['SuccessRate']}%"),
        html.P("Typical Characteristics:", className="mt-3"),
        html.Ul([
            html.Li("Employee count: 20-50" if selected_stage in ['Seed', 'Pre-Seed'] else 
                  "Employee count: 50-200" if selected_stage == 'Series A' else
                  "Employee count: 200-500"),
            html.Li("Funding range: $1-3M" if selected_stage == 'Seed' else
                  "$5-15M" if selected_stage == 'Series A' else
                  "$20-50M")
        ])
    ]

@callback(
    Output('success-prob', 'children'),
    [Input('calc-stage', 'value'),
     Input('calc-employees', 'value'),
     Input('calc-funding', 'value')]
)
def calculate_success(stage_idx, employees, funding):
    base_rate = df.iloc[stage_idx]['SuccessRate']
    emp_factor = min(1.0, employees / 100)  # Normalize employee count
    funding_factor = min(1.5, funding / 20)  # Normalize funding amount
    prob = min(99, base_rate * emp_factor * funding_factor)
    return [
        html.H4("Estimated Success Probability"),
        html.P(f"{prob:.1f}%", style={'fontSize': '24px', 'fontWeight': 'bold'})
    ]

@callback(
    Output('data-table', 'data'),
    Input('stage-filter', 'value')
)
def filter_table(stage):
    if stage == 'all':
        return df.to_dict('records')
    return df[df['Stage'] == stage].to_dict('records')

# Routing callback
@callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == '/objective':
        return objective_layout
    elif pathname == '/methods':
        return methods_layout
    elif pathname == '/findings':
        return findings_layout
    elif pathname == '/explorer':
        return explorer_layout
    else:
        return home_layout

if __name__ == '__main__':
    app.run(debug=True)