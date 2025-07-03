"""Web application for interactive Lorenz attractor exploration."""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import json
import time

from ..core.lorenz import LorenzSystem
from ..core.simulator import Simulator
from ..core.parameters import LorenzParameters, InitialConditions, SimulationConfig
from ..visualization.plotter import LorenzPlotter


def create_app() -> dash.Dash:
    """Create and configure the Dash web application."""
    
    app = dash.Dash(__name__, title="Lorenz Attractor Explorer")
    
    # Initialize components
    simulator = Simulator()
    
    # Define the app layout
    app.layout = html.Div([
        html.Div([
            html.H1("Lorenz Attractor Explorer", 
                   style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '30px'}),
            
            html.Div([
                # Parameter Controls
                html.Div([
                    html.H3("System Parameters", style={'color': '#34495e'}),
                    
                    html.Label("Sigma (σ):", style={'fontWeight': 'bold'}),
                    dcc.Slider(
                        id='sigma-slider',
                        min=0.1, max=20, step=0.1, value=10,
                        marks={i: str(i) for i in range(0, 21, 5)},
                        tooltip={'placement': 'bottom', 'always_visible': True}
                    ),
                    
                    html.Label("Rho (ρ):", style={'fontWeight': 'bold', 'marginTop': '20px'}),
                    dcc.Slider(
                        id='rho-slider',
                        min=0.1, max=50, step=0.1, value=28,
                        marks={i: str(i) for i in range(0, 51, 10)},
                        tooltip={'placement': 'bottom', 'always_visible': True}
                    ),
                    
                    html.Label("Beta (β):", style={'fontWeight': 'bold', 'marginTop': '20px'}),
                    dcc.Slider(
                        id='beta-slider',
                        min=0.1, max=10, step=0.1, value=8/3,
                        marks={i: str(i) for i in range(0, 11, 2)},
                        tooltip={'placement': 'bottom', 'always_visible': True}
                    ),
                    
                    html.Hr(),
                    
                    html.H3("Initial Conditions", style={'color': '#34495e'}),
                    
                    html.Div([
                        html.Div([
                            html.Label("X₀:", style={'fontWeight': 'bold'}),
                            dcc.Input(id='x0-input', type='number', value=1.0, step=0.1),
                        ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '5%'}),
                        
                        html.Div([
                            html.Label("Y₀:", style={'fontWeight': 'bold'}),
                            dcc.Input(id='y0-input', type='number', value=1.0, step=0.1),
                        ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '5%'}),
                        
                        html.Div([
                            html.Label("Z₀:", style={'fontWeight': 'bold'}),
                            dcc.Input(id='z0-input', type='number', value=1.0, step=0.1),
                        ], style={'width': '30%', 'display': 'inline-block'}),
                    ]),
                    
                    html.Button("Random Initial Conditions", id='random-ic-button', 
                               style={'marginTop': '20px', 'backgroundColor': '#3498db', 
                                     'color': 'white', 'border': 'none', 'padding': '10px 20px',
                                     'borderRadius': '5px', 'cursor': 'pointer'}),
                    
                    html.Hr(),
                    
                    html.H3("Simulation Settings", style={'color': '#34495e'}),
                    
                    html.Label("Integration Method:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='integration-method',
                        options=[
                            {'label': 'Euler', 'value': 'euler'},
                            {'label': 'Runge-Kutta 4th Order', 'value': 'rk4'},
                            {'label': 'Adaptive RK', 'value': 'adaptive'},
                            {'label': 'Dormand-Prince 5(4)', 'value': 'dopri5'}
                        ],
                        value='rk4',
                        style={'marginBottom': '20px'}
                    ),
                    
                    html.Label("Number of Steps:", style={'fontWeight': 'bold'}),
                    dcc.Slider(
                        id='steps-slider',
                        min=1000, max=50000, step=1000, value=10000,
                        marks={i: f'{i//1000}k' for i in range(0, 51000, 10000)},
                        tooltip={'placement': 'bottom', 'always_visible': True}
                    ),
                    
                    html.Label("Time Step (dt):", style={'fontWeight': 'bold', 'marginTop': '20px'}),
                    dcc.Slider(
                        id='dt-slider',
                        min=0.001, max=0.1, step=0.001, value=0.01,
                        marks={i/100: f'{i/100:.3f}' for i in range(0, 11, 2)},
                        tooltip={'placement': 'bottom', 'always_visible': True}
                    ),
                    
                    html.Button("Run Simulation", id='run-button', 
                               style={'marginTop': '30px', 'backgroundColor': '#e74c3c', 
                                     'color': 'white', 'border': 'none', 'padding': '15px 30px',
                                     'borderRadius': '5px', 'cursor': 'pointer', 'fontSize': '16px',
                                     'fontWeight': 'bold'}),
                    
                ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top',
                         'padding': '20px', 'backgroundColor': '#ecf0f1', 'borderRadius': '10px',
                         'marginRight': '2%'}),
                
                # Visualization Area
                html.Div([
                    dcc.Tabs(id='viz-tabs', value='3d-plot', children=[
                        dcc.Tab(label='3D Trajectory', value='3d-plot'),
                        dcc.Tab(label='2D Projections', value='2d-projections'),
                        dcc.Tab(label='Phase Analysis', value='phase-analysis'),
                        dcc.Tab(label='Parameter Sweep', value='parameter-sweep'),
                    ]),
                    
                    html.Div(id='plot-container', style={'marginTop': '20px'}),
                    
                    # Real-time controls
                    html.Div([
                        html.H4("Real-time Controls", style={'color': '#34495e'}),
                        html.Button("Start Real-time", id='start-realtime-button', 
                                   style={'backgroundColor': '#27ae60', 'color': 'white', 
                                         'border': 'none', 'padding': '10px 20px',
                                         'borderRadius': '5px', 'cursor': 'pointer',
                                         'marginRight': '10px'}),
                        html.Button("Stop Real-time", id='stop-realtime-button', 
                                   style={'backgroundColor': '#e74c3c', 'color': 'white', 
                                         'border': 'none', 'padding': '10px 20px',
                                         'borderRadius': '5px', 'cursor': 'pointer'}),
                        dcc.Interval(id='realtime-interval', interval=100, n_intervals=0, disabled=True),
                    ], style={'marginTop': '20px', 'padding': '15px', 
                             'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),
                    
                ], style={'width': '70%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            ]),
            
            # Information Panel
            html.Div([
                html.H3("System Information", style={'color': '#34495e'}),
                html.Div(id='system-info', style={'backgroundColor': '#f8f9fa', 
                                                 'padding': '15px', 'borderRadius': '5px',
                                                 'fontFamily': 'monospace'}),
            ], style={'marginTop': '30px'}),
            
        ], style={'padding': '20px', 'maxWidth': '1400px', 'margin': '0 auto'}),
        
        # Hidden div to store simulation data
        html.Div(id='simulation-data', style={'display': 'none'}),
        html.Div(id='realtime-data', style={'display': 'none'}),
    ])
    
    # Callbacks
    @app.callback(
        [Output('x0-input', 'value'),
         Output('y0-input', 'value'),
         Output('z0-input', 'value')],
        [Input('random-ic-button', 'n_clicks')]
    )
    def generate_random_initial_conditions(n_clicks):
        if n_clicks is None:
            return 1.0, 1.0, 1.0
        
        # Generate random initial conditions
        ic = InitialConditions.true_random(scale=2.0)
        return ic.x, ic.y, ic.z
    
    @app.callback(
        Output('simulation-data', 'children'),
        [Input('run-button', 'n_clicks')],
        [State('sigma-slider', 'value'),
         State('rho-slider', 'value'),
         State('beta-slider', 'value'),
         State('x0-input', 'value'),
         State('y0-input', 'value'),
         State('z0-input', 'value'),
         State('integration-method', 'value'),
         State('steps-slider', 'value'),
         State('dt-slider', 'value')]
    )
    def run_simulation(n_clicks, sigma, rho, beta, x0, y0, z0, 
                      integration_method, num_steps, dt):
        if n_clicks is None:
            return json.dumps({})
        
        # Create parameters
        params = LorenzParameters(sigma=sigma, rho=rho, beta=beta)
        initial_conditions = InitialConditions(x=x0, y=y0, z=z0)
        config = SimulationConfig(
            dt=dt,
            num_steps=num_steps,
            integration_method=integration_method
        )
        
        # Create simulator and run
        system = LorenzSystem(params)
        sim = Simulator(system)
        result = sim.simulate(initial_conditions, config)
        
        # Store result data
        data = {
            'time': result.time.tolist(),
            'x': result.x.tolist(),
            'y': result.y.tolist(),
            'z': result.z.tolist(),
            'parameters': params.to_dict(),
            'initial_conditions': [x0, y0, z0],
            'metadata': result.metadata
        }
        
        return json.dumps(data)
    
    @app.callback(
        Output('plot-container', 'children'),
        [Input('viz-tabs', 'value'),
         Input('simulation-data', 'children')]
    )
    def update_visualization(active_tab, simulation_data):
        if not simulation_data:
            return html.Div("Run a simulation to see results.", 
                          style={'textAlign': 'center', 'color': '#7f8c8d', 
                                'fontSize': '18px', 'marginTop': '100px'})
        
        try:
            data = json.loads(simulation_data)
            if not data:
                return html.Div("No simulation data available.")
            
            time_data = np.array(data['time'])
            x_data = np.array(data['x'])
            y_data = np.array(data['y'])
            z_data = np.array(data['z'])
            
            if active_tab == '3d-plot':
                # 3D trajectory plot
                fig = go.Figure(data=[
                    go.Scatter3d(
                        x=x_data,
                        y=y_data,
                        z=z_data,
                        mode='lines',
                        line=dict(
                            color=np.linspace(0, 1, len(time_data)),
                            colorscale='Viridis',
                            width=2
                        ),
                        name='Trajectory'
                    ),
                    go.Scatter3d(
                        x=[x_data[0]],
                        y=[y_data[0]],
                        z=[z_data[0]],
                        mode='markers',
                        marker=dict(
                            color='red',
                            size=8
                        ),
                        name='Start'
                    )
                ])
                
                fig.update_layout(
                    title='Lorenz Attractor 3D Trajectory',
                    scene=dict(
                        xaxis_title='X',
                        yaxis_title='Y',
                        zaxis_title='Z',
                        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                    ),
                    height=600
                )
                
                return dcc.Graph(figure=fig)
            
            elif active_tab == '2d-projections':
                # 2D projections
                fig = go.Figure()
                
                # Create subplots
                from plotly.subplots import make_subplots
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('XY Projection', 'XZ Projection', 
                                  'YZ Projection', 'Z vs Time'),
                    specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                           [{'type': 'scatter'}, {'type': 'scatter'}]]
                )
                
                # XY projection
                fig.add_trace(
                    go.Scatter(x=x_data, y=y_data, mode='lines', 
                              name='XY', line=dict(color='blue', width=1)),
                    row=1, col=1
                )
                
                # XZ projection
                fig.add_trace(
                    go.Scatter(x=x_data, y=z_data, mode='lines', 
                              name='XZ', line=dict(color='green', width=1)),
                    row=1, col=2
                )
                
                # YZ projection
                fig.add_trace(
                    go.Scatter(x=y_data, y=z_data, mode='lines', 
                              name='YZ', line=dict(color='red', width=1)),
                    row=2, col=1
                )
                
                # Z vs Time
                fig.add_trace(
                    go.Scatter(x=time_data, y=z_data, mode='lines', 
                              name='Z(t)', line=dict(color='purple', width=1)),
                    row=2, col=2
                )
                
                fig.update_layout(height=600, title_text="2D Projections")
                fig.update_xaxes(title_text="X", row=1, col=1)
                fig.update_yaxes(title_text="Y", row=1, col=1)
                fig.update_xaxes(title_text="X", row=1, col=2)
                fig.update_yaxes(title_text="Z", row=1, col=2)
                fig.update_xaxes(title_text="Y", row=2, col=1)
                fig.update_yaxes(title_text="Z", row=2, col=1)
                fig.update_xaxes(title_text="Time", row=2, col=2)
                fig.update_yaxes(title_text="Z", row=2, col=2)
                
                return dcc.Graph(figure=fig)
            
            elif active_tab == 'phase-analysis':
                # Phase space analysis
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('3D Phase Space', 'Poincaré Section (Z=27)', 
                                  'Return Map', 'Power Spectrum'),
                    specs=[[{'type': 'scatter3d'}, {'type': 'scatter'}],
                           [{'type': 'scatter'}, {'type': 'scatter'}]]
                )
                
                # 3D phase space
                fig.add_trace(
                    go.Scatter3d(x=x_data, y=y_data, z=z_data, mode='lines',
                                line=dict(color='blue', width=2), name='Trajectory'),
                    row=1, col=1
                )
                
                # Poincaré section (simplified)
                poincare_indices = np.where(np.diff(np.sign(z_data - 27)))[0]
                if len(poincare_indices) > 0:
                    poincare_x = x_data[poincare_indices]
                    poincare_y = y_data[poincare_indices]
                    
                    fig.add_trace(
                        go.Scatter(x=poincare_x, y=poincare_y, mode='markers',
                                  marker=dict(size=3, color='red'), name='Poincaré'),
                        row=1, col=2
                    )
                    
                    # Return map
                    if len(poincare_indices) > 1:
                        poincare_z = z_data[poincare_indices]
                        fig.add_trace(
                            go.Scatter(x=poincare_z[:-1], y=poincare_z[1:], mode='markers',
                                      marker=dict(size=3, color='green'), name='Return Map'),
                            row=2, col=1
                        )
                
                # Power spectrum
                freqs = np.fft.fftfreq(len(z_data), time_data[1] - time_data[0])
                power = np.abs(np.fft.fft(z_data))**2
                
                fig.add_trace(
                    go.Scatter(x=freqs[1:len(freqs)//2], y=power[1:len(freqs)//2],
                              mode='lines', name='Power Spectrum'),
                    row=2, col=2
                )
                
                fig.update_layout(height=700, title_text="Phase Space Analysis")
                
                return dcc.Graph(figure=fig)
            
            elif active_tab == 'parameter-sweep':
                # Parameter sweep interface
                return html.Div([
                    html.H4("Parameter Sweep", style={'color': '#34495e'}),
                    html.P("Select parameter to sweep:"),
                    dcc.Dropdown(
                        id='sweep-parameter',
                        options=[
                            {'label': 'Sigma (σ)', 'value': 'sigma'},
                            {'label': 'Rho (ρ)', 'value': 'rho'},
                            {'label': 'Beta (β)', 'value': 'beta'}
                        ],
                        value='rho',
                        style={'width': '200px', 'marginBottom': '20px'}
                    ),
                    html.Button("Run Parameter Sweep", id='sweep-button',
                               style={'backgroundColor': '#9b59b6', 'color': 'white',
                                     'border': 'none', 'padding': '10px 20px',
                                     'borderRadius': '5px', 'cursor': 'pointer'}),
                    html.Div(id='sweep-results', style={'marginTop': '20px'})
                ])
            
        except Exception as e:
            return html.Div(f"Error loading data: {str(e)}", 
                          style={'color': 'red', 'textAlign': 'center'})
    
    @app.callback(
        Output('system-info', 'children'),
        [Input('simulation-data', 'children')]
    )
    def update_system_info(simulation_data):
        if not simulation_data:
            return "No simulation data available."
        
        try:
            data = json.loads(simulation_data)
            if not data:
                return "No simulation data available."
            
            params = data.get('parameters', {})
            metadata = data.get('metadata', {})
            
            info = [
                f"Parameters: σ={params.get('sigma', 'N/A'):.3f}, ρ={params.get('rho', 'N/A'):.3f}, β={params.get('beta', 'N/A'):.3f}",
                f"Integration method: {metadata.get('integrator', 'N/A')}",
                f"Simulation time: {metadata.get('simulation_time', 'N/A'):.3f} seconds",
                f"Total points: {metadata.get('total_points', 'N/A'):,}",
                f"Effective dt: {metadata.get('effective_dt', 'N/A'):.6f}",
            ]
            
            return html.Pre('\n'.join(info))
            
        except Exception as e:
            return f"Error loading system info: {str(e)}"
    
    # Real-time visualization callbacks
    @app.callback(
        Output('realtime-interval', 'disabled'),
        [Input('start-realtime-button', 'n_clicks'),
         Input('stop-realtime-button', 'n_clicks')]
    )
    def control_realtime(start_clicks, stop_clicks):
        ctx = callback_context
        if not ctx.triggered:
            return True
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if button_id == 'start-realtime-button':
            return False
        elif button_id == 'stop-realtime-button':
            return True
        
        return True
    
    return app


def main():
    """Main function to run the web application."""
    app = create_app()
    app.run_server(debug=True, host='0.0.0.0', port=8050)


if __name__ == '__main__':
    main()