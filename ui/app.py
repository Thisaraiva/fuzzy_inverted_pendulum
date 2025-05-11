import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
import base64

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import numpy as np
from core.dynamics import simulate_inverted_pendulum, dt, l
from core.fuzzy_controller import combined_fis_control, pendulum_rules, car_rules, pendulum_angle_mf, pendulum_angular_velocity_mf, car_position_mf, car_velocity_mf, pendulum_force_sets, car_force_sets, index_to_force
from core.genetic_fuzzy import genetic_algorithm, get_force_from_chromosome
from core.neuro_fuzzy import NeuroFuzzyMLPController, collect_training_data_with_fis

# Variável global para armazenar o controlador Neuro-Fuzzy
neuro_fuzzy_controller = None

# Verificar o número de regras
print(f"App.py - Total de regras: {len(pendulum_rules) + len(car_rules)} (pendulum_rules: {len(pendulum_rules)}, car_rules: {len(car_rules)})")

# Condições iniciais
initial_state = [0.0, 0.0, np.pi + 0.1, 0.0]
simulation_time = 20.0
n_steps = int(simulation_time / dt)
time_points = np.linspace(0, simulation_time, n_steps + 1)

# Simulação com o controlador FIS (para os gráficos iniciais)
states_history_fis = [initial_state]
force_history_fis = []
current_state = initial_state
for _ in range(n_steps):
    force = combined_fis_control(current_state)
    next_state = simulate_inverted_pendulum(current_state, force, dt)
    states_history_fis.append(next_state)
    force_history_fis.append(force)
    current_state = next_state

x_trajectory_fis = [state[0] for state in states_history_fis]
theta_trajectory_fis = [state[2] for state in states_history_fis]
theta_error_trajectory_fis = [angle - np.pi for angle in theta_trajectory_fis]
x_dot_trajectory_fis = [state[1] for state in states_history_fis]
theta_dot_trajectory_fis = [state[3] for state in states_history_fis]

# Inicializar a aplicação Dash
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Layout da aplicação
app.layout = html.Div(style={'display': 'flex', 'flexDirection': 'column', 'height': '100vh', 'fontFamily': 'Segoe UI, Tahoma, Geneva, Verdana, sans-serif'}, children=[
    html.H1("Controle Inteligente do Pêndulo Invertido", style={'textAlign': 'center', 'color': '#333', 'marginBottom': '20px'}),
    dcc.Store(id='best-chromosome', storage_type='session'),
    dcc.Store(id='neuro-fuzzy-controller', storage_type='session', data=False),  # Armazenar apenas se foi treinado
    html.Div(style={'display': 'flex', 'flexDirection': 'row', 'flexGrow': 1}, children=[
        html.Div(style={'width': '30%', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRight': '1px solid #ddd'}, children=[
            html.H3("Controles Iniciais", style={'color': '#555', 'marginBottom': '15px'}),
            html.Div([
                html.Label("Ângulo Inicial (graus):", style={'display': 'block', 'marginBottom': '5px'}),
                dcc.Slider(id='initial-angle', min=-180, max=180, step=1, value=np.degrees(initial_state[2] - np.pi),
                           marks={-180: '-180', -90: '-90', 0: '0', 90: '90', 180: '180'}),
                html.Div(id='angle-output', style={'fontSize': '12px', 'color': '#777', 'marginBottom': '10px'})
            ]),
            html.Div([
                html.Label("Velocidade Angular Inicial (rad/s):", style={'display': 'block', 'marginBottom': '5px'}),
                dcc.Slider(id='initial-angular-velocity', min=-5, max=5, step=0.1, value=initial_state[3],
                           marks={-5: '-5', 0: '0', 5: '5'}),
                html.Div(id='angular-velocity-output', style={'fontSize': '12px', 'color': '#777', 'marginBottom': '10px'})
            ]),
            html.Div([
                html.Label("Posição Inicial do Carro (m):", style={'display': 'block', 'marginBottom': '5px'}),
                dcc.Slider(id='initial-position', min=-5, max=5, step=0.1, value=initial_state[0],
                           marks={-5: '-5', 0: '0', 5: '5'}),
                html.Div(id='position-output', style={'fontSize': '12px', 'color': '#777', 'marginBottom': '10px'})
            ]),
            html.Div([
                html.Label("Velocidade Inicial do Carro (m/s):", style={'display': 'block', 'marginBottom': '5px'}),
                dcc.Slider(id='initial-velocity', min=-2, max=2, step=0.1, value=initial_state[1],
                           marks={-2: '-2', 0: '0', 2: '2'}),
                html.Div(id='velocity-output', style={'fontSize': '12px', 'color': '#777', 'marginBottom': '20px'})
            ]),
            html.H3("Controle", style={'color': '#555', 'marginBottom': '15px'}),
            dcc.RadioItems(
                id='controller-type',
                options=[
                    {'label': 'FIS (Fuzzy Inference System)', 'value': 'fis'},
                    {'label': 'Genético-Fuzzy', 'value': 'genetic-fuzzy'},
                    {'label': 'Neuro-Fuzzy', 'value': 'neuro-fuzzy'}
                ],
                value='fis',
                labelStyle={'display': 'block', 'marginBottom': '5px'}
            ),
            html.Button("Iniciar Simulação", id='start-simulation', n_clicks=0, style={'backgroundColor': '#007bff', 'color': 'white', 'padding': '10px 15px', 'border': 'none', 'borderRadius': '5px', 'marginTop': '20px', 'cursor': 'pointer'}),
            html.Button("Treinar Neuro-Fuzzy", id='train-neuro-fuzzy', n_clicks=0, style={'backgroundColor': '#28a745', 'color': 'white', 'padding': '10px 15px', 'border': 'none', 'borderRadius': '5px', 'marginTop': '10px', 'cursor': 'pointer'}),
            html.Div(id='simulation-status', style={'marginTop': '10px', 'fontSize': '14px', 'color': '#28a745'})
        ]),
        html.Div(style={'width': '70%', 'padding': '20px'}, children=[
            html.H3("Visualização 3D do Sistema", style={'color': '#555', 'marginBottom': '15px'}),
            dcc.Graph(id='pendulum-visualization', figure={'layout': {'scene': {'aspectratio': {'x': 1, 'y': 1, 'z': 1}}}}),
            html.Div(style={'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'space-around', 'marginTop': '20px'}, children=[
                dcc.Graph(id='theta-plot'),
                dcc.Graph(id='x-plot')
            ]),
            html.Div(style={'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'space-around'}, children=[
                dcc.Graph(id='theta-dot-plot'),
                dcc.Graph(id='x-dot-plot')
            ]),
            dcc.Graph(id='force-plot')
        ])
    ]),
    html.Footer("Desenvolvimento em Python para Controle Inteligente", style={'textAlign': 'center', 'marginTop': '30px', 'padding': '10px', 'backgroundColor': '#f0f0f0', 'color': '#777'})
])

# Callbacks para interatividade
@app.callback(
    Output('angle-output', 'children'),
    Input('initial-angle', 'value'))
def update_angle_output(value):
    return f"Valor: {value} graus ({np.radians(value):.2f} rad)"

@app.callback(
    Output('angular-velocity-output', 'children'),
    Input('initial-angular-velocity', 'value'))
def update_angular_velocity_output(value):
    return f"Valor: {value:.2f} rad/s"

@app.callback(
    Output('position-output', 'children'),
    Input('initial-position', 'value'))
def update_position_output(value):
    return f"Valor: {value:.2f} m"

@app.callback(
    Output('velocity-output', 'children'),
    Input('initial-velocity', 'value'))
def update_velocity_output(value):
    return f"Valor: {value:.2f} m/s"

@app.callback(
    [Output('pendulum-visualization', 'figure'),
     Output('theta-plot', 'figure'),
     Output('x-plot', 'figure'),
     Output('theta-dot-plot', 'figure'),
     Output('x-dot-plot', 'figure'),
     Output('force-plot', 'figure'),
     Output('simulation-status', 'children'),
     Output('best-chromosome', 'data'),
     Output('neuro-fuzzy-controller', 'data')],
    [Input('start-simulation', 'n_clicks'),
     Input('train-neuro-fuzzy', 'n_clicks')],
    [State('initial-angle', 'value'),
     State('initial-angular-velocity', 'value'),
     State('initial-position', 'value'),
     State('initial-velocity', 'value'),
     State('controller-type', 'value'),
     State('best-chromosome', 'data'),
     State('neuro-fuzzy-controller', 'data')],
    prevent_initial_call=True
)
def run_simulation(start_clicks, train_clicks, initial_angle_deg, initial_angular_velocity, initial_position, initial_velocity, controller_type, stored_chromosome, stored_controller):
    initial_theta = np.radians(initial_angle_deg) + np.pi
    initial_state = [initial_position, initial_velocity, initial_theta, initial_angular_velocity]
    states_history = [initial_state]
    force_history = []
    current_state = initial_state
    best_chromosome_data = stored_chromosome
    
    # Desserializar o controlador se existir
    if stored_controller:
        try:
            neuro_fuzzy_controller = pickle.loads(base64.b64decode(stored_controller.encode()))
        except:
            neuro_fuzzy_controller = NeuroFuzzyMLPController()
    else:
        neuro_fuzzy_controller = NeuroFuzzyMLPController()
    
    simulation_message = ""

    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'train-neuro-fuzzy' and train_clicks > 0:
        simulation_message = "Treinando Neuro-Fuzzy..."
        neuro_fuzzy_controller = NeuroFuzzyMLPController()
        training_data = collect_training_data_with_fis(num_episodes=50, max_steps=200)
        neuro_fuzzy_controller.train(training_data, epochs=10)
        simulation_message = "Treinamento Neuro-Fuzzy concluído."
        return (dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, simulation_message, dash.no_update, True)

    if start_clicks == 0:
        raise dash.exceptions.PreventUpdate

    if controller_type == 'genetic-fuzzy':
        if best_chromosome_data is None:
            simulation_message = "Otimizando o controlador Genético-Fuzzy..."
            best_chromosome_result, _ = genetic_algorithm(population_size=30, num_generations=50)
            best_chromosome_data = best_chromosome_result
            simulation_message = "Otimização Genética Concluída. Iniciando simulação."
        else:
            simulation_message = "Usando o controlador Genético-Fuzzy otimizado."

    if controller_type == 'neuro-fuzzy' and not neuro_fuzzy_controller:
        simulation_message = "Neuro-Fuzzy não treinado. Clique em 'Treinar Neuro-Fuzzy' primeiro."
        return (dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, simulation_message, best_chromosome_data, neuro_fuzzy_controller)
    
    # No callback principal, adicione:
    if controller_type == 'neuro-fuzzy' and not neuro_fuzzy_controller.trained:
        simulation_message = "Neuro-Fuzzy não treinado. Treinando agora..."
        training_data = collect_training_data_with_fis(num_episodes=50)
        neuro_fuzzy_controller.train(training_data, epochs=10)
        simulation_message = "Neuro-Fuzzy treinado. Iniciando simulação."

    for _ in range(n_steps):
        if controller_type == 'fis':
            force = combined_fis_control(current_state)
        elif controller_type == 'genetic-fuzzy' and best_chromosome_data is not None:
            force = get_force_from_chromosome(current_state, best_chromosome_data)
        elif controller_type == 'neuro-fuzzy' and neuro_fuzzy_controller:
            force = neuro_fuzzy_controller.predict_force(current_state)
        else:
            force = 0
        next_state = simulate_inverted_pendulum(current_state, force, dt)
        states_history.append(next_state)
        force_history.append(force)
        current_state = next_state
        if abs(current_state[2] - np.pi) > np.pi / 2:
            simulation_message = "Pêndulo caiu!"
            break

    x_traj = [state[0] for state in states_history]
    theta_traj = [state[2] for state in states_history]
    theta_error_traj = [angle - np.pi for angle in theta_traj]
    x_dot_traj = [state[1] for state in states_history]
    theta_dot_traj = [state[3] for state in states_history]

    cart_width = 0.5
    cart_height = 0.2
    pendulum_length = l

    cart_x = x_traj[-1]
    pendulum_x = cart_x + pendulum_length * np.sin(theta_traj[-1])
    pendulum_y = pendulum_length * np.cos(theta_traj[-1])

    cart_x_min = cart_x - cart_width / 2
    cart_x_max = cart_x + cart_width / 2
    cart_y_min = -0.05
    cart_y_max = 0.05
    cart_z_min = 0
    cart_z_max = cart_height

    vertices = [
        [cart_x_min, cart_y_min, cart_z_min], [cart_x_max, cart_y_min, cart_z_min],
        [cart_x_max, cart_y_max, cart_z_min], [cart_x_min, cart_y_max, cart_z_min],
        [cart_x_min, cart_y_min, cart_z_max], [cart_x_max, cart_y_min, cart_z_max],
        [cart_x_max, cart_y_max, cart_z_max], [cart_x_min, cart_y_max, cart_z_max]
    ]

    i = [0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7]
    j = [1, 2, 2, 3, 3, 0, 0, 4, 5, 6, 6, 7, 7, 4, 4, 0, 4, 5, 5, 6, 6, 7, 7, 4]
    k = [3, 0, 0, 1, 1, 2, 2, 3, 7, 4, 4, 5, 5, 6, 6, 7, 1, 0, 2, 1, 3, 2, 0, 3]

    fig_3d = go.Figure(data=[
        go.Mesh3d(x=[v[0] for v in vertices], y=[v[1] for v in vertices], z=[v[2] for v in vertices], i=i, j=j, k=k, color='blue', opacity=0.8, name='Carro'),
        go.Scatter3d(x=[cart_x, pendulum_x], y=[0, 0], z=[cart_height, cart_height + pendulum_y], mode='lines+markers', line=dict(color='red', width=3), marker=dict(size=8, color='red'), name='Pêndulo'),
        go.Scatter3d(x=[cart_x], y=[0], z=[cart_height], marker=dict(size=10, color='black'), name='Pivô')
    ], layout=go.Layout(
        scene=dict(xaxis=dict(title='X (m)'), yaxis=dict(title='Y'), zaxis=dict(title='Z (m)'), aspectratio=dict(x=1, y=0.1, z=1.5), camera=dict(eye=dict(x=1.5, y=-1.5, z=1))),
        title='Visualização do Pêndulo Invertido'
    ))

    fig_theta = go.Figure(data=[go.Scatter(x=time_points[:len(theta_error_traj)], y=theta_error_traj, mode='lines', name='Erro Ângulo')])
    fig_theta.update_layout(title='Erro do Ângulo do Pêndulo', xaxis_title='Tempo (s)', yaxis_title='Ângulo (rad)')

    fig_x = go.Figure(data=[go.Scatter(x=time_points[:len(x_traj)], y=x_traj, mode='lines', name='Posição do Carro')])
    fig_x.update_layout(title='Posição do Carro', xaxis_title='Tempo (s)', yaxis_title='Posição (m)')

    fig_theta_dot = go.Figure(data=[go.Scatter(x=time_points[:len(theta_dot_traj)], y=theta_dot_traj, mode='lines', name='Velocidade Angular')])
    fig_theta_dot.update_layout(title='Velocidade Angular do Pêndulo', xaxis_title='Tempo (s)', yaxis_title='Velocidade (rad/s)')

    fig_x_dot = go.Figure(data=[go.Scatter(x=time_points[:len(x_dot_traj)], y=x_dot_traj, mode='lines', name='Velocidade do Carro')])
    fig_x_dot.update_layout(title='Velocidade do Carro', xaxis_title='Tempo (s)', yaxis_title='Velocidade (m/s)')

    fig_force = go.Figure(data=[go.Scatter(x=time_points[:len(force_history)], y=force_history, mode='lines', name='Força de Controle')])
    fig_force.update_layout(title='Força de Controle', xaxis_title='Tempo (s)', yaxis_title='Força (N)')

    controller_data = base64.b64encode(pickle.dumps(neuro_fuzzy_controller)).decode()
    
    return fig_3d, fig_theta, fig_x, fig_theta_dot, fig_x_dot, fig_force, simulation_message, best_chromosome_data, controller_data

if __name__ == '__main__':
    app.run(debug=True)