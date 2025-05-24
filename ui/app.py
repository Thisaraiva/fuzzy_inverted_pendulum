# C:\Programacao\Projetos\Python\fuzzy_inverted_pendulum\ui\app.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import csv
from datetime import datetime
import pickle
import base64
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import numpy as np
from core.dynamics import simulate_inverted_pendulum, L
from core.fuzzy_controller import combined_fis_control, pendulum_rules, car_rules
from core.genetic_fuzzy import genetic_algorithm, get_force_from_chromosome
from core.neuro_fuzzy import NeuroFuzzyMLPController, collect_training_data_with_fis
from core.config import DT, SIMULATION_TIME, INITIAL_STATE, FORCE_LIMIT

app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Simulação inicial para gráficos
n_steps = int(SIMULATION_TIME / DT)
time_points = np.linspace(0, SIMULATION_TIME, n_steps + 1)
states_history_fis = [INITIAL_STATE]
force_history_fis = []
current_state = INITIAL_STATE
for _ in range(n_steps):
    force = combined_fis_control(current_state)
    next_state = simulate_inverted_pendulum(current_state, force)
    states_history_fis.append(next_state)
    force_history_fis.append(force)
    current_state = next_state

x_trajectory_fis = [state[0] for state in states_history_fis]
theta_trajectory_fis = [state[2] for state in states_history_fis]
theta_error_trajectory_fis = [angle - np.pi for angle in theta_trajectory_fis]
x_dot_trajectory_fis = [state[1] for state in states_history_fis]
theta_dot_trajectory_fis = [state[3] for state in states_history_fis]

# Layout da aplicação
app.layout = html.Div(style={'display': 'flex', 'flexDirection': 'column', 'height': '100vh', 'fontFamily': 'Arial, sans-serif'}, children=[
    html.H1("Controle Inteligente do Pêndulo Invertido", style={'textAlign': 'center', 'color': '#2c3e50', 'margin': '20px'}),
    dcc.Store(id='best-chromosome', storage_type='session'),
    dcc.Store(id='neuro-fuzzy-controller', storage_type='session', data=False),
    dcc.Store(id='simulation-progress-store', storage_type='session'),
    dcc.Store(id='comparison-data', storage_type='session', data={'fis': [], 'genetic': [], 'neuro': []}),
    dcc.Interval(id='progress-interval', interval=2000, n_intervals=0, disabled=True),
    html.Div(style={'display': 'flex', 'flexDirection': 'row', 'flexGrow': 1, 'gap': '20px', 'padding': '20px'}, children=[
        html.Div(style={'width': '30%', 'padding': '20px', 'backgroundColor': '#ecf0f1', 'borderRadius': '8px', 'boxShadow': '0 2px 5px rgba(0,0,0,0.1)'}, children=[
            html.H3("Configurações", style={'color': '#34495e', 'marginBottom': '15px'}),
            html.Label("Ângulo Inicial (graus):", style={'display': 'block', 'marginBottom': '5px'}),
            dcc.Slider(id='initial-angle', min=-30, max=30, step=1, value=np.degrees(INITIAL_STATE[2] - np.pi),
                       marks={-30: '-30', 0: '0', 30: '30'}),
            html.Div(id='angle-output', style={'fontSize': '12px', 'color': '#7f8c8d', 'marginBottom': '15px'}),
            html.Label("Velocidade Angular Inicial (rad/s):", style={'display': 'block', 'marginBottom': '5px'}),
            dcc.Slider(id='initial-angular-velocity', min=-2, max=2, step=0.1, value=INITIAL_STATE[3],
                       marks={-2: '-2', 0: '0', 2: '2'}),
            html.Div(id='angular-velocity-output', style={'fontSize': '12px', 'color': '#7f8c8d', 'marginBottom': '15px'}),
            html.Label("Posição Inicial do Carro (m):", style={'display': 'block', 'marginBottom': '5px'}),
            dcc.Slider(id='initial-position', min=-3, max=3, step=0.1, value=INITIAL_STATE[0],
                       marks={-3: '-3', 0: '0', 3: '3'}),
            html.Div(id='position-output', style={'fontSize': '12px', 'color': '#7f8c8d', 'marginBottom': '15px'}),
            html.Label("Velocidade Inicial do Carro (m/s):", style={'display': 'block', 'marginBottom': '5px'}),
            dcc.Slider(id='initial-velocity', min=-1, max=1, step=0.1, value=INITIAL_STATE[1],
                       marks={-1: '-1', 0: '0', 1: '1'}),
            html.Div(id='velocity-output', style={'fontSize': '12px', 'color': '#7f8c8d', 'marginBottom': '20px'}),
            html.H3("Critérios de Estabilização", style={'color': '#34495e', 'marginBottom': '15px'}),
            html.Label("Limiar de Erro Angular (graus):", style={'display': 'block', 'marginBottom': '5px'}),
            dcc.Slider(id='angle-threshold', min=0.5, max=5, step=0.1, value=1.0,
                       marks={0.5: '0.5', 3: '3', 5: '5'}),
            html.Div(id='angle-threshold-output', style={'fontSize': '12px', 'color': '#7f8c8d', 'marginBottom': '15px'}),
            html.Label("Limiar de Velocidade (m/s e rad/s):", style={'display': 'block', 'marginBottom': '5px'}),
            dcc.Slider(id='velocity-threshold', min=0.05, max=0.5, step=0.01, value=0.1,
                       marks={0.05: '0.05', 0.3: '0.3', 0.5: '0.5'}),
            html.Div(id='velocity-threshold-output', style={'fontSize': '12px', 'color': '#7f8c8d', 'marginBottom': '20px'}),
            html.H3("Controlador", style={'color': '#34495e', 'marginBottom': '15px'}),
            dcc.RadioItems(
                id='controller-type',
                options=[
                    {'label': 'FIS (Fuzzy Inference System)', 'value': 'fis'},
                    {'label': 'Genético-Fuzzy', 'value': 'genetic-fuzzy'},
                    {'label': 'Neuro-Fuzzy', 'value': 'neuro-fuzzy'}
                ],
                value='fis',
                labelStyle={'display': 'block', 'marginBottom': '10px', 'color': '#2c3e50'}
            ),
            html.Button("Iniciar Simulação", id='start-simulation', n_clicks=0, style={
                'backgroundColor': '#3498db', 'color': 'white', 'padding': '12px 20px', 'border': 'none',
                'borderRadius': '5px', 'marginTop': '20px', 'cursor': 'pointer', 'width': '100%'
            }),
            html.Button("Treinar Neuro-Fuzzy", id='train-neuro-fuzzy', n_clicks=0, style={
                'backgroundColor': '#2ecc71', 'color': 'white', 'padding': '12px 20px', 'border': 'none',
                'borderRadius': '5px', 'marginTop': '10px', 'cursor': 'pointer', 'width': '100%'
            }),
            html.Div(id='simulation-status', style={'marginTop': '15px', 'fontSize': '14px', 'color': '#27ae60', 'fontWeight': 'bold'}),
            html.Div(id='stabilization-info', style={'marginTop': '10px', 'fontSize': '13px', 'color': '#34495e'}),
            html.H3("Comparação de Desempenho", style={'color': '#34495e', 'marginTop': '20px', 'marginBottom': '15px'}),
            html.Div(id='comparison-table', style={'fontSize': '12px', 'color': '#2c3e50'})
        ]),
        html.Div(style={'width': '70%', 'padding': '20px', 'backgroundColor': '#f5f6fa', 'borderRadius': '8px'}, children=[
            html.H3("Visualização do Sistema", style={'color': '#34495e', 'marginBottom': '15px'}),
            dcc.Graph(id='pendulum-visualization', style={'height': '40vh'}),
            html.Div(style={'display': 'flex', 'flexDirection': 'row', 'gap': '20px', 'marginTop': '20px'}, children=[
                dcc.Graph(id='theta-plot', style={'width': '50%'}),
                dcc.Graph(id='x-plot', style={'width': '50%'})
            ]),
            html.Div(style={'display': 'flex', 'flexDirection': 'row', 'gap': '20px', 'marginTop': '20px'}, children=[
                dcc.Graph(id='theta-dot-plot', style={'width': '50%'}),
                dcc.Graph(id='x-dot-plot', style={'width': '50%'})
            ]),
            dcc.Graph(id='force-plot', style={'marginTop': '20px'})
        ])
    ]),
    html.Footer("Desenvolvido para Controle Inteligente de Pêndulo Invertido", style={
        'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#2c3e50', 'color': '#ecf0f1', 'marginTop': '20px'
    })
])

def log_simulation(controller_type, initial_state, stabilization_time, success, max_time, mean_theta_error, mean_x_error):
    with open('stabilization_log.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        if os.stat('stabilization_log.csv').st_size == 0:
            writer.writerow(['Timestamp', 'Controller', 'Initial_X', 'Initial_X_dot', 'Initial_Theta', 'Initial_Theta_dot', 'Stabilization_Time', 'Success', 'Mean_Theta_Error', 'Mean_X_Error'])
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            controller_type,
            initial_state[0],
            initial_state[1],
            initial_state[2],
            initial_state[3],
            stabilization_time if stabilization_time is not None else max_time,
            'True' if success else 'False',
            mean_theta_error,
            mean_x_error
        ])

@app.callback(
    [Output('angle-output', 'children'),
     Output('angular-velocity-output', 'children'),
     Output('position-output', 'children'),
     Output('velocity-output', 'children'),
     Output('angle-threshold-output', 'children'),
     Output('velocity-threshold-output', 'children')],
    [Input('initial-angle', 'value'),
     Input('initial-angular-velocity', 'value'),
     Input('initial-position', 'value'),
     Input('initial-velocity', 'value'),
     Input('angle-threshold', 'value'),
     Input('velocity-threshold', 'value')]
)
def update_outputs(angle, angular_velocity, position, velocity, angle_threshold, velocity_threshold):
    return (
        f"Ângulo: {angle:.1f}° ({np.radians(angle):.2f} rad)",
        f"Velocidade Angular: {angular_velocity:.2f} rad/s",
        f"Posição: {position:.2f} m",
        f"Velocidade: {velocity:.2f} m/s",
        f"Limiar Angular: {angle_threshold:.2f}° ({np.radians(angle_threshold):.2f} rad)",
        f"Limiar de Velocidade: {velocity_threshold:.2f} m/s e rad/s"
    )

@app.callback(
    [Output('progress-interval', 'disabled'),
     Output('progress-interval', 'n_intervals')],
    Input('start-simulation', 'n_clicks'),
    prevent_initial_call=True
)
def toggle_progress_interval(n_clicks):
    return (False, 0) if n_clicks > 0 else (True, 0)

@app.callback(
    Output('simulation-status', 'children', allow_duplicate=True),
    Input('progress-interval', 'n_intervals'),
    State('simulation-progress-store', 'data'),
    State('start-simulation', 'n_clicks'),
    prevent_initial_call=True
)
def update_progress(n_intervals, progress_data, n_clicks):
    if n_clicks == 0 or not progress_data:
        return "Aguardando ação do usuário..."
    return progress_data.get('message', "Processando...")

@app.callback(
    [Output('pendulum-visualization', 'figure'),
     Output('theta-plot', 'figure'),
     Output('x-plot', 'figure'),
     Output('theta-dot-plot', 'figure'),
     Output('x-dot-plot', 'figure'),
     Output('force-plot', 'figure'),
     Output('simulation-status', 'children'),
     Output('stabilization-info', 'children'),
     Output('best-chromosome', 'data'),
     Output('neuro-fuzzy-controller', 'data'),
     Output('simulation-progress-store', 'data'),
     Output('comparison-table', 'children'),
     Output('comparison-data', 'data')],
    [Input('start-simulation', 'n_clicks'),
     Input('train-neuro-fuzzy', 'n_clicks')],
    [State('initial-angle', 'value'),
     State('initial-angular-velocity', 'value'),
     State('initial-position', 'value'),
     State('initial-velocity', 'value'),
     State('angle-threshold', 'value'),
     State('velocity-threshold', 'value'),
     State('controller-type', 'value'),
     State('best-chromosome', 'data'),
     State('neuro-fuzzy-controller', 'data'),
     State('comparison-data', 'data')],
    prevent_initial_call=True
)
def run_simulation(start_clicks, train_clicks, initial_angle_deg, initial_angular_velocity, initial_position, initial_velocity, angle_threshold, velocity_threshold, controller_type, stored_chromosome, stored_controller, comparison_data):
    angle_threshold_rad = np.radians(angle_threshold)
    initial_theta = np.radians(initial_angle_deg) + np.pi
    initial_state = [initial_position, initial_velocity, initial_theta, initial_angular_velocity]
    states_history = [initial_state]
    force_history = []
    current_state = initial_state
    best_chromosome_data = stored_chromosome
    stabilization_time = None
    success = False
    stable_count = 0
    stabilization_steps = int(1.0 / DT)
    progress_data = {'step': 0, 'total_steps': n_steps, 'message': f"Inicializando simulação com {controller_type.upper()}..."}
    theta_errors = []
    x_errors = []

    if stored_controller:
        try:
            neuro_fuzzy_controller = pickle.loads(base64.b64decode(stored_controller.encode()))
        except:
            neuro_fuzzy_controller = NeuroFuzzyMLPController()
    else:
        neuro_fuzzy_controller = NeuroFuzzyMLPController()

    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'train-neuro-fuzzy' and train_clicks > 0:
        progress_data['message'] = "Iniciando treinamento do Neuro-Fuzzy..."
        neuro_fuzzy_controller = NeuroFuzzyMLPController()
        training_data = collect_training_data_with_fis()
        neuro_fuzzy_controller.train(training_data)
        progress_data['message'] = "Treinamento Neuro-Fuzzy concluído com sucesso!"
        controller_data = base64.b64encode(pickle.dumps(neuro_fuzzy_controller)).decode()
        return (dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, progress_data['message'], dash.no_update, dash.no_update, controller_data, progress_data, dash.no_update, comparison_data)

    if start_clicks == 0:
        raise dash.exceptions.PreventUpdate

    if controller_type == 'genetic-fuzzy' and best_chromosome_data is None:
        progress_data['message'] = "Otimizando controlador Genético-Fuzzy..."
        best_chromosome_data, _ = genetic_algorithm()
        progress_data['message'] = "Otimização Genética concluída. Iniciando simulação..."

    if controller_type == 'neuro-fuzzy' and not neuro_fuzzy_controller.trained:
        progress_data['message'] = "Neuro-Fuzzy não treinado. Iniciando treinamento..."
        training_data = collect_training_data_with_fis()
        neuro_fuzzy_controller.train(training_data)
        progress_data['message'] = "Treinamento Neuro-Fuzzy concluído. Iniciando simulação..."

    progress_data['message'] = f"Executando simulação com controlador {controller_type.upper()}..."
    for step in range(n_steps):
        if controller_type == 'fis':
            force = combined_fis_control(current_state)
        elif controller_type == 'genetic-fuzzy':
            force = get_force_from_chromosome(current_state, best_chromosome_data)
        else:
            force = neuro_fuzzy_controller.predict_force(current_state)
        force = np.clip(force, -FORCE_LIMIT, FORCE_LIMIT)
        next_state = simulate_inverted_pendulum(current_state, force)
        states_history.append(next_state)
        force_history.append(force)
        current_state = next_state

        theta_error = abs(current_state[2] - np.pi)
        x_error = abs(current_state[0])
        theta_errors.append(theta_error)
        x_errors.append(x_error)

        if (theta_error < angle_threshold_rad and
            abs(current_state[3]) < velocity_threshold and
            x_error < angle_threshold_rad and
            abs(current_state[1]) < velocity_threshold):
            stable_count += 1
            if stable_count >= stabilization_steps and stabilization_time is None:
                stabilization_time = step * DT
                success = True
        else:
            stable_count = 0

        if step % 500 == 0:
            progress_data = {'step': step, 'total_steps': n_steps, 'message': f"Passo {step}/{n_steps} ({controller_type.upper()}): Erro angular: {theta_error:.3f} rad, Posição: {x_error:.3f} m"}

        if theta_error > np.pi / 2:
            progress_data['message'] = f"Simulação com {controller_type.upper()} falhou: Pêndulo caiu!"
            break

    mean_theta_error = np.mean(theta_errors)
    mean_x_error = np.mean(x_errors)
    log_simulation(controller_type, initial_state, stabilization_time, success, SIMULATION_TIME, mean_theta_error, mean_x_error)

    result_message = f"Resultado {controller_type.upper()}: {'Estabilizado' if success else 'Não estabilizado'} em {stabilization_time if stabilization_time is not None else SIMULATION_TIME:.2f}s. Erro médio: θ={mean_theta_error:.3f} rad, x={mean_x_error:.3f} m"
    progress_data['message'] = result_message

    x_traj = [state[0] for state in states_history]
    theta_traj = [state[2] for state in states_history]
    theta_error_traj = [angle - np.pi for angle in theta_traj]
    x_dot_traj = [state[1] for state in states_history]
    theta_dot_traj = [state[3] for state in states_history]
    time_points = np.linspace(0, SIMULATION_TIME, len(states_history))

    cart_width = 0.5
    cart_height = 0.2
    cart_x = x_traj[-1]
    pendulum_x = cart_x + L * np.sin(theta_traj[-1])
    pendulum_z = cart_height + L * np.cos(theta_traj[-1])

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
        go.Scatter3d(x=[cart_x, pendulum_x], y=[0, 0], z=[cart_height, pendulum_z], mode='lines+markers', line=dict(color='red', width=4), marker=dict(size=8, color='red'), name='Pêndulo'),
        go.Scatter3d(x=[cart_x], y=[0], z=[cart_height], marker=dict(size=10, color='black'), name='Pivô')
    ], layout=go.Layout(
        scene=dict(
            xaxis=dict(title='X (m)', range=[-5, 5]),
            yaxis=dict(title='Y', range=[-0.5, 0.5]),
            zaxis=dict(title='Z (m)', range=[-0.1, L + cart_height + 0.1]),
            aspectratio=dict(x=1, y=0.1, z=1.5),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1))
        ),
        title='Visualização 3D do Pêndulo Invertido',
        margin=dict(l=0, r=0, t=50, b=0)
    ))

    fig_theta = go.Figure(data=[go.Scatter(x=time_points, y=theta_error_traj, mode='lines', name='Erro Angular', line=dict(color='#e74c3c'))])
    if success and stabilization_time is not None:
        fig_theta.add_vline(x=stabilization_time, line_dash="dash", line_color="#2ecc71", annotation_text="Estabilizado", annotation_position="top left")
    fig_theta.update_layout(title='Erro Angular do Pêndulo', xaxis_title='Tempo (s)', yaxis_title='Erro (rad)', template='plotly_white')

    fig_x = go.Figure(data=[go.Scatter(x=time_points, y=x_traj, mode='lines', name='Posição do Carro', line=dict(color='#3498db'))])
    fig_x.update_layout(title='Posição do Carro', xaxis_title='Tempo (s)', yaxis_title='Posição (m)', template='plotly_white')

    fig_theta_dot = go.Figure(data=[go.Scatter(x=time_points, y=theta_dot_traj, mode='lines', name='Velocidade Angular', line=dict(color='#9b59b6'))])
    fig_theta_dot.update_layout(title='Velocidade Angular do Pêndulo', xaxis_title='Tempo (s)', yaxis_title='Vel. (rad/s)', template='plotly_white')

    fig_x_dot = go.Figure(data=[go.Scatter(x=time_points, y=x_dot_traj, mode='lines', name='Velocidade do Carro', line=dict(color='#f1c40f'))])
    fig_x_dot.update_layout(title='Velocidade do Carro', xaxis_title='Tempo (s)', yaxis_title='Vel. (m/s)', template='plotly_white')

    fig_force = go.Figure(data=[go.Scatter(x=time_points, y=force_history, mode='lines', name='Força de Controle', line=dict(color='#e67e22'))])
    fig_force.update_layout(title='Força de Controle Aplicada', xaxis_title='Tempo (s)', yaxis_title='Força (N)', template='plotly_white')

    stabilization_info = (
        f"Estado Final: x={current_state[0]:.2f} m, x_dot={current_state[1]:.2f} m/s, "
        f"θ={(current_state[2] - np.pi) * 180 / np.pi:.2f}°, θ_dot={current_state[3]:.2f} rad/s\n"
        f"Erro Médio: θ={mean_theta_error:.3f} rad, x={mean_x_error:.3f} m"
    )

    # Atualizar dados de comparação
    comparison_data = comparison_data or {'fis': [], 'genetic': [], 'neuro': []}
    controller_key = {'fis': 'fis', 'genetic-fuzzy': 'genetic', 'neuro-fuzzy': 'neuro'}[controller_type]
    comparison_data[controller_key].append({
        'time': stabilization_time if stabilization_time is not None else SIMULATION_TIME,
        'success': success,
        'theta_error': mean_theta_error,
        'x_error': mean_x_error
    })

    # Gerar tabela de comparação
    table_header = [html.Th(col) for col in ['Controlador', 'Taxa de Sucesso', 'Tempo Médio (s)', 'Erro Angular Médio (rad)', 'Erro de Posição Médio (m)']]
    table_rows = []
    for ctrl, data in comparison_data.items():
        if data:
            success_rate = sum(1 for d in data if d['success']) / len(data)
            avg_time = np.mean([d['time'] for d in data])
            avg_theta_error = np.mean([d['theta_error'] for d in data])
            avg_x_error = np.mean([d['x_error'] for d in data])
            table_rows.append(html.Tr([
                html.Td(ctrl.upper()),
                html.Td(f"{success_rate:.2%}"),
                html.Td(f"{avg_time:.2f}"),
                html.Td(f"{avg_theta_error:.3f}"),
                html.Td(f"{avg_x_error:.3f}")
            ]))
    comparison_table = html.Table([html.Tr(table_header)] + table_rows, style={'width': '100%', 'textAlign': 'center', 'borderCollapse': 'collapse', 'marginTop': '10px'})

    controller_data = base64.b64encode(pickle.dumps(neuro_fuzzy_controller)).decode()
    return fig_3d, fig_theta, fig_x, fig_theta_dot, fig_x_dot, fig_force, progress_data['message'], stabilization_info, best_chromosome_data, controller_data, progress_data, comparison_table, comparison_data

if __name__ == '__main__':
    app.run(debug=True)