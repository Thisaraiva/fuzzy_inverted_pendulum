# Dentro de core/fuzzy_controller.py

import numpy as np

# --- Funções de Pertinência ---
def triangular_mf(x, a, b, c):
    if a < x <= b:
        return (x - a) / (b - a)
    elif b < x < c:
        return (c - x) / (c - b)
    elif x == b:
        return 1
    else:
        return 0

def trapezoidal_mf(x, a, b, c, d):
    if b <= x <= c:
        return 1
    elif a < x < b:
        return (x - a) / (b - a)
    elif c < x < d:
        return (d - x) / (d - c)
    else:
        return 0

# --- Conjuntos Fuzzy para o Pêndulo ---
def pendulum_angle_mf(theta):
    N = trapezoidal_mf(theta, -0.15, -0.1, -0.03, -0.03)
    Z = triangular_mf(theta, -0.03, 0, 0.03)
    P = trapezoidal_mf(theta, 0.03, 0.03, 0.1, 0.15)
    return {'N': N, 'Z': Z, 'P': P}

def pendulum_angular_velocity_mf(theta_dot):
    N = trapezoidal_mf(theta_dot, -0.15, -0.1, -0.03, -0.03)
    Z = triangular_mf(theta_dot, -0.03, 0, 0.03)
    P = trapezoidal_mf(theta_dot, 0.03, 0.03, 0.1, 0.15)
    return {'N': N, 'Z': Z, 'P': P}

# --- Conjuntos Fuzzy para o Carro ---
def car_position_mf(x):
    N = trapezoidal_mf(x, -np.inf, -np.inf, -2, 0)
    Z = triangular_mf(x, -1.5, 0, 1.5)
    P = trapezoidal_mf(x, 0, 2, np.inf, np.inf)
    return {'N': N, 'Z': Z, 'P': P}

def car_velocity_mf(x_dot):
    N = trapezoidal_mf(x_dot, -np.inf, -np.inf, -2, 0)
    Z = triangular_mf(x_dot, -1.5, 0, 1.5)
    P = trapezoidal_mf(x_dot, 0, 2, np.inf, np.inf)
    return {'N': N, 'Z': Z, 'P': P}

# --- Conjuntos Fuzzy para a Força de Saída ---
def output_force_mf():
    # Pêndulo
    NL_p = {'range': (-np.inf, -100), 'peak': -100}
    NM_p = {'range': (-100, -40), 'peak': -40}
    NS_p = {'range': (-40, -5), 'peak': -5}
    Z_p = {'range': (-5, 5), 'peak': 0}
    PS_p = {'range': (5, 40), 'peak': 5}
    PM_p = {'range': (40, 100), 'peak': 40}
    PL_p = {'range': (100, np.inf), 'peak': 100}
    pendulum_force_sets = {'NL': NL_p, 'NM': NM_p, 'NS': NS_p, 'Z': Z_p, 'PS': PS_p, 'PM': PM_p, 'PL': PL_p}

    # Carro
    NL_c = {'range': (-np.inf, -50), 'peak': -50}
    NM_c = {'range': (-50, -10), 'peak': -10}
    NS_c = {'range': (-10, -1), 'peak': -1}
    Z_c = {'range': (-1, 1), 'peak': 0}
    PS_c = {'range': (1, 10), 'peak': 1}
    PM_c = {'range': (10, 50), 'peak': 10}
    PL_c = {'range': (50, np.inf), 'peak': 50}
    car_force_sets = {'NL': NL_c, 'NM': NM_c, 'NS': NS_c, 'Z': Z_c, 'PS': PS_c, 'PM': PM_c, 'PL': PL_c}

    return pendulum_force_sets, car_force_sets

pendulum_force_sets, car_force_sets = output_force_mf()

# --- Base de Regras Fuzzy ---
pendulum_rules = [
    {'if': {'theta': 'N', 'theta_dot': 'N'}, 'then': 'NL'},
    {'if': {'theta': 'N', 'theta_dot': 'Z'}, 'then': 'NS'},
    {'if': {'theta': 'N', 'theta_dot': 'P'}, 'then': 'Z'},
    {'if': {'theta': 'Z', 'theta_dot': 'N'}, 'then': 'NM'},
    {'if': {'theta': 'Z', 'theta_dot': 'Z'}, 'then': 'Z'},
    {'if': {'theta': 'Z', 'theta_dot': 'P'}, 'then': 'PM'},
    {'if': {'theta': 'P', 'theta_dot': 'N'}, 'then': 'Z'},
    {'if': {'theta': 'P', 'theta_dot': 'Z'}, 'then': 'PS'},
    {'if': {'theta': 'P', 'theta_dot': 'P'}, 'then': 'PL'}
]

car_rules = [
    {'if': {'x': 'N', 'x_dot': 'N'}, 'then': 'PL'},
    {'if': {'x': 'N', 'x_dot': 'Z'}, 'then': 'PM'},
    {'if': {'x': 'N', 'x_dot': 'P'}, 'then': 'Z'},
    {'if': {'x': 'Z', 'x_dot': 'N'}, 'then': 'PS'},
    {'if': {'x': 'Z', 'x_dot': 'Z'}, 'then': 'Z'},
    {'if': {'x': 'Z', 'x_dot': 'P'}, 'then': 'NS'},
    {'if': {'x': 'P', 'x_dot': 'N'}, 'then': 'Z'},
    {'if': {'x': 'P', 'x_dot': 'Z'}, 'then': 'NM'},
    {'if': {'x': 'P', 'x_dot': 'P'}, 'then': 'NL'}
]

# --- Mapeamento de índice para nome da força ---
# Usar apenas 7 índices (0 a 6) para consistência com genetic_fuzzy.py
index_to_force = {
    0: 'NL',
    1: 'NM',
    2: 'NS',
    3: 'Z',
    4: 'PS',
    5: 'PM',
    6: 'PL'
}

def fis_pendulum_control(theta, theta_dot):
    theta_mfs = pendulum_angle_mf(theta)
    theta_dot_mfs = pendulum_angular_velocity_mf(theta_dot)
    activated_rules = []
    for rule in pendulum_rules:
        theta_degree = theta_mfs[rule['if']['theta']]
        theta_dot_degree = theta_dot_mfs[rule['if']['theta_dot']]
        activation_degree = min(theta_degree, theta_dot_degree)
        if activation_degree > 0:
            activated_rules.append({'force': rule['then'], 'degree': activation_degree})

    numerator = 0
    denominator = 0
    for rule in activated_rules:
        force_set = pendulum_force_sets[rule['force']]
        numerator += rule['degree'] * force_set['peak']
        denominator += rule['degree']
    if denominator == 0:
        return 0
    return numerator / denominator

def fis_car_control(x, x_dot):
    x_mfs = car_position_mf(x)
    x_dot_mfs = car_velocity_mf(x_dot)
    activated_rules = []
    for rule in car_rules:
        x_degree = x_mfs[rule['if']['x']]
        x_dot_degree = x_dot_mfs[rule['if']['x_dot']]
        activation_degree = min(x_degree, x_dot_degree)
        if activation_degree > 0:
            activated_rules.append({'force': rule['then'], 'degree': activation_degree})

    numerator = 0
    denominator = 0
    for rule in activated_rules:
        force_set = car_force_sets[rule['force']]
        numerator += rule['degree'] * force_set['peak']
        denominator += rule['degree']
    if denominator == 0:
        return 0
    return numerator / denominator

def combined_fis_control(state):
    x, x_dot, theta, theta_dot = state
    force_pendulum = fis_pendulum_control(theta, theta_dot)
    force_car = fis_car_control(x, x_dot)
    # Uma forma simples de combinar as forças
    return force_pendulum + 0.5 * force_car

if __name__ == '__main__':
    # Exemplo de uso do controlador (pode ser removido ou adaptado para testes)
    initial_state = [0.0, 0.0, np.pi + 0.1, 0.0]
    force = combined_fis_control(initial_state)
    print(f"Força de controle inicial: {force}")