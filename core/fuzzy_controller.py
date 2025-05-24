# C:\Programacao\Projetos\Python\fuzzy_inverted_pendulum\core\fuzzy_controller.py

import numpy as np
from core.config import FORCE_WEIGHT_PENDULUM, FORCE_WEIGHT_CAR, FORCE_LIMIT

# Funções de pertinência
def triangular_mf(x, a, b, c):
    if a < x <= b:
        return (x - a) / (b - a)
    elif b < x < c:
        return (c - x) / (c - b)
    elif x == b:
        return 1
    return 0

def trapezoidal_mf(x, a, b, c, d):
    if b <= x <= c:
        return 1
    elif a < x < b:
        return (x - a) / (b - a)
    elif c < x < d:
        return (d - x) / (d - c)
    return 0

# Conjuntos fuzzy ajustados para maior sensibilidade
def pendulum_angle_mf(theta):
    N = trapezoidal_mf(theta, -np.pi, -np.pi, -0.3, -0.1)
    Z = triangular_mf(theta, -0.2, 0, 0.2)
    P = trapezoidal_mf(theta, 0.1, 0.3, np.pi, np.pi)
    return {'N': N, 'Z': Z, 'P': P}

def pendulum_angular_velocity_mf(theta_dot):
    N = trapezoidal_mf(theta_dot, -np.pi, -np.pi, -0.6, -0.2)
    Z = triangular_mf(theta_dot, -0.3, 0, 0.3)
    P = trapezoidal_mf(theta_dot, 0.2, 0.6, np.pi, np.pi)
    return {'N': N, 'Z': Z, 'P': P}

def car_position_mf(x):
    N = trapezoidal_mf(x, -np.inf, -np.inf, -2, -0.5)
    Z = triangular_mf(x, -1, 0, 1)
    P = trapezoidal_mf(x, 0.5, 2, np.inf, np.inf)
    return {'N': N, 'Z': Z, 'P': P}

def car_velocity_mf(x_dot):
    N = trapezoidal_mf(x_dot, -np.inf, -np.inf, -1, -0.3)
    Z = triangular_mf(x_dot, -0.5, 0, 0.5)
    P = trapezoidal_mf(x_dot, 0.3, 1, np.inf, np.inf)
    return {'N': N, 'Z': Z, 'P': P}

# Conjuntos fuzzy para a força de saída
def output_force_mf():
    pendulum_force_sets = {
        'NL': {'range': (-np.inf, -60), 'peak': -60},
        'NM': {'range': (-60, -30), 'peak': -30},
        'NS': {'range': (-30, -10), 'peak': -10},
        'Z': {'range': (-10, 10), 'peak': 0},
        'PS': {'range': (10, 30), 'peak': 10},
        'PM': {'range': (30, 60), 'peak': 30},
        'PL': {'range': (60, np.inf), 'peak': 60}
    }
    car_force_sets = {
        'NL': {'range': (-np.inf, -40), 'peak': -40},
        'NM': {'range': (-40, -20), 'peak': -20},
        'NS': {'range': (-20, -5), 'peak': -5},
        'Z': {'range': (-5, 5), 'peak': 0},
        'PS': {'range': (5, 20), 'peak': 5},
        'PM': {'range': (20, 40), 'peak': 20},
        'PL': {'range': (40, np.inf), 'peak': 40}
    }
    return pendulum_force_sets, car_force_sets

pendulum_force_sets, car_force_sets = output_force_mf()

# Regras fuzzy ajustadas para priorizar estabilização
pendulum_rules = [
    {'if': {'theta': 'N', 'theta_dot': 'N'}, 'then': 'NL'},
    {'if': {'theta': 'N', 'theta_dot': 'Z'}, 'then': 'NM'},
    {'if': {'theta': 'N', 'theta_dot': 'P'}, 'then': 'NS'},
    {'if': {'theta': 'Z', 'theta_dot': 'N'}, 'then': 'NS'},
    {'if': {'theta': 'Z', 'theta_dot': 'Z'}, 'then': 'Z'},
    {'if': {'theta': 'Z', 'theta_dot': 'P'}, 'then': 'PS'},
    {'if': {'theta': 'P', 'theta_dot': 'N'}, 'then': 'PS'},
    {'if': {'theta': 'P', 'theta_dot': 'Z'}, 'then': 'PM'},
    {'if': {'theta': 'P', 'theta_dot': 'P'}, 'then': 'PL'}
]

car_rules = [
    {'if': {'x': 'N', 'x_dot': 'N'}, 'then': 'PL'},
    {'if': {'x': 'N', 'x_dot': 'Z'}, 'then': 'PM'},
    {'if': {'x': 'N', 'x_dot': 'P'}, 'then': 'PS'},
    {'if': {'x': 'Z', 'x_dot': 'N'}, 'then': 'PS'},
    {'if': {'x': 'Z', 'x_dot': 'Z'}, 'then': 'Z'},
    {'if': {'x': 'Z', 'x_dot': 'P'}, 'then': 'NS'},
    {'if': {'x': 'P', 'x_dot': 'N'}, 'then': 'NS'},
    {'if': {'x': 'P', 'x_dot': 'Z'}, 'then': 'NM'},
    {'if': {'x': 'P', 'x_dot': 'P'}, 'then': 'NL'}
]

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
    return numerator / denominator if denominator else 0

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
    return numerator / denominator if denominator else 0

def combined_fis_control(state):
    x, x_dot, theta, theta_dot = state
    force_pendulum = fis_pendulum_control(theta - np.pi, theta_dot)
    force_car = fis_car_control(x, x_dot)
    combined_force = FORCE_WEIGHT_PENDULUM * force_pendulum + FORCE_WEIGHT_CAR * force_car
    return np.clip(combined_force, -FORCE_LIMIT, FORCE_LIMIT)

if __name__ == '__main__':
    initial_state = [0.0, 0.0, np.pi + 0.1, 0.0]
    force = combined_fis_control(initial_state)
    print(f"Força de controle inicial: {force:.2f} N")