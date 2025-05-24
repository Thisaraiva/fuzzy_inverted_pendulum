# C:\Programacao\Projetos\Python\fuzzy_inverted_pendulum\core\dynamics.py

import numpy as np
from core.config import MC, MP, L, G, I, DT, FORCE_LIMIT

def calculate_accelerations(state, force):
    """Calcula as acelerações linear e angular do pêndulo invertido."""
    x, x_dot, theta, theta_dot = state
    force = np.clip(force, -FORCE_LIMIT, FORCE_LIMIT)  # Limitar a força

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    # Equações dinâmicas baseadas em Lagrange
    denom = MC + MP * sin_theta**2
    if abs(denom) < 1e-6:
        denom = 1e-6  # Evitar divisão por zero

    x_double_dot = (force + MP * L * theta_dot**2 * sin_theta - MP * G * sin_theta * cos_theta) / denom
    theta_double_dot = (G * sin_theta - cos_theta * (force + MP * L * theta_dot**2 * sin_theta) / (MC + MP)) / (L * (1 - MP * cos_theta**2 / (MC + MP)))

    return x_double_dot, theta_double_dot

def dynamics_derivative(state, force):
    """Calcula as derivadas do estado."""
    x, x_dot, theta, theta_dot = state
    x_double_dot, theta_double_dot = calculate_accelerations(state, force)
    return np.array([x_dot, x_double_dot, theta_dot, theta_double_dot])

def simulate_inverted_pendulum(state, force, dt=DT):
    """
    Simula um passo do pêndulo invertido usando Runge-Kutta 4 (RK4).

    Args:
        state (list): Estado atual [x, x_dot, theta, theta_dot].
        force (float): Força aplicada ao carro (N).
        dt (float): Intervalo de tempo (s).

    Returns:
        list: Próximo estado [x_new, x_dot_new, theta_new, theta_dot_new].
    """
    state = np.array(state)

    k1 = dynamics_derivative(state, force)
    k2 = dynamics_derivative(state + 0.5 * dt * k1, force)
    k3 = dynamics_derivative(state + 0.5 * dt * k2, force)
    k4 = dynamics_derivative(state + dt * k3, force)

    state_new = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # Limitar theta para evitar overflow
    state_new[2] = (state_new[2] + np.pi) % (2 * np.pi) - np.pi
    return state_new.tolist()

if __name__ == '__main__':
    initial_state = [0.0, 0.0, np.pi + 0.1, 0.0]
    total_time = 10.0
    n_steps = int(total_time / DT)
    states_history = [initial_state]
    for _ in range(n_steps):
        current_state = states_history[-1]
        next_state = simulate_inverted_pendulum(current_state, 0.0)
        states_history.append(next_state)
    print("Simulação básica executada.")