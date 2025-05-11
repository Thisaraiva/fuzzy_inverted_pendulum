import numpy as np

# Parâmetros do sistema (declarados no nível do módulo)
mc = 1.0  # Massa do carro (kg)
mp = 0.1  # Massa do pêndulo (kg)
l = 0.5   # Comprimento do pêndulo (m)
g = 9.8   # Aceleração da gravidade (m/s^2)
I = 0.006 # Momento de inércia do pêndulo (kg.m^2)
dt = 0.02 # Intervalo de tempo (s) - movido para o nível do módulo

def calculate_accelerations(state, force):
    """Calcula as acelerações linear e angular do sistema."""
    x, x_dot, theta, theta_dot = state

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    denominator = (mc + mp) * (I + mp * l**2) - (mp * l * cos_theta)**2

    if denominator == 0:
        raise ZeroDivisionError("Denominator is zero, check system parameters or state.")

    # Calcular a aceleração angular (theta_double_dot)
    numerator_theta = (mp * l * cos_theta * (mp * l * theta_dot**2 * sin_theta + force) +
                       (mc + mp) * (mp * l * g * sin_theta))
    theta_double_dot = numerator_theta / denominator

    # Calcular a aceleração linear (x_double_dot)
    numerator_x = (mp * l * cos_theta * (mp * l * g * sin_theta) -
                   (I + mp * l**2) * (mp * l * theta_dot**2 * sin_theta + force))
    x_double_dot = numerator_x / -denominator # Multiplicamos por -1 para ajustar o sinal devido à ordem dos termos no denominador

    return x_double_dot, theta_double_dot

def simulate_inverted_pendulum(state, force, dt):
    """
    Simula um passo do pêndulo invertido sobre o carro usando Euler integration.

    Args:
        state (list): Lista contendo o estado atual [x, x_dot, theta, theta_dot].
        force (float): Força aplicada ao carro (N).
        dt (float): Intervalo de tempo para a simulação (s).

    Returns:
        list: O próximo estado do sistema [x_new, x_dot_new, theta_new, theta_dot_new].
    """
    x, x_dot, theta, theta_dot = state

    x_double_dot, theta_double_dot = calculate_accelerations(state, force)

    # Integrar para obter o próximo estado (método de Euler)
    x_dot_new = x_dot + dt * x_double_dot
    x_new = x + dt * x_dot_new
    theta_dot_new = theta_dot + dt * theta_double_dot
    theta_new = theta + dt * theta_dot_new

    return [x_new, x_dot_new, theta_new, theta_dot_new]

if __name__ == '__main__':
    # Exemplo de simulação (pode ser removido ou adaptado para testes)
    initial_state = [0.0, 0.0, np.pi + 0.1, 0.0]
    total_time = 10.0
    n_steps = int(total_time / dt)
    states_history = [initial_state]
    for _ in range(n_steps):
        current_state = states_history[-1]
        next_state = simulate_inverted_pendulum(current_state, 0.0, dt)
        states_history.append(next_state)
    print("Simulação básica executada.")