# C:\Programacao\Projetos\Python\fuzzy_inverted_pendulum\core\config.py

import numpy as np

# Parâmetros do sistema físico
MC = 1.2  # Massa do carro (kg)
MP = 0.1  # Massa do pêndulo (kg)
L = 0.5   # Comprimento do pêndulo (m)
G = 9.81  # Aceleração da gravidade (m/s^2)
I = 0.006 # Momento de inércia do pêndulo (kg.m^2)
DT = 0.01 # Intervalo de tempo (s)

# Parâmetros de simulação
SIMULATION_TIME = 20.0  # Duração da simulação (s)
INITIAL_STATE = [0.0, 0.0, np.pi + 0.1, 0.0]  # [x, x_dot, theta, theta_dot]
FORCE_LIMIT = 50.0  # Limite máximo da força (N) reduzido

# Parâmetros do controlador fuzzy
FORCE_WEIGHT_PENDULUM = 0.8  # Maior peso para o pêndulo
FORCE_WEIGHT_CAR = 0.2       # Menor peso para o carro

# Parâmetros do algoritmo genético
POPULATION_SIZE = 100
NUM_GENERATIONS = 200
MUTATION_RATE = 0.02
NUM_PARENTS = 40
ELITISM_COUNT = 5

# Parâmetros do Neuro-Fuzzy
NUM_EPISODES = 200
MAX_STEPS = 500
EPOCHS = 20
BATCH_SIZE = 64