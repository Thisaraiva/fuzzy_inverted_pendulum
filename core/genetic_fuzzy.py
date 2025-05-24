# C:\Programacao\Projetos\Python\fuzzy_inverted_pendulum\core\genetic_fuzzy.py

import numpy as np
from core.dynamics import simulate_inverted_pendulum
from core.fuzzy_controller import pendulum_angle_mf, pendulum_angular_velocity_mf, car_position_mf, car_velocity_mf, pendulum_rules, car_rules, pendulum_force_sets, car_force_sets, index_to_force
from core.config import DT, POPULATION_SIZE, NUM_GENERATIONS, MUTATION_RATE, NUM_PARENTS, ELITISM_COUNT, SIMULATION_TIME, FORCE_LIMIT

all_rules = pendulum_rules + car_rules
num_rules = len(all_rules)
num_output_options = len(index_to_force)

def get_force_from_chromosome(state, chromosome):
    if len(chromosome) != num_rules:
        print(f"Erro: Cromossomo tem tamanho {len(chromosome)}, esperado {num_rules}")
        return 0
    theta_mfs = pendulum_angle_mf(state[2] - np.pi)
    theta_dot_mfs = pendulum_angular_velocity_mf(state[3])
    x_mfs = car_position_mf(state[0])
    x_dot_mfs = car_velocity_mf(state[1])

    activated_rules = []
    for i, rule in enumerate(pendulum_rules):
        theta_degree = theta_mfs[rule['if']['theta']]
        theta_dot_degree = theta_dot_mfs[rule['if']['theta_dot']]
        activation_degree = min(theta_degree, theta_dot_degree)
        if activation_degree > 0:
            force_index = chromosome[i]
            output_force_name = index_to_force.get(force_index, 'Z')
            activated_rules.append({'force': output_force_name, 'degree': activation_degree, 'type': 'pendulum'})

    for i, rule in enumerate(car_rules):
        idx = len(pendulum_rules) + i
        x_degree = x_mfs[rule['if']['x']]
        x_dot_degree = x_dot_mfs[rule['if']['x_dot']]
        activation_degree = min(x_degree, x_dot_degree)
        if activation_degree > 0:
            force_index = chromosome[idx]
            output_force_name = index_to_force.get(force_index, 'Z')
            activated_rules.append({'force': output_force_name, 'degree': activation_degree, 'type': 'car'})

    pendulum_force_numerator = 0
    pendulum_force_denominator = 0
    car_force_numerator = 0
    car_force_denominator = 0

    for activated_rule in activated_rules:
        if activated_rule['type'] == 'pendulum':
            force_set = pendulum_force_sets[activated_rule['force']]
            pendulum_force_numerator += activated_rule['degree'] * force_set['peak']
            pendulum_force_denominator += activated_rule['degree']
        else:
            force_set = car_force_sets[activated_rule['force']]
            car_force_numerator += activated_rule['degree'] * force_set['peak']
            car_force_denominator += activated_rule['degree']

    force_pendulum = pendulum_force_numerator / pendulum_force_denominator if pendulum_force_denominator else 0
    force_car = car_force_numerator / car_force_denominator if car_force_denominator else 0
    combined_force = 0.7 * force_pendulum + 0.3 * force_car
    return np.clip(combined_force, -FORCE_LIMIT, FORCE_LIMIT)

def evaluate_chromosome(chromosome, num_steps=int(SIMULATION_TIME / DT), initial_state=[0.0, 0.0, np.pi + 0.1, 0.0]):
    current_state = initial_state[:]
    total_reward = 0
    fell = False

    for i in range(num_steps):
        force = get_force_from_chromosome(current_state, chromosome)
        current_state = simulate_inverted_pendulum(current_state, force)

        theta_error = abs(current_state[2] - np.pi)
        x_error = abs(current_state[0])
        theta_dot = abs(current_state[3])
        x_dot = abs(current_state[1])

        # Recompensa mais agressiva para estabilização
        reward = 1000 * np.exp(-10 * theta_error**2 - 5 * x_error**2 - theta_dot**2 - x_dot**2)
        total_reward += reward

        if theta_error > np.pi / 2:
            fell = True
            total_reward -= 10000  # Penalidade maior por queda
            break

    return total_reward if not fell else total_reward

def select_parents(population, fitnesses, num_parents):
    fitnesses = np.array(fitnesses)
    min_fitness = np.min(fitnesses)
    adjusted_fitnesses = fitnesses - min_fitness if min_fitness < 0 else fitnesses
    total_fitness = np.sum(adjusted_fitnesses)
    probabilities = adjusted_fitnesses / total_fitness if total_fitness > 0 else np.ones(len(fitnesses)) / len(fitnesses)
    indices = np.random.choice(range(len(population)), size=num_parents, replace=True, p=probabilities)
    return indices

def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1))
    child1 = list(parent1[:crossover_point]) + list(parent2[crossover_point:])
    child2 = list(parent2[:crossover_point]) + list(parent1[crossover_point:])
    return tuple(child1), tuple(child2)

def mutate(chromosome, mutation_rate, num_output_options=num_output_options):
    mutated_chromosome = list(chromosome)
    for i in range(len(mutated_chromosome)):
        if np.random.rand() < mutation_rate:
            mutated_chromosome[i] = np.random.randint(num_output_options)
    return tuple(mutated_chromosome)

def genetic_algorithm(population_size=POPULATION_SIZE, num_generations=NUM_GENERATIONS, mutation_rate=MUTATION_RATE, num_parents=NUM_PARENTS):
    print(f"Iniciando algoritmo genético com {num_rules} regras...")
    population = [tuple(np.random.randint(num_output_options, size=num_rules)) for _ in range(population_size)]
    best_fitness_history = []
    best_chromosome = None
    best_fitness = -np.inf

    for generation in range(num_generations):
        fitnesses = [evaluate_chromosome(chromo) for chromo in population]
        best_fitness_current = max(fitnesses)
        best_chromosome_current = population[np.argmax(fitnesses)]

        if best_fitness_current > best_fitness:
            best_fitness = best_fitness_current
            best_chromosome = best_chromosome_current

        best_fitness_history.append(best_fitness)
        print(f"Geração {generation + 1}/{num_generations}, Melhor Fitness: {best_fitness:.2f}")

        # Elitismo: preservar os melhores
        elite_indices = np.argsort(fitnesses)[-ELITISM_COUNT:]
        next_generation = [population[i] for i in elite_indices]

        parents_indices = select_parents(population, fitnesses, num_parents)
        parents = [population[i] for i in parents_indices]

        while len(next_generation) < population_size:
            parent1 = parents[np.random.choice(len(parents))]
            parent2 = parents[np.random.choice(len(parents))]
            child1, child2 = crossover(parent1, parent2)
            next_generation.append(mutate(child1, mutation_rate))
            if len(next_generation) < population_size:
                next_generation.append(mutate(child2, mutation_rate))

        population = next_generation

    print("Otimização Genética Concluída!")
    return best_chromosome, best_fitness_history

if __name__ == '__main__':
    best_chromo, fitness_history = genetic_algorithm()
    print("Melhor Cromossomo:", best_chromo)
    print("Histórico de Fitness:", fitness_history)