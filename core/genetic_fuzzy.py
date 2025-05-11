# Dentro de genetic_fuzzy.py

import numpy as np
from core.dynamics import simulate_inverted_pendulum, dt
from core.fuzzy_controller import pendulum_angle_mf, pendulum_angular_velocity_mf, pendulum_rules, car_rules, car_position_mf, car_velocity_mf, pendulum_force_sets, car_force_sets

# Combinando as regras (a ordem é importante para a representação genética)
all_rules = pendulum_rules + car_rules
num_rules = len(all_rules)
output_force_indices = {'NL': 0, 'NM': 1, 'NS': 2, 'Z': 3, 'PS': 4, 'PM': 5, 'PL': 6}
index_to_force = {v: k for k, v in output_force_indices.items()}
num_output_options = len(output_force_indices)  # Garante consistência com o número de opções de saída

def get_force_from_chromosome(state, chromosome):
    """Calcula a força de controle com base no cromossomo e no estado atual."""
    theta_mfs = pendulum_angle_mf(state[2])
    theta_dot_mfs = pendulum_angular_velocity_mf(state[3])
    x_mfs = car_position_mf(state[0])
    x_dot_mfs = car_velocity_mf(state[1])

    activated_rules = []
    # Para as regras do pêndulo
    for i, rule in enumerate(pendulum_rules):
        if i >= len(chromosome):
            print(f"Erro: índice {i} fora do intervalo do cromossomo (tamanho: {len(chromosome)})")
            continue
        theta_degree = theta_mfs[rule['if']['theta']]
        theta_dot_degree = theta_dot_mfs[rule['if']['theta_dot']]
        activation_degree = min(theta_degree, theta_dot_degree)
        if activation_degree > 0:
            force_index = chromosome[i]
            output_force_name = index_to_force.get(force_index)
            if output_force_name:
                activated_rules.append({'force': output_force_name, 'degree': activation_degree, 'type': 'pendulum'})

    # Para as regras do carro
    for i, rule in enumerate(car_rules):
        idx = len(pendulum_rules) + i
        if idx >= len(chromosome):
            print(f"Erro: índice {idx} fora do intervalo do cromossomo (tamanho: {len(chromosome)})")
            continue
        x_degree = x_mfs[rule['if']['x']]
        x_dot_degree = x_dot_mfs[rule['if']['x_dot']]
        activation_degree = min(x_degree, x_dot_degree)
        if activation_degree > 0:
            force_index = chromosome[idx]
            output_force_name = index_to_force.get(force_index)
            if output_force_name:
                activated_rules.append({'force': output_force_name, 'degree': activation_degree, 'type': 'car'})

    pendulum_force_numerator = 0
    pendulum_force_denominator = 0
    car_force_numerator = 0
    car_force_denominator = 0

    for activated_rule in activated_rules:
        if activated_rule['type'] == 'pendulum':
            force_set = pendulum_force_sets.get(activated_rule['force'])
            if force_set:
                pendulum_force_numerator += activated_rule['degree'] * force_set['peak']
                pendulum_force_denominator += activated_rule['degree']
        elif activated_rule['type'] == 'car':
            force_set = car_force_sets.get(activated_rule['force'])
            if force_set:
                car_force_numerator += activated_rule['degree'] * force_set['peak']
                car_force_denominator += activated_rule['degree']

    force_pendulum = pendulum_force_numerator / pendulum_force_denominator if pendulum_force_denominator else 0
    force_car = car_force_numerator / car_force_denominator if car_force_denominator else 0

    return force_pendulum + 0.5 * force_car

def evaluate_chromosome(chromosome, num_steps=200, initial_state=[0.0, 0.0, np.pi + 0.1, 0.0]):
    """Avalia um cromossomo simulando o controle do pêndulo."""
    current_state = initial_state[:]
    total_reward = 0
    fell = False

    for i in range(num_steps):
        force = get_force_from_chromosome(current_state, chromosome)
        current_state = simulate_inverted_pendulum(current_state, force, dt)

        theta_error = abs(current_state[2] - np.pi)
        x_penalty = abs(current_state[0])
        # Modificação na recompensa para garantir valores mais positivos
        reward = np.exp(-theta_error**2 - 0.1 * x_penalty**2)
        total_reward += reward

        # Penalizar se o pêndulo cair muito (com um valor negativo, mas não infinito)
        if abs(current_state[2] - np.pi) > np.pi / 2:
            fell = True
            total_reward -= 5  # Penalidade para queda
            break

        # Pequena penalidade por passo
        total_reward -= 0.001

        # Penalizar velocidade do carro
        total_reward -= 0.0005 * abs(current_state[1])

    if fell:
        return total_reward
    else:
        return total_reward

def select_parents(population, fitnesses, num_parents):
    """Seleciona pais usando o método da roleta e retorna seus índices."""
    fitnesses = np.array(fitnesses)
    min_fitness = np.min(fitnesses)
    if min_fitness < 0:
        adjusted_fitnesses = fitnesses - min_fitness
    else:
        adjusted_fitnesses = fitnesses

    total_fitness = np.sum(adjusted_fitnesses)

    if total_fitness <= 0:
        probabilities = np.ones(len(fitnesses)) / len(fitnesses)
    else:
        probabilities = adjusted_fitnesses / total_fitness
        probabilities = np.nan_to_num(probabilities, nan=1/len(fitnesses))

    indices = np.random.choice(range(len(population)), size=num_parents, replace=True, p=probabilities)
    return indices  # Retorna os índices dos pais

def crossover(parent1, parent2):
    """Realiza o cruzamento de um ponto."""
    crossover_point = np.random.randint(1, len(parent1))
    child1 = list(parent1[:crossover_point]) + list(parent2[crossover_point:])
    child2 = list(parent2[:crossover_point]) + list(parent1[crossover_point:])
    return tuple(child1), tuple(child2)

def mutate(chromosome, mutation_rate=0.01, num_output_options=num_output_options):
    """Realiza a mutação em um cromossomo."""
    mutated_chromosome = list(chromosome)
    for i in range(len(mutated_chromosome)):
        if np.random.rand() < mutation_rate:
            mutated_chromosome[i] = np.random.randint(num_output_options)
    return tuple(mutated_chromosome)

def genetic_algorithm(population_size=50, num_generations=100, mutation_rate=0.01, num_parents=20):
    """Implementa o algoritmo genético."""
    # Verificar o número de regras
    print(f"Total de regras: {num_rules} (pendulum_rules: {len(pendulum_rules)}, car_rules: {len(car_rules)})")
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
        print(f"Geração {generation + 1}, Melhor Fitness: {best_fitness:.4f}")

        parents_indices = select_parents(population, fitnesses, num_parents)
        parents = [population[i] for i in parents_indices]

        print(f"Tipo de 'parents': {type(parents)}")
        if parents:
            print(f"Tipo do primeiro elemento de 'parents': {type(parents[0])}")
            print(f"Formato do primeiro elemento de 'parents': {np.array(parents[0]).shape}")

        next_generation = list(parents)

        while len(next_generation) < population_size:
            parent1 = parents[np.random.choice(len(parents))]
            parent2 = parents[np.random.choice(len(parents))]
            child1, child2 = crossover(parent1, parent2)
            next_generation.append(mutate(child1, mutation_rate))
            if len(next_generation) < population_size:
                next_generation.append(mutate(child2, mutation_rate))

        population = [tuple(chromo) for chromo in next_generation]

    print("Otimização Genética Concluída!")
    return best_chromosome, best_fitness_history

if __name__ == '__main__':
    best_chromo, fitness_history = genetic_algorithm(population_size=20, num_generations=30)
    print("Melhor Cromossomo Encontrado:", best_chromo)
    print("Histórico de Fitness:", fitness_history)