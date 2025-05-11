# C:\Programacao\Projetos\Python\fuzzy_inverted_pendulum\core\neuro_fuzzy.py

import numpy as np
from tensorflow import keras
from keras import layers
from core.dynamics import simulate_inverted_pendulum, dt
from core.fuzzy_controller import pendulum_angle_mf, pendulum_angular_velocity_mf, car_position_mf, car_velocity_mf, pendulum_rules, car_rules, index_to_force, pendulum_force_sets, car_force_sets

class NeuroFuzzyMLPController:
    def __init__(self):
        self.index_to_force = index_to_force
        self.pendulum_rules = pendulum_rules
        self.car_rules = car_rules
        self.pendulum_angle_mf = pendulum_angle_mf
        self.pendulum_angular_velocity_mf = pendulum_angular_velocity_mf
        self.car_position_mf = car_position_mf
        self.car_velocity_mf = car_velocity_mf
        self.num_rules = len(pendulum_rules) + len(car_rules)  # Definir antes de _build_model
        self.model = self._build_model()
        self.trained = False

    def _build_model(self):
        """Define a arquitetura da rede neural MLP."""
        model = keras.Sequential([
            layers.Input(shape=(self.num_rules,)),
            layers.Dense(64, activation='relu', input_shape=(self.num_rules,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(len(self.index_to_force), activation='softmax')
        ])
        model.compile(optimizer='adam', 
                     loss='sparse_categorical_crossentropy', 
                     metrics=['accuracy'])
        return model

    def _infer_fuzzy_strength(self, state):
        """Calcula a 'força' de cada regra fuzzy para um dado estado."""
        theta_mfs = self.pendulum_angle_mf(state[2])
        theta_dot_mfs = self.pendulum_angular_velocity_mf(state[3])
        x_mfs = self.car_position_mf(state[0])
        x_dot_mfs = self.car_velocity_mf(state[1])

        rule_strengths = []
        for rule in self.pendulum_rules:
            strength = min(theta_mfs[rule['if']['theta']], theta_dot_mfs[rule['if']['theta_dot']])
            rule_strengths.append(strength)
        for rule in self.car_rules:
            strength = min(x_mfs[rule['if']['x']], x_dot_mfs[rule['if']['x_dot']])
            rule_strengths.append(strength)
        return np.array(rule_strengths)

    def train(self, training_data, epochs=10, batch_size=32):
        """Treina a rede neural com dados de estado e a ação de controle desejada."""
        inputs = np.array([self._infer_fuzzy_strength(data[0]) for data in training_data])
        outputs = np.array([data[1] for data in training_data])  # Assumindo que training_data tem (state, action_index)
        self.model.fit(inputs, outputs, epochs=epochs, batch_size=batch_size, verbose=0)
        self.trained = True

    def predict_force(self, state):
        """Prevê a força de controle para um determinado estado usando a rede neural."""
        if self.trained:
            fuzzy_strengths = self._infer_fuzzy_strength(state)
            prediction = self.model.predict(np.array([fuzzy_strengths]), verbose=0)
            predicted_action_index = np.argmax(prediction, axis=1)[0]
            force_name = self.index_to_force.get(predicted_action_index, 'Z')
            # Mapear o nome da força para o valor numérico correspondente
            if predicted_action_index <= 6:  # Forças do pêndulo (índices 0 a 6)
                force_value = pendulum_force_sets[force_name]['peak']
            else:  # Forças do carro (índices 7 a 13, mas não usamos mais)
                force_value = car_force_sets[force_name]['peak']
            return force_value
        else:
            print("Neuro-Fuzzy não treinado. Usando FIS padrão.")
            from core.fuzzy_controller import combined_fis_control
            return combined_fis_control(state)

def collect_training_data_with_fis(num_episodes=100, max_steps=200, initial_state_range=1.0):
    """Gera dados de treinamento simulando o FIS para coletar (estado, ação)."""
    training_data = []
    for _ in range(num_episodes):
        initial_state = [
            np.random.uniform(-initial_state_range, initial_state_range),
            np.random.uniform(-initial_state_range, initial_state_range),
            np.pi + np.random.uniform(-0.2, 0.2),
            np.random.uniform(-initial_state_range, initial_state_range)
        ]
        current_state = initial_state[:]
        for _ in range(max_steps):
            from core.fuzzy_controller import combined_fis_control
            force = combined_fis_control(current_state)
            # Mapear a força para o índice mais próximo com base nos valores de pico
            force_diffs_pendulum = {idx: abs(pendulum_force_sets[force_name]['peak'] - force)
                                    for idx, force_name in index_to_force.items()}
            force_diffs_car = {idx: abs(car_force_sets[force_name]['peak'] - force)
                               for idx, force_name in index_to_force.items()}
            # Combinar as diferenças e encontrar o índice mais próximo
            force_diffs = {**force_diffs_pendulum, **force_diffs_car}
            action_index = min(force_diffs, key=force_diffs.get)
            training_data.append((current_state, action_index))
            next_state = simulate_inverted_pendulum(current_state, force, dt)
            current_state = next_state
            if abs(current_state[2] - np.pi) > np.pi / 2:
                break
    return training_data

if __name__ == '__main__':
    controller = NeuroFuzzyMLPController()
    training_data = collect_training_data_with_fis(num_episodes=50)
    controller.train(training_data, epochs=10)

    initial_state = [0.1, 0.1, np.pi + 0.05, 0.05]
    current_state = initial_state[:]
    simulation_history = [current_state]

    for _ in range(200):
        force = controller.predict_force(current_state)
        next_state = simulate_inverted_pendulum(current_state, force, dt)
        simulation_history.append(next_state)
        current_state = next_state
        if abs(current_state[2] - np.pi) > np.pi / 2:
            break
    print("Simulação com Neuro-Fuzzy (MLP) concluída.")