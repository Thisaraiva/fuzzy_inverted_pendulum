# C:\Programacao\Projetos\Python\fuzzy_inverted_pendulum\core\neuro_fuzzy.py

import numpy as np
from tensorflow import keras
from keras import layers
from core.dynamics import simulate_inverted_pendulum
from core.fuzzy_controller import pendulum_angle_mf, pendulum_angular_velocity_mf, car_position_mf, car_velocity_mf, pendulum_rules, car_rules, index_to_force, pendulum_force_sets
from core.config import DT, NUM_EPISODES, MAX_STEPS, EPOCHS, BATCH_SIZE, SIMULATION_TIME, FORCE_LIMIT

class NeuroFuzzyMLPController:
    def __init__(self):
        self.index_to_force = index_to_force
        self.pendulum_rules = pendulum_rules
        self.car_rules = car_rules
        self.num_rules = len(pendulum_rules) + len(car_rules)
        self.model = self._build_model()
        self.trained = False

    def _build_model(self):
        model = keras.Sequential([
            layers.Input(shape=(self.num_rules,)),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(64, activation='relu'),
            layers.Dense(len(self.index_to_force), activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def _infer_fuzzy_strength(self, state):
        theta_mfs = pendulum_angle_mf(state[2] - np.pi)
        theta_dot_mfs = pendulum_angular_velocity_mf(state[3])
        x_mfs = car_position_mf(state[0])
        x_dot_mfs = car_velocity_mf(state[1])

        rule_strengths = []
        for rule in self.pendulum_rules:
            strength = min(theta_mfs[rule['if']['theta']], theta_dot_mfs[rule['if']['theta_dot']])
            rule_strengths.append(strength)
        for rule in self.car_rules:
            strength = min(x_mfs[rule['if']['x']], x_dot_mfs[rule['if']['x_dot']])
            rule_strengths.append(strength)
        return np.clip(rule_strengths, 0, 1)

    def train(self, training_data, epochs=EPOCHS, batch_size=BATCH_SIZE):
        inputs = np.array([self._infer_fuzzy_strength(data[0]) for data in training_data])
        outputs = np.array([data[1] for data in training_data])
        self.model.fit(inputs, outputs, epochs=epochs, batch_size=batch_size, verbose=0)
        self.trained = True

    def predict_force(self, state):
        if self.trained:
            fuzzy_strengths = self._infer_fuzzy_strength(state)
            prediction = self.model.predict(np.array([fuzzy_strengths]), verbose=0)
            predicted_action_index = np.argmax(prediction, axis=1)[0]
            force_name = self.index_to_force.get(predicted_action_index, 'Z')
            return np.clip(pendulum_force_sets[force_name]['peak'], -FORCE_LIMIT, FORCE_LIMIT)
        else:
            from core.fuzzy_controller import combined_fis_control
            return combined_fis_control(state)

def collect_training_data_with_fis(num_episodes=NUM_EPISODES, max_steps=MAX_STEPS, initial_state_range=1.0):
    training_data = []
    for episode in range(num_episodes):
        initial_state = [
            np.random.uniform(-initial_state_range, initial_state_range),
            np.random.uniform(-initial_state_range, initial_state_range),
            np.pi + np.random.uniform(-0.3, 0.3),
            np.random.uniform(-initial_state_range, initial_state_range)
        ]
        current_state = initial_state[:]
        for _ in range(max_steps):
            from core.fuzzy_controller import combined_fis_control
            force = combined_fis_control(current_state)
            force_diffs = {idx: min(
                abs(pendulum_force_sets[force_name]['peak'] - force),
                abs(car_force_sets[force_name]['peak'] - force)
            ) for idx, force_name in index_to_force.items()}
            action_index = min(force_diffs, key=force_diffs.get)
            training_data.append((current_state, action_index))
            next_state = simulate_inverted_pendulum(current_state, force)
            current_state = next_state
            if abs(current_state[2] - np.pi) > np.pi / 2:
                break
        print(f"Episódio {episode + 1}/{num_episodes} concluído.")
    return training_data

if __name__ == '__main__':
    controller = NeuroFuzzyMLPController()
    training_data = collect_training_data_with_fis()
    controller.train(training_data)
    initial_state = [0.0, 0.0, np.pi + 0.1, 0.0]
    current_state = initial_state[:]
    for _ in range(int(SIMULATION_TIME / DT)):
        force = controller.predict_force(current_state)
        next_state = simulate_inverted_pendulum(current_state, force)
        current_state = next_state
        if abs(current_state[2] - np.pi) > np.pi / 2:
            print("Pêndulo caiu!")
            break
    print("Simulação Neuro-Fuzzy concluída.")