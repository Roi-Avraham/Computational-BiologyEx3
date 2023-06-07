import random
import numpy as np

# Constants
POPULATION_SIZE = 10
NUM_GENERATIONS = 10
MUTATION_RATE = 0.1
CROSSOVER_RATE = 1
INPUT_SIZE = 16
OUTPUT_SIZE = 1


# Load data
# Parse the data from nn0.txt
data = []
with open('nn0.txt', 'r') as file:
    for line in file:
        binary_string, label = line.strip().split()
        data.append((binary_string, int(label)))

# Split data into training and test sets
random.shuffle(data)
train_size = int(0.8 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]


# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


# Neural network class
class Network:
    def __init__(self, structure, weights=None):
        self.structure = structure
        if weights is None:
            self.weights = [np.random.randn(structure[i+1], structure[i]) for i in range(len(structure)-1)]
        else:
            self.weights = weights

    def predict(self, input_data):
        hidden = input_data
        for layer_weights in self.weights[:-1]:
            hidden = sigmoid(np.dot(layer_weights, hidden))
        output = sigmoid(np.dot(self.weights[-1], hidden))
        return output


# Genetic algorithm
def initialize_population(population_size, input_size, output_size):
    population = []
    for _ in range(population_size):
        number_of_layers = np.random.randint(2, 11)
        structure = [input_size]
        for _ in range(number_of_layers):
            size_of_layer = np.random.randint(2, 11)
            structure.append(size_of_layer)
        structure.append(output_size)
        print(structure)
        population.append(Network(structure))
    return population


def calculate_fitness(network, data):
    correct_predictions = 0
    for example in data:
        input_data = np.array(list(map(int, example[0])))
        target = example[1]
        output = network.predict(input_data)
        predicted_class = 1 if np.any(output > 0.5) else 0
        if predicted_class == target:
            correct_predictions += 1
    fitness = correct_predictions / len(data)
    return fitness


def selection(population, fitness):
    parent_indices = np.random.choice(range(len(population)), size=2, replace=True, p=fitness / np.sum(fitness))
    parents = [population[idx] for idx in parent_indices]
    return parents[0], parents[1]


def crossover(parent1, parent2):
    offspring_structure = parent1.structure.copy()
    offspring_weights = parent1.weights.copy()
    print(parent1.structure)
    print(parent2.structure)

    crossover_point = np.random.randint(1, min(len(parent1.structure), len(parent2.structure)) - 1)
    offspring_structure[crossover_point:] = parent2.structure[crossover_point:]
    offspring_weights[crossover_point:] = parent2.weights[crossover_point:]
    print("off: ", offspring_structure)
    offspring = Network(offspring_structure, offspring_weights)
    return offspring


def mutation(network):
    return network
    # for i in range(len(network.structure)):
    #     if np.random.uniform() < MUTATION_RATE:
    #         print("hi")
    #         network.structure[i] = np.random.randint(2, 11)  # Randomly mutate structure
    # return network


def evolve(population, train_data):
    fitness = [calculate_fitness(network, train_data) for network in population]

    new_population = []

    # Elitism: Keep the best individual in the population
    best_index = np.argmax(fitness)
    new_population.append(population[best_index])

    while len(new_population) < len(population):
        parent1, parent2 = selection(population, fitness)
        if np.random.uniform() < CROSSOVER_RATE:
            offspring = crossover(parent1, parent2)
        else:
            offspring = parent1
        offspring = mutation(offspring)
        new_population.append(offspring)

    return new_population


# Main loop
population = initialize_population(POPULATION_SIZE, INPUT_SIZE, OUTPUT_SIZE)

for generation in range(NUM_GENERATIONS):
    print("generation number: ", generation + 1)
    population = evolve(population, train_data)
    best_network = population[np.argmax([calculate_fitness(network, train_data) for network in population])]
    print("best is: ", max([calculate_fitness(network, train_data) for network in population]))
    exit(0)

# Evaluate the best network on the test data
test_accuracy = calculate_fitness(best_network, test_data)
print("Test Accuracy:", test_accuracy)
