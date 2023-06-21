import random
import statistics

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score

# Constants
POPULATION_SIZE = 100
NUM_GENERATIONS = 100
MUTATION_RATE = 0.1
CROSSOVER_RATE = 1
INPUT_SIZE = 16
OUTPUT_SIZE = 1
ELITE_SIZE = 5
BEST = (POPULATION_SIZE * 10) // 100
dict_graphs = {}
best_fittness = []
average_fittness = []
worst_fittness = []

# Load data
# Parse the data from nn0.txt
data = []
with open('nn1.txt', 'r') as file:
    for line in file:
        binary_string, label = line.strip().split()
        data.append((binary_string, int(label)))

# Split data into training and test sets
random.shuffle(data)
train_size = int(0.8 * len(data))
train_data = data[:train_size//16]
test_data = data[train_size:]


# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)

def xavier_init(shape):
    """
    Xavier initialization for weight matrices.
    """
    n_inputs, n_outputs = shape[1], shape[0]
    limit = np.sqrt(6 / (n_inputs + n_outputs))
    return np.random.uniform(-limit, limit, shape)


# Neural network class
class Network:
    def __init__(self, structure, weights=None):
        self.structure = structure
        if weights is None:
            # self.weights = [np.random.randn(structure[i + 1], structure[i]) for i in range(len(structure) - 1)]
            self.weights = [xavier_init((structure[i + 1], structure[i])) for i in range(len(structure) - 1)]
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
        structure = [input_size, 2, output_size]  # Randomly initialize network structure
        population.append(Network(structure))
    return population


def calculate_fitness(network, data):
    correct_predictions = 0
    for example in data:
        input_data = np.array(list(map(int, example[0])))
        target = int(example[1])
        output = network.predict(input_data)
        predicted_class = 1 if output > 0.5 else 0

        if predicted_class == target:
            correct_predictions += 1
    print(correct_predictions)
    fitness = correct_predictions / len(data)
    return fitness


def selection(population, fitness):
    parent_indices = np.random.choice(range(len(population)), size=2, replace=True, p=fitness / np.sum(fitness))
    parents = [population[idx] for idx in parent_indices]
    return parents[0], parents[1]


def crossover(parent1, parent2):
    offspring_weights_one = []
    offspring_weights_two = []
    for i in range(len(parent1.weights)):
        # Get the shape of the matrices
        rows, cols = parent1.weights[i].shape

        # Generate a random crossover point
        crossover_point = np.random.randint(1, cols)

        # Perform one-point crossover
        offspring_one = np.concatenate((parent1.weights[i][:, :crossover_point], parent2.weights[i][:, crossover_point:]),
                                   axis=1)
        offspring_weights_one.append(offspring_one)

        offspring_two = np.concatenate(
            (parent2.weights[i][:, :crossover_point], parent1.weights[i][:, crossover_point:]),
            axis=1)
        offspring_weights_two.append(offspring_two)

    child_one = Network(parent1.structure, offspring_weights_one)
    child_two = Network(parent1.structure, offspring_weights_two)

    return child_one, child_two


def mutation(network, mutation_rate=0.1, distribution_index=20):
    mutated_weights = []
    for weight_matrix in network.weights:
        mutated_matrix = weight_matrix.copy()
        rows, cols = mutated_matrix.shape
        for i in range(rows):
            for j in range(cols):
                if random.random() <= MUTATION_RATE:
                    u = random.random()
                    if u <= 0.5:
                        delta = (2 * u) ** (1.0 / (distribution_index + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1.0 / (distribution_index + 1))

                    mutated_matrix[i, j] += delta

                    # Clip the mutated value to the range of -1 to 1
                    mutated_matrix[i, j] = np.clip(mutated_matrix[i, j], -1, 1)

        mutated_weights.append(mutated_matrix)

    mutated_network = Network(network.structure, mutated_weights)
    return mutated_network


def evolve(population, train_data):
    global best_network
    global best_fittness
    global average_fittness
    global worst_fittness
    fitness_array = [(network, calculate_fitness(network, train_data)) for network in population]
    fitness_array.sort(key=lambda x: x[1], reverse=True)
    fitness = [f for network, f in fitness_array]
    worst_fittness.append(fitness_array[-1][1])
    average_fittness.append(statistics.mean(fitness))

    fitness_array = fitness_array[:-BEST] + [fitness_array[0]]*BEST
    fitness_array.sort(key=lambda x: x[1], reverse=True)
    population = [network for network, f in fitness_array]
    fitness = [f for network, f in fitness_array]

    # Elitism: Keep the best individual in the population
    best_network = fitness_array[0][0]
    print("best is: ", fitness_array[0][1])
    best_fittness.append(fitness_array[0][1])
    new_population = [network for network, f in fitness_array[:ELITE_SIZE]]

    while len(new_population) < len(population):
        parent1, parent2 = selection(population, fitness)
        if np.random.uniform() < CROSSOVER_RATE:
            offspring_one, offspring_two = crossover(parent1, parent2)
            offspring_one = mutation(offspring_one)
            offspring_two = mutation(offspring_two)
            new_population.append(offspring_one)
            new_population.append(offspring_two)
        else:
            offspring = parent1
            new_population.append(offspring)

    return new_population


def genetic(key):
    global dict_graphs
    # Main loop
    population = initialize_population(POPULATION_SIZE, INPUT_SIZE, OUTPUT_SIZE)

    for generation in range(NUM_GENERATIONS):
        print("generation number: ", generation + 1)
        population = evolve(population, train_data)


    # Evaluate the best network on the test data
    test_accuracy = calculate_fitness(best_network, test_data)
    print("Test Accuracy:", test_accuracy)

    best_fittness.append(test_accuracy)
    dict_graphs[key] = [best_fittness, average_fittness, worst_fittness]



def make_arguments(pz, ng, mr, cr):
    global POPULATION_SIZE
    global NUM_GENERATIONS
    global MUTATION_RATE
    global CROSSOVER_RATE
    global ELITE_SIZE
    global BEST
    global best_fittness
    global average_fittness
    global worst_fittness
    POPULATION_SIZE = pz
    NUM_GENERATIONS = ng
    MUTATION_RATE = mr
    CROSSOVER_RATE = cr
    ELITE_SIZE = (POPULATION_SIZE * 10) // 100
    BEST = (POPULATION_SIZE * 10) // 100
    best_fittness = []
    average_fittness = []
    worst_fittness = []



def start():
    global dict_graphs

    population_array = [30, 50, 100, 150]
    for i in range(len(population_array)):
        make_arguments(population_array[i], 100, 0.1, 1)
        genetic(population_array[i])

    create_graphs("POPULATION_SIZE")
    dict_graphs = {}

    # num_generation_array = [30, 50, 100, 150]
    # for i in range(len(num_generation_array)):
    #     make_arguments(100, num_generation_array[i], 0.1, 1)
    #     genetic(num_generation_array[i])
    # create_graphs("NUM_GENERATIONS")
    # dict_graphs = {}
    #
    # mutation_array = [0.1, 0.5, 0.75, 1]
    # for i in range(len(mutation_array)):
    #     make_arguments(100, 100, mutation_array[i], 1)
    #     genetic(mutation_array[i])
    # create_graphs("MUTATION_RATE")
    # dict_graphs = {}




def create_graphs(parm):
    global dict_graphs
    # Create a new figure and axis
    plt.figure()  # create a new figure

    for key in dict_graphs:
        generation = [i for i in range(1, len(dict_graphs[key][0]))]
        generation = np.array(generation)
        plt.plot(generation, dict_graphs[key][0][:-1], label=f'{key} {parm}')

    # Set labels and title
    plt.xlabel("generation")
    plt.ylabel('best accuracy')

    # Add a legend
    plt.legend()

    plt.figure()  # create a new figure
    # Plot the data
    for key in dict_graphs:
        plt.bar(key, dict_graphs[key][0][-1], label=f'{key} {parm}')

    plt.xlabel(f'{parm}')
    plt.ylabel('test accuracy')

    # Add a legend
    plt.legend()
    plt.xlim(left=0)

    plt.figure()  # create a new figure

    for key in dict_graphs:
        generation = [i for i in range(0, len(dict_graphs[key][1]))]
        generation = np.array(generation)
        plt.plot(generation, dict_graphs[key][1], label=f'{key} {parm}')

    # Set labels and title
    plt.xlabel("generation")
    plt.ylabel('average accuracy')

    # Add a legend
    plt.legend()

    plt.figure()  # create a new figure

    for key in dict_graphs:
        generation = [i for i in range(0, len(dict_graphs[key][2]))]
        generation = np.array(generation)
        plt.plot(generation, dict_graphs[key][2], label=f'{key} {parm}')

    # Set labels and title
    plt.xlabel("generation")
    plt.ylabel('worst accuracy')

    # Add a legend
    plt.legend()

    # Display the plot
    plt.show()


start()

