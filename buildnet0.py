import random
import numpy as np

# Define the genetic algorithm parameters
population_size = 100
mutation_rate = 0.01
num_generations = 50

# Define the neural network architecture
input_size = 16
hidden_size = 10
output_size = 1

# Load the data from files
# Parse the data from nn0.txt
data = []
with open('nn0.txt', 'r') as file:
    for line in file:
        binary_string, label = line.strip().split()
        data.append((binary_string, int(label)))


# Combine the data and shuffle
# data = np.concatenate((nn0_data, nn1_data))
np.random.shuffle(data)

# Split the data into training and testing sets
train_size = int(0.8 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]


# Define the fitness function
def calculate_fitness(network):
    correct_predictions = 0
    for example in train_data:
        input_data = example[:-1]
        target = example[-1]
        output = network.predict(input_data)
        predicted_class = 1 if output > 0.5 else 0
        if predicted_class == target:
            correct_predictions += 1
    fitness = correct_predictions / len(train_data)
    return fitness


# Define the network class
class Network:
    def __init__(self, weights=None):
        if weights is None:
            # Initialize random weights
            self.weights = [
                np.random.randn(input_size, hidden_size),
                np.random.randn(hidden_size, output_size)
            ]
        else:
            self.weights = weights

    def predict(self, input_data):
        hidden = np.dot(input_data.reshape(1, -1), self.weights[0])
        hidden = sigmoid(hidden)
        output = np.dot(hidden, self.weights[1])
        output = sigmoid(output)
        return output


# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Create the initial population
population = []
for _ in range(population_size):
    network = Network()
    population.append(network)

# Run the genetic algorithm
for generation in range(num_generations):
    print("Generation:", generation + 1)

    # Calculate fitness for each network
    fitness_scores = []
    for network in population:
        fitness = calculate_fitness(network)
        fitness_scores.append((network, fitness))

    # Sort the population based on fitness
    fitness_scores.sort(key=lambda x: x[1], reverse=True)

    # Select top performing networks for breeding
    fittest_networks = [network for network, _ in fitness_scores[:10]]

    # Create the next generation
    next_generation = fittest_networks.copy()

    # Breed new networks
    while len(next_generation) < population_size:
        parent1 = random.choice(fittest_networks)
        parent2 = random.choice(fittest_networks)
        child_weights = []
        for i in range(len(parent1.weights)):
            # Crossover
            crossover_point = random.randint(0, parent1.weights[i].size)
            child_weights.append(np.concatenate((
                parent1.weights[i].flatten()[:crossover_point],
                parent2.weights[i].flatten()[crossover_point:]
            )).reshape(parent1.weights[i].shape))
            # Mutation
            if random.random() < mutation_rate:
                mutation_index = random.randint(0, child_weights[i].size - 1)
                child_weights[i].flatten()[mutation_index] = np.random.randn()

        child = Network(child_weights)
        next_generation.append(child)

    # Update the population
    population = next_generation


def calculate_fitness_on_test_data(network):
    correct_predictions = 0
    for example in test_data:
        input_data = example[:-1]
        target = example[-1]
        output = network.predict(input_data)
        predicted_class = 1 if output > 0.5 else 0
        if predicted_class == target:
            correct_predictions += 1
    fitness = correct_predictions / len(train_data)
    return fitness


# Evaluate the best network on the test data
best_network = fitness_scores[0][0]
test_accuracy = calculate_fitness_on_test_data(best_network)
print("Test accuracy:", test_accuracy)