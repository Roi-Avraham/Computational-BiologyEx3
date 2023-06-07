import random
import numpy as np

# Genetic Algorithm Parameters
POPULATION_SIZE = 25
NUM_GENERATIONS = 50
MUTATION_RATE = 0.9

# Neural Network Parameters
INPUT_SIZE = 16
HIDDEN_SIZE = 10
OUTPUT_SIZE = 1

# Parse the data from nn0.txt
data = []
with open('nn0.txt', 'r') as file:
    for line in file:
        binary_string, label = line.strip().split()
        data.append((binary_string, int(label)))

# Split data into training and test sets
random.shuffle(data)
train_size = int(0.8 * len(data))
train_set = data[:train_size]
test_set = data[train_size:]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Initialize the population of neural networks
def create_network():
    network = {
        'hidden_weights': np.random.uniform(size=(INPUT_SIZE, HIDDEN_SIZE)),
        'output_weights': np.random.uniform(size=(HIDDEN_SIZE, OUTPUT_SIZE))
    }
    return network


population = [create_network() for _ in range(POPULATION_SIZE)]


# Evaluate the fitness of a network
def evaluate(network):
    correct_predictions = 0
    for binary_string, label in train_set:
        input_vector = np.array(list(map(int, binary_string)))
        hidden_activations = np.dot(input_vector, network['hidden_weights'])
        # hidden_outputs = np.maximum(hidden_activations, 0)  # ReLU activation
        hidden_outputs = sigmoid(hidden_activations)
        output_activations = np.dot(hidden_outputs, network['output_weights'])
        output_sigmoid = sigmoid(output_activations)
        output = 1 if output_sigmoid >= 0.5 else 0
        if output == label:
            correct_predictions += 1
    return correct_predictions/ len(train_set)

# Genetic Algorithm
for generation in range(NUM_GENERATIONS):
    print("in generation", generation + 1)

    # Evaluate fitness of each network
    fitness_scores = [(network, evaluate(network)) for network in population]

    # Sort the population based on fitness
    fitness_scores.sort(key=lambda x: x[1], reverse=True)
    print("best score is ", fitness_scores[0][1])

    # Select top performing networks for breeding
    fittest_networks = [network for network, _ in fitness_scores[:5]]

    fittness = [fittnes for _, fittnes in fitness_scores]

    # Select parents for reproduction
    parent_indices = np.random.choice(range(POPULATION_SIZE), size=POPULATION_SIZE, replace=True, p=fittness/np.sum(fittness))

    # Create offspring through crossover and mutation
    offspring = []
    for i in range(POPULATION_SIZE-5):

        random_index_one = np.random.randint(0, len(parent_indices))
        random_index_two = np.random.randint(0, len(parent_indices))

        parent1 = population[parent_indices[random_index_one]]
        parent2 = population[parent_indices[random_index_two]]
        # offspring_network = {
        #     'hidden_weights': np.where(np.random.rand(*parent1['hidden_weights'].shape) < 0.5, parent1['hidden_weights'], parent2['hidden_weights']),
        #     'output_weights': np.where(np.random.rand(*parent1['output_weights'].shape) < 0.5, parent1['output_weights'], parent2['output_weights'])
        # }

        random_index = np.random.randint(0, len(parent1['hidden_weights']))

        offspring_network = {
            'hidden_weights': np.concatenate(
                (parent1['hidden_weights'][:random_index], parent2['hidden_weights'][random_index:])),
            'output_weights': np.concatenate(
                (parent1['output_weights'][:random_index], parent2['output_weights'][random_index:]))
        }


        # Mutation
        for layer in ['hidden_weights', 'output_weights']:
            mask = np.random.rand(*offspring_network[layer].shape) < MUTATION_RATE
            random_uniform = np.random.uniform(size=offspring_network[layer].shape)
            offspring_network[layer] = np.where(mask, random_uniform, offspring_network[layer])
        offspring.append(offspring_network)

    # Replace old population with the offspring
    population = fittest_networks + offspring

# Select the best-performing network
best_network = max(population, key=evaluate)

# Evaluate the best network on the test set
correct_predictions = 0
for binary_string, label in test_set:
    input_vector = np.array(list(map(int, binary_string)))
    hidden_activations = np.dot(input_vector, best_network['hidden_weights'])
    # hidden_outputs = np.maximum(hidden_activations, 0)  # ReLU activation
    hidden_outputs = sigmoid(hidden_activations)
    output_activations = np.dot(hidden_outputs, best_network['output_weights'])
    output_sigmoid = sigmoid(output_activations)
    output = 1 if output_sigmoid >= 0.5 else 0
    if output == label:
        correct_predictions += 1

accuracy = correct_predictions / len(test_set)
print("Accuracy on test set:", accuracy)