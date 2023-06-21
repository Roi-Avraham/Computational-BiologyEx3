import sys

import numpy as np


def relu(x):
    return np.maximum(0, x)


# Define the Neural Network class
class NeuralNetwork:
    def __init__(self, structure):
        self.structure = structure
        self.num_layers = len(structure)
        self.weights = []

    def predict(self, input_data):
        hidden = input_data
        for layer_weights in self.weights[:-1]:
            hidden = relu(np.dot(layer_weights, hidden))
        output = relu(np.dot(self.weights[-1],hidden))
        return output


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Load the network structure and weights from file
def load_network(network_file):
    with open(network_file, 'r') as file:
        lines = file.readlines()
        structure = [int(val) for val in lines[0].strip().split()]
        weights = []
        num_layers = len(structure) - 1
        current_line = 1

        for i in range(num_layers):
            rows = structure[i+1]
            print(rows)
            cols = structure[i]
            print(cols)
            weight_matrix = np.array([float(val) for val in lines[current_line].strip().split()])
            weight_matrix = weight_matrix.reshape((rows, cols))
            weights.append(weight_matrix)
            current_line += 1

    return structure, weights


# Load the data from file
def load_data(data_file):
    with open(data_file, 'r') as file:
        data = [line.strip() for line in file]
    return data


# Save the classifications to file
def save_classifications(classifications, output_file, data):
    # Write the data and classifications to the output file
    with open(output_file, 'w') as file:
        for i in range(len(data)):
            # file.write(data[i] + '   ' + str(classifications[i]) + '\n')
            file.write(str(classifications[i]) + '\n')
            # file.write(data[i] + ' ' + str(classifications[i]) + '\n')


# Main program
def run_network(network_file, data_file, output_file):
    # Load the network structure and weights
    network_structure, weights = load_network(network_file)

    # Create the neural network
    network = NeuralNetwork(network_structure)
    network.weights = weights

    # Load the data
    data = load_data(data_file)

    # Run the network on the data
    classifications = []
    for input_data in data:
        input_data = np.array([int(val) for val in input_data])
        classification = (network.predict(input_data))
        if classification > 0.5:
            classification = 1
        else:
            classification = 0
        classifications.append(classification)

    # Save the classifications to file
    save_classifications(classifications, output_file, data)


if __name__ == '__main__':
    # Example usage
    # network_file = 'wnet0.txt'  # Network structure and weights file
    # data_file = 'testnet_test.txt'  # Data file
    # output_file = 'classifications_test.txt'  # Output file

    network_file = sys.argv[1]  # Network structure and weights file
    data_file = sys.argv[2]  # Data file
    output_file = 'classifications0.txt'  # Output file

    run_network(network_file, data_file, output_file)
