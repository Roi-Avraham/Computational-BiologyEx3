correct_predictions = 0


def load_data(data_file):
    with open(data_file, 'r') as file:
        data = [line.strip() for line in file]
    return data


dataset = load_data('nn0.txt')
number_line = 0
for line in dataset:
    number_line+=1
    string = line[:-1]
    actual_label = line[-1]
    ones_count = string.count("1")
    zeros_count = string.count("0")

    # Apply the consistency rule
    predicted_label = 1 if  13> ones_count > 7 else 0

    if str(predicted_label) == actual_label:
        correct_predictions += 1
    else:
        print("mmm")

accuracy = correct_predictions / len(dataset)
print("Accuracy: {:.2%}".format(accuracy))

