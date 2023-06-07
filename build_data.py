def remove_last_number(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    modified_lines = [line.strip().rsplit(' ', 1)[0] for line in lines]

    with open(output_file, 'w') as file:
        file.write('\n'.join(modified_lines))

# Usage
input_file = 'nn0.txt'
output_file = 'data.txt'
remove_last_number(input_file, output_file)