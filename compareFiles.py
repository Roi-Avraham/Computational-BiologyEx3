def compare_files(file1_path, file2_path):
    with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
        lines1 = file1.readlines()
        lines2 = file2.readlines()
    count = 0
    # Compare line by line ignoring spaces
    for line_num, (line1, line2) in enumerate(zip(lines1, lines2), start=1):
        line1_stripped = line1.replace("    ", "").strip()
        line2_stripped = line2.replace("    ", "").strip()
        if line1_stripped != line2_stripped:
            count += 1
            print(f"Difference found at line {line_num}:")
            print(f"File 1: {line1.strip()}")
            print(f"File 2: {line2.strip()}")
            print()  # Empty line for separation

    # Check for differences in file length
    if len(lines1) != len(lines2):

        print("The files have different lengths.")

    # If files are identical, print a message
    if lines1 == lines2:
        print("The files are identical.")

    print(count)


# Provide the paths to the files you want to compare
file1_path = 'new_test1.txt'
file2_path = 'classifications_test1.txt'

# Call the compare_files function
compare_files(file1_path, file2_path)