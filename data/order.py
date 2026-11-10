import re, numpy as np
from collections import OrderedDict
plus_one = False

def read_rock_matrices(file_path, apply_increment=False):
    rock_data = []           # Will store tuples: (rock_id, header, matrix) 
    current_matrix = []      # Buffer for the current matrix rows
    current_rock_id = None   # To store the extracted (and incremented) rock id
    current_header = None    # To store the header line as read and modified
    header_pattern = re.compile(r"(Rock_[0-9]+_)(\d+)(\.raw)")

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                # Blank line signals end of the current matrix block.
                if current_matrix and current_rock_id is not None:
                    rock_data.append((current_rock_id, current_header, current_matrix))
                    current_matrix = []     # reset for next block
                    current_rock_id = None
                    current_header = None
                continue

            # Process header lines that start with "# file"
            if line.lower().startswith("# file"):
                current_header = line  # Save the header line as is
                # Use header_pattern to update the rock id (group2) by adding 1.
                match = header_pattern.search(line)
                if match:
                    prefix = match.group(1)
                    orig_id = match.group(2)
                    extension = match.group(3)
                    # Increase the rock id by 1.
                    if plus_one: new_id = int(orig_id) + 1
                    else: new_id = int(orig_id)
                    current_rock_id = new_id
                    # Create the new filename portion.
                    new_filename = f"{prefix}{new_id}{extension}"
                    # Replace only the matched portion with new filename.
                    current_header = header_pattern.sub(new_filename, current_header, count=1)
                else:
                    current_rock_id = None
                continue  # Move to next lines that contain matrix row data

            cleaned_line = line.replace(',', ' ').replace(';', ' ')
            parts = cleaned_line.split()
            row = []
            for part in parts:
                try:
                    num = float(part)
                    row.append(num)
                except ValueError:
                    continue
            if row:
                current_matrix.append(row)

        # Add any final matrix after EOF.
        if current_matrix and current_rock_id is not None:
            rock_data.append((current_rock_id, current_header, current_matrix))
    # Sort data by rock_id.
    rock_data.sort(key=lambda x: x[0])
    rock_dict = OrderedDict()
    for rock_id, header, matrix in rock_data:
        np_matrix = np.array(matrix)
        if apply_increment:
            np_matrix = np_matrix + 1
        rock_dict[rock_id] = {"header": header,
                              "stiffness_matrix": np_matrix}
    return rock_dict

def write_matrices_to_dat(rock_dict, output_file):
    with open(output_file, 'w') as f:
        for rock_id, data in rock_dict.items():
            header = data["header"]
            matrix = data["stiffness_matrix"]
            f.write(f"{header}\n\n")
            for row in matrix:
                row_str = "   ".join(f"{num:10.4f}" for num in row)
                f.write(row_str + "\n")
            f.write("\n\n")  # Separate entries with extra blank lines

if __name__ == "__main__":
    input_file = '../model/size_75_stiffness.dat'
    rock_matrices = read_rock_matrices(input_file, apply_increment=False)
    output_file = '../size_75_stiffness.dat'
    write_matrices_to_dat(rock_matrices, output_file)
    print(f"Processed and reordered stiffness matrices saved to {output_file}")