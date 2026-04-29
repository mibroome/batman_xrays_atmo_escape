# Efficiently check if the 3rd column in a large text file is all 1's
def check_third_column_all_ones(filename):
    with open(filename, 'r') as f:
        for i, line in enumerate(f, 1):
            # Skip empty lines
            if not line.strip():
                continue
            parts = line.strip().split(',')
            if len(parts) < 3:
                print(f"Line {i} does not have 3 columns: {line.strip()}")
                return False
            if parts[2] != '1':
                print(f"First non-1 found at line {i}: {parts[2]}")
                return False
    print("All values in the 3rd column are 1.")
    return True

# Example usage:
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python reader.py <filename>")
    else:
        check_third_column_all_ones(sys.argv[1])
