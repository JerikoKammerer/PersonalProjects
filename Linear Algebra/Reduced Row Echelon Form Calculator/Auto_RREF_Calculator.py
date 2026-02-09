# RREF Auto Calculator
# This program computes the Reduced Row Echelon Form (RREF) of a given matrix using elementary row operations.

# Import the necessary library for matrix operations
import numpy as np

# Prompt user to determine if matrix is augement or not
def is_augmented():
    while True:
        response = input("Is the matrix augmented? (y/n): ").strip().lower()
        if response in ['yes', 'y']:
            return True
        elif response in ['no', 'n']:
            return False
        else:
            print("Please enter 'y' or 'n'.")

#Get dimensions of matrix from user
def get_dimensions():
    try:
        #Prompt user for number of rows and columns
        rows = int(input("Enter the number of rows: "))
        cols = int(input("Enter the number of columns: "))
        return rows, cols
    except ValueError:
        #Return error if non-integer values input
        print("Please enter valid integers for rows and columns.")
        return get_dimensions()
    
# Get matrix entries from user
def get_matrix(rows, cols):
    # Prompt user to enter matrix entries
    print("Enter the matrix entries row by row (separated by spaces):")
    matrix = []
    for i in range(rows):
        while True:
            try:
                # Read a row of input, split it into values, and convert to integers
                row = list(map(float, input(f"Row {i + 1}: ").split()))
                if len(row) != cols:
                    # If the number of values does not match the number of columns, prompt again
                    print(f"Please enter exactly {cols} values.")
                    continue
                matrix.append(row) #Adds user's row inputs into matrix
                break
            except ValueError:
                #Return error if non-integer values input
                print("Please enter valid numbers.")
    return np.array(matrix, dtype=float) #Convert list of lists to a numpy array for easier manipulation

# Function to perform row swapping
def swap(matrix, i, j):
    # Swap rows i and j in the matrix
    matrix[[i, j]] = matrix[[j, i]]
    return matrix

def scale(matrix, i, k):
    # Scale row i by a factor of k
    matrix[i] = k * matrix[i]
    return matrix

def shear(matrix, i, j, k):
    # Add k times row j to row i
    matrix[i] = matrix[i] + k * matrix[j]
    return matrix

def display_matrix(matrix, is_aug):
    # Display the matrix in a readable format
    def fmt(x):
        return str(int(round(x))) if np.isclose(x, round(x)) else str(x)
    
    print("Current Matrix:")
    for row in matrix:
        if is_aug:
            if len(row) > 1:
                left = " ".join(fmt(v) for v in row[:-1])  # All elements except the last one
                right = fmt(row[-1])  # The last element
                print(f"{left} | {right}")  # Print the row with a vertical bar separating the last element
            else:
                print(f"[{fmt(row[0])}]")  # If there's only one element, just print it
        else:
            print(" ".join(fmt(v) for v in row))

def rref(matrix):
    rows, cols = matrix.shape
    lead = 0 # Initialize the leading column index to 0
    for c in range(cols):
        if lead >= rows:
            # If the leading row index exceeds the number of rows, return the matrix
            return matrix 
        if c > lead:
            i = lead # Initialize the row index to the current leading row
            lead = c # Update the leading column index to the current column
        else:
            i = lead # Initialize the row index to the current leading row3
        while c == lead: # Loop until we find a leading entry in the current column
            if matrix[i][c] == 0: # If the leading entry is zero, find a row below it with a non-zero entry
                for r in range(i + 1, rows):
                    if matrix[r][c] != 0:
                        matrix = swap(matrix, i, r) # Swap the current row with the row containing the non-zero entry
                        print(f"Swapped row {i} with row {r}:")
                        display_matrix(matrix, False)
                        break
                    else:
                        c += 1 # Move to the next column if the current column is zero
                        if c >= cols:
                            return matrix
            elif matrix[i][c] != 1 and matrix[i][c] != 0: # If the leading entry isn't 1 or 0, scale the row to make it 1
                scale_factor = 1 / matrix[i][c] # Calculate the factor to scale the row to make the leading entry 1
                matrix = scale(matrix, i, scale_factor) # Scale the current row to make the leading entry 1
                print(f"Scaled row {i} by {scale_factor} to make leading entry 1:")
                display_matrix(matrix, False)
            else: # If the leading entry is 1, eliminate the entries below it
                for r in range(rows):
                    if r != i: # For each row other than the current row, eliminate the leading entry
                        shear_factor = -matrix[r][c] # Calculate the factor to eliminate the leading entry
                        matrix = shear(matrix, r, i, shear_factor) # Add a multiple of the current row to eliminate the leading entry
                        print(f"Added {shear_factor} times row {i} to row {r} to eliminate leading entry:")
                        display_matrix(matrix, False)
                lead += 1 # Move to the next leading column
    return matrix

def rref_aug(matrix):
    rows, cols = matrix.shape
    lead = 0 # Initialize the leading column index to 0
    for c in range(cols-1):
        if lead >= rows:
            # If the leading row index exceeds the number of rows, return the matrix
            return matrix 
        if c > lead:
            i = lead # Initialize the row index to the current leading row
            lead = c # Update the leading column index to the current column
        else:
            i = lead # Initialize the row index to the current leading row3
        while c == lead: # Loop until we find a leading entry in the current column
            if matrix[i][c] == 0: # If the leading entry is zero, find a row below it with a non-zero entry
                for r in range(i + 1, rows):
                    if matrix[r][c] != 0:
                        matrix = swap(matrix, i, r) # Swap the current row with the row containing the non-zero entry
                        print(f"Swapped row {i} with row {r}:")
                        display_matrix(matrix, True)
                        break
                    else:
                        c += 1 # Move to the next column if the current column is zero
                        if c >= (cols-1):
                            return matrix
            elif matrix[i][c] != 1 and matrix[i][c] != 0: # If the leading entry isn't 1 or 0, scale the row to make it 1
                scale_factor = 1 / matrix[i][c] # Calculate the factor to scale the row to make the leading entry 1
                matrix = scale(matrix, i, scale_factor) # Scale the current row to make the leading entry 1
                print(f"Scaled row {i} by {scale_factor} to make leading entry 1:")
                display_matrix(matrix, True)
            else: # If the leading entry is 1, eliminate the entries below it
                for r in range(rows):
                    if r != i: # For each row other than the current row, eliminate the leading entry
                        shear_factor = -matrix[r][c] # Calculate the factor to eliminate the leading entry
                        matrix = shear(matrix, r, i, shear_factor) # Add a multiple of the current row to eliminate the leading entry
                        print(f"Added {shear_factor} times row {i} to row {r} to eliminate leading entry:")
                        display_matrix(matrix, True)
                lead += 1 # Move to the next leading column
    return matrix

def main():
    is_aug = is_augmented()  # Determine if the matrix is augmented
    rows, cols = get_dimensions()  # Get the dimensions of the matrix from the user
    matrix = get_matrix(rows, cols)  # Get the matrix entries from the user
    print("Original Matrix:")
    display_matrix(matrix, is_aug)  # Display the original matrix
    if is_aug:
        rref_matrix = rref_aug(matrix)  # Compute the RREF of the augmented matrix
    else:
        rref_matrix = rref(matrix)  # Compute the RREF of the matrix
    print("Reduced Row Echelon Form:")
    display_matrix(rref_matrix, is_aug)  # Display the RREF of the matrix

    # Automatically calculate the final matrix without user input
    # The rref function already handles the necessary operations
    # No need for further user interaction

try:
    while True:
        main()
except KeyboardInterrupt:
    print("\nProgram terminated by user.")


