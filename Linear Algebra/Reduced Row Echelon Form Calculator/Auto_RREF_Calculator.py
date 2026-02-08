# RREF Auto Calculator
# This program computes the Reduced Row Echelon Form (RREF) of a given matrix using elementary row operations.

# Import the necessary library for matrix operations
import numpy as np

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
                row = list(map(int, input(f"Row {i + 1}: ").split()))
                if len(row) != cols:
                    # If the number of values does not match the number of columns, prompt again
                    print(f"Please enter exactly {cols} values.")
                    continue
                matrix.append(row) #Adds user's row inputs into matrix
                break
            except ValueError:
                #Return error if non-integer values input
                print("Please enter valid numbers.")
    return np.array(matrix) #Convert list of lists to a numpy array for easier manipulation

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

def display_matrix(matrix):
    # Display the matrix in a readable format
    print("Current Matrix:")
    for row in matrix:
        print(" ".join(map(str, row)))

def rref(matrix):
    rows, cols = matrix.shape # Get the number of rows and columns in the matrix
    lead = 0 # Initialize the leading column index to 0
    for r in range(rows):
        if lead >= cols:
            return matrix # If the leading column index exceeds the number of columns, return the matrix
        i = r # Initialize the row index to the current row
        while matrix[i][lead] == 0: # If the leading entry is zero, find a row below it with a non-zero entry
            i += 1
            if i == rows: # If we have checked all rows and found no non-zero entry, move to the next column
                i = r
                lead += 1
                if cols == lead: # If the leading column index exceeds the number of columns,
                    return matrix
        matrix = swap(matrix, i, r) # Swap the current row with the row containing the non-zero entry
        lv = matrix[r][lead] # Get the leading value of the current row
        matrix = scale(matrix, r, 1 / lv) # Scale the current row to make the leading entry 1
        for i in range(rows):
            if i != r: # For each row other than the current row, eliminate the leading entry
                lv = matrix[i][lead] # Get the leading value of the current row
                matrix = shear(matrix, i, r, -lv) # Add a multiple of the current row to eliminate the leading entry
        lead += 1 # Move to the next leading column
    return matrix

def main():
    rows, cols = get_dimensions()  # Get the dimensions of the matrix from the user
    matrix = get_matrix(rows, cols)  # Get the matrix entries from the user
    print("Original Matrix:")
    display_matrix(matrix)  # Display the original matrix
    rref_matrix = rref(matrix)  # Compute the RREF of the matrix
    print("Reduced Row Echelon Form:")
    display_matrix(rref_matrix)  # Display the RREF of the matrix

    # Automatically calculate the final matrix without user input
    # The rref function already handles the necessary operations
    # No need for further user interaction

try:
    while True:
        main()
except KeyboardInterrupt:
    print("\nProgram terminated by user.")


