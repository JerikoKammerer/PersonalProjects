# Elementary Row Operations Calculator for Matrices
# User inputs a matrix and selects from follow EROs:
# 1. Row Swapping (Ri <-> Rj)
# 2. Row Scaling (k * Ri)
# 3. Row Addition (Ri + k * Rj -> Ri)

# Import the numpy library for matrix operations
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

def main():
    #Main function to run the ERO calculator
    rows, cols = get_dimensions() #Get dimensions of matrix from user
    matrix = get_matrix(rows, cols) #Get matrix entries from user
    display_matrix(matrix) #Display the initial matrix
    while True:
        print("\nSelect an elementary row operation:")
        print("1. Row Swapping (Ri <-> Rj)")
        print("2. Row Scaling (k * Ri)")
        print("3. Row Addition (Ri + k * Rj -> Ri)")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            i = int(input("Enter the first row index to swap (0-based): "))
            j = int(input("Enter the second row index to swap (0-based): "))
            matrix = swap(matrix, i, j)
            display_matrix(matrix)
        
        elif choice == '2':
            i = int(input("Enter the row index to scale (0-based): "))
            k = float(input("Enter the scaling factor: "))
            matrix = scale(matrix, i, k)
            display_matrix(matrix)
        
        elif choice == '3':
            i = int(input("Enter the target row index (0-based): "))
            j = int(input("Enter the source row index (0-based): "))
            k = float(input("Enter the scaling factor for the source row: "))
            matrix = shear(matrix, i, j, k)
            display_matrix(matrix)
        
        elif choice == '4':
            print("Exiting the calculator.")
            break
        
        else:
            print("Invalid choice. Please select a valid option.")

try:
    while True:
        main()
except KeyboardInterrupt:
    print("\nCalculator terminated by user.")
