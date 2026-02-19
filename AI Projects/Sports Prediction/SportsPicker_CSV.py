# Reads CSV data of two NFL/CFB opposing teams and their stats, and predicts the winner using a pre-trained model. This is a simplified version without API calls.
# OR
# Reads CSV data of one NFL/CFB player's stats and their opponents' stats, and predicts over/under on a specific statistic using a pre-trained model. This is a simplified version without API calls.

import pandas as pd # Reads csv
import numpy as np # For numerical operations
from sklearn.model_selection import train_test_split # For splitting data into training and testing sets
from sklearn.linear_model import LogisticRegression # For logistic regression model
from sklearn.metrics import accuracy_score # For evaluating the model

# NFL Stats File Path
STATS_CSV_PATH = r"C:\Users\jerik\.vscode\PersonalProjects\AI Projects\Sports Prediction\nfl_team_stats_2002-2025.csv" # Path to the CSV file containing the stats data

# Method to read CSV Data using pandas
def read_csv_data(file_path):
    try:
        dataFrame = pd.read_csv(file_path)
        print(f"Data successfully read from {file_path}")
        return dataFrame
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: The file {file_path} is empty.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

# Preprocess data for model training
def preprocess_data(df):
    # Example preprocessing: Fill missing values, encode categorical variables, etc.
    features = ['score_away', 'score_home', 'yards_away', 'yards_home',
                 'pass_yards_away', 'pass_yards_home', 'rush_yards_away', 'rush_yards_home',
                 'pen_yards_away', 'pen_yards_home', 'fumbles_away', 'fumbles_home', 'interceptions_away', 'interceptions_home',
                 'def_st_td_away', 'def_st_td_home', 'possession_away', 'possession_home'] # Relevant Stats as features
    
    # Define input & output variables
    X = df[features] # Input features
    y = df['target_winner'] # Target variable (1 for home team win, 0 for away team win)

    # Split into Training and Testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    #Check for accuracy on the test set
    y_pred = model.predict(X_test)
    print("Model Accuracy: ", accuracy_score(y_test, y_pred))

    return model



