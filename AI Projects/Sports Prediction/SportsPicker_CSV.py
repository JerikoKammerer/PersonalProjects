# Reads CSV data of two NFL/CFB opposing teams and their stats, and predicts the winner using a pre-trained model. This is a simplified version without API calls.
# OR
# Reads CSV data of one NFL/CFB player's stats and their opponents' stats, and predicts over/under on a specific statistic using a pre-trained model. This is a simplified version without API calls.

import pandas as pd # Reads csv
import numpy as np # For numerical operations
from sklearn.model_selection import train_test_split # For splitting data into training and testing sets
from sklearn.linear_model import LogisticRegression # For logistic regression model
from sklearn.metrics import accuracy_score # For evaluating the model
from sklearn.preprocessing import StandardScaler # For feature scaling

# NFL Stats File Path
STATS_CSV_PATH = r"C:\Users\jerik\.vscode\PersonalProjects\AI Projects\Sports Prediction\nfl_team_stats_2002-2025.csv" # Path to the CSV file containing the stats data


# Scale inputs since features may be on different scales
scaler = StandardScaler()

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
    
# Calculate season averages for each team and merge with original data
def calculate_season_averages(df):
    # Ensure date is in datetime format and sort chronologically
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['season', 'date'])

    # Define the stats we want to average
    stats_cols = [
        'score', 'yards', 'pass_yards', 'rush_yards', 
        'pen_yards', 'fumbles', 'interceptions', 'def_st_td'
    ]

    # Create two dataframes: one for home team view, one for away team view
    # This aligns the data so every row represents ONE team's performance
    home_df = df[['season', 'home_team'] + [f"{c}_home" for c in stats_cols]].rename(
        columns=lambda x: x.replace('_home', '') if '_home' in x else x
    ).rename(columns={'home_team': 'team'})

    away_df = df[['season', 'away_team'] + [f"{c}_away" for c in stats_cols]].rename(
        columns=lambda x: x.replace('_away', '') if '_away' in x else x
    ).rename(columns={'away_team': 'team'})

    # Combine them to get a master list of all team performances
    all_games = pd.concat([home_df, away_df]).sort_index()

    # Calculate rolling average (expanding window)
    # .shift(1) is crucial: it ensures we use stats PRIOR to the current game
    averages = all_games.groupby(['season', 'team'])[stats_cols].apply(
        lambda x: x.expanding().mean().shift(1)
    ).reset_index(level=[0,1], drop=True)

    # Rename columns to 'avg_score', etc.
    averages.columns = ['avg_' + col for col in averages.columns]

    return pd.concat([all_games[['season', 'team']], averages], axis=1).dropna()

# Preprocess data for model training
def preprocess_data(df):
    # Example preprocessing: Fill missing values, encode categorical variables, etc.
    features = ['score_away', 'score_home', 'yards_away', 'yards_home',
                 'pass_yards_away', 'pass_yards_home', 'rush_yards_away', 'rush_yards_home',
                 'pen_yards_away', 'pen_yards_home', 'fumbles_away', 'fumbles_home', 'interceptions_away', 'interceptions_home',
                 'def_st_td_away', 'def_st_td_home'] # Relevant Stats as features
    
    # Define input & output variables
    X = df[features] # Input features
    y = (df['score_home'] > df['score_away']).astype(int) # Target variable (1 for home team win, 0 for away team win)

    # Split into Training and Testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit only on training data, then transform both
    X_trained_scale = scaler.fit_transform(X_train)
    X_test_scale = scaler.transform(X_test)

    # Initialize and train the model
    model = LogisticRegression()
    model.fit(X_trained_scale, y_train)

    #Check for accuracy on the test set
    y_pred = model.predict(X_test)
    print("Model Accuracy: ", accuracy_score(y_test, y_pred))

    return model

# Predict New Game that hasn't happened yet
def predict_new_game(model, new_game_data):
    # new_game_data should be a dictionary with the same features as the training data
    features = ['score_away', 'score_home', 'yards_away', 'yards_home',
                'pass_yards_away', 'pass_yards_home', 'rush_yards_away', 'rush_yards_home',
                'pen_yards_away', 'pen_yards_home', 'fumbles_away', 'fumbles_home', 'interceptions_away', 'interceptions_home',
                'def_st_td_away', 'def_st_td_home'] # Relevant Stats as features
    
    # Convert new_game_data to DataFrame
    new_game_df = pd.DataFrame([new_game_data], columns=features)

    # Scale data
    new_game_scaled = scaler.transform(new_game_df)

    # Predict the winner (1 for home team win, 0 for away team win)
    prediction = model.predict(new_game_scaled)
    probability = model.predict_proba(new_game_scaled)
    if prediction[0] == 1:
        print(f"Predicted Winner: Home Team with probability {probability[0][1]:.2f}")
    else:
        print(f"Predicted Winner: Away Team with probability {probability[0][0]:.2f}")


