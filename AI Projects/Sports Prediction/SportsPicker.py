# Reads Data from two different teams to predict the winner of a match/game between them
# OR
# Reads defensive data from a team and offensive data from an individual player to predict higher/lower on statistics for the player

#import libraries
import pandas as pd # Reads csv
import numpy as np # For numerical operations
from sklearn.model_selection import train_test_split # For splitting data into training and testing sets
from sklearn.linear_model import LogisticRegression # For logistic regression model
from sklearn.metrics import accuracy_score # For evaluating the model
import requests # For making API calls to get data
import http.client # For handling HTTP exceptions

#API Constants
API_KEY = "596a0705df0ab87de4e3edf7cd1b2829"
conn = http.client.HTTPSConnection("v1.american-football.api-sports.io")

#Define header for auth
headers = {
    "x-apisports-key": API_KEY
}

# Team ID Dictionary for reference

#Method to get team IDs from the API
def get_nfl_teams():
    conn.request("GET", "/teams?id=1", headers=headers)
    res = conn.getresponse()
    if res.status == 200:
        data = res.read()
        teams = pd.DataFrame(data)
        print(teams)
    else:
        print(f"Error fetching teams: {res.status}")

#Method to get team statistics from the API
def get_team_stats(team_id):
    url = f"{BASE_URL}/teams/statistics?team={team_id}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching data for team {team_id}: {response.status_code}")
        return None
    
#Method to get player statistics from the API
def get_player_stats(player_id):
    url = f"{BASE_URL}/players/statistics?player={player_id}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching data for player {player_id}: {response.status_code}")
        return None
    
try:
    get_nfl_teams()
except Exception as e:
    print(f"An error occurred: {e}")