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
import json # For parsing JSON data from API responses

#API Constants
API_KEY = "596a0705df0ab87de4e3edf7cd1b2829"
conn = http.client.HTTPSConnection("v1.american-football.api-sports.io")

#Define header for auth
headers = {
    "x-apisports-key": API_KEY
}
# Team Code Dictionary for reference
NFL_TEAMS_NAME_TO_CODE = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS"
}  
# Team ID Dictionary for reference
NFL_TEAMS_CODE_TO_ID = {
    "ARI": 11,
    "ATL": 8,
    "BAL": 5,
    "BUF": 20,
    "CAR": 19,
    "CHI": 16,
    "CIN": 10,
    "CLE": 9,
    "DAL": 29,
    "DEN": 28,
    "DET": 7,
    "GB": 15,
    "HOU": 26,
    "IND": 21,
    "JAX": 2,
    "KC": 17,
    "LV": 1,
    "LAC": 30,
    "LA": 31,
    "MIA": 25,
    "MIN": 32,
    "NE": 3,
    "NO": 27,
    "NYG": 4,
    "NYJ": 13,
    "PHI": 12,
    "PIT": 22,
    "SF": 14,
    "SEA": 23,
    "TB": 24,
    "TEN": 6,
    "WAS": 18
}

# Method to Print Team Codes for reference
def print_team_codes():
    print("\n--- NFL Team Codes ---\n")
    for team, code in NFL_TEAMS_NAME_TO_CODE.items():
        print(f"{team}: {code}")
#Method to get team IDs from the API
"""
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
"""
try:
    print("Fetching NFL teams...")
    conn.request("GET", "/teams?league=1&season=2024", headers=headers)
    res = conn.getresponse()
    data = res.read()
    json_data = data.decode("utf-8")
    parsed_data = json.loads(json_data)
    if not parsed_data.get('response'):
        print("Warning: API returned 0 results. Check your Season or League ID.")
        print(f"API Message: {parsed_data.get('errors')}")
    else:
        # If data exists, then we normalize
        teams_df = pd.json_normalize(parsed_data['response'])
        
        try:
            subset = teams_df[['id', 'name', 'code']]
            print("\n --- NFL Teams ---\n")
            print(subset.head(32))
        except KeyError as ke:
            print(f"KeyError: {ke}. Available columns are: {teams_df.columns.tolist()}")
                  
except Exception as e:
    print(f"An error occurred: {e}")