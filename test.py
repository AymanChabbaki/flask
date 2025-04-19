import joblib
#  'winner': {
#             'model': joblib.load('models/epl_winner.pkl'),
#             'categorical_features': ['strHomeTeam', 'strAwayTeam'],
#             'numerical_features': [
#                 'homeGD', 'homeWinRate', 'homeClassement', 'lastHome5GamesScore',
#                 'awayGD', 'awayWinRate', 'awayClassement', 'lastAway5GamesScore',
#                 'homeScored', 'homeWins'
#             ]
#         },
#         'goals': {
#             'model': joblib.load('models/epl_goals.pkl'),
#             'categorical_features': ['strHomeTeam', 'strAwayTeam'],
#             'numerical_features': [
#                 'homeGD', 'homeWinRate', 'homeClassement', 'lastHome5GamesScore',
#                 'awayGD', 'awayWinRate', 'awayClassement', 'lastAway5GamesScore'
#             ]
#         }
import pandas as pd
import joblib

# Load the model
regression_model = joblib.load('C:/Users/HP/Desktop/L.E.AI/S6/AppAI/ProjetAppAI/my-react-app/api/models/epl_goals.pkl')

# Sample data for testing
match_data={
        "strHomeTeam": "Arsenal",
        "strAwayTeam": "Chelsea",
        "homeGoalsFor": 30,
        "homeGoalsAgainst": 15,
        "homeWins": 8,
        "homeGD": 43,
        "homePlayed": 12,
        "homeRank": 3,
        "awayGoalsFor": 25,
        "awayGoalsAgainst": 20,
        "awayWins": 6,
        "awayGD":30,
        "awayPlayed": 12,
        "awayRank": 5
    }
data = {
            'homeGD': [match_data['homeGD']],
            'homeWinRate': [match_data['homeWins'] / match_data['homePlayed']],
            'homeClassement': [match_data['homeRank']],
            'lastHome5GamesScore': [(match_data['homeGoalsFor'] / match_data['homePlayed']) * 5],
            'strHomeTeam': [match_data['strHomeTeam']],
            'awayGD': [match_data['awayGD']],
            'awayWinRate': [match_data['awayWins'] / match_data['awayPlayed']],
            'awayClassement': [match_data['awayRank']],
            'lastAway5GamesScore': [(match_data['awayGoalsFor'] / match_data['awayPlayed']) * 5],
            'strAwayTeam': [match_data['strAwayTeam']]
        }
# Create a DataFrame
df = pd.DataFrame(data)

# Select features
home_features = ['homeGD', 'homeWinRate', 'homeClassement', 'lastHome5GamesScore', 'strHomeTeam']
away_features = ['awayGD', 'awayWinRate', 'awayClassement', 'lastAway5GamesScore', 'strAwayTeam']

# Combine features
X_home = df[home_features]
X_away = df[away_features]
X_full = X_home.join(X_away)

# Predict
predictions = regression_model.predict(X_full)

print(predictions)
