from flask import Flask, request, jsonify
import pandas as pd
import joblib
from flask_cors import CORS
import numpy as np
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

app = Flask(__name__)
CORS(app)

def create_preprocessor(categorical_features, numerical_features):
    """Create a preprocessor pipeline for the given features"""
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

# Model configuration with built-in preprocessing
LEAGUE_MODELS = {
    'epl': {
        'winner': {
            'model': joblib.load('./app/models/epl_winner.pkl'),
            'categorical_features': ['strHomeTeam', 'strAwayTeam'],
            'numerical_features': [
                'homeGD', 'homeWinRate', 'homeClassement', 'predictedIntHomeScore', 'predictedResult',
                'awayGD', 'awayWinRate', 'awayClassement', 'predictedIntAwayScore'
            ]
        },
        'goals': {
            'model': joblib.load('./app/models/epl_goals.pkl'),
            'categorical_features': ['strHomeTeam', 'strAwayTeam'],
            'numerical_features': [
                'homeGD', 'homeWinRate', 'homeClassement', 'lastHome5GamesScore',
                'awayGD', 'awayWinRate', 'awayClassement', 'lastAway5GamesScore'
            ]
        }
    }
    # Add other leagues similarly
}

# Initialize preprocessors for each model
for league, models in LEAGUE_MODELS.items():
    for model_type, config in models.items():
        config['preprocessor'] = create_preprocessor(
            config['categorical_features'],
            config['numerical_features']
        )

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'version': '1.2.2',
        'availableLeagues': [k.replace('_', ' ') for k in LEAGUE_MODELS.keys()],
        'timestamp': datetime.now().isoformat()
    })
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        match_data = data['match_data']
        
        # Create initial DataFrame with basic features
        features = {
            'homeGD': [match_data['homeGD']],
            'homeWinRate': [match_data['homeWins'] / match_data['homePlayed']],
            'homeClassement': [match_data['homeRank']],
            'lastHome5GamesScore': [match_data['lastHome5GamesScore']],
            'strHomeTeam': [match_data['strHomeTeam']],
            'awayGD': [match_data['awayGD']],
            'awayWinRate': [match_data['awayWins'] / match_data['awayPlayed']],
            'awayClassement': [match_data['awayRank']],
            'lastAway5GamesScore': [match_data['lastAway5GamesScore']],
            'strAwayTeam': [match_data['strAwayTeam']]
        }
        match_df = pd.DataFrame(features)

        # Get regression model predictions (scores)
        home_features = ['homeGD', 'homeWinRate', 'homeClassement', 'lastHome5GamesScore', 'strHomeTeam']
        away_features = ['awayGD', 'awayWinRate', 'awayClassement', 'lastAway5GamesScore', 'strAwayTeam']
        
        X_home = match_df[home_features]
        X_away = match_df[away_features]
        X_full = X_home.join(X_away)
        
        regression_pred = LEAGUE_MODELS['epl']['goals']['model'].predict(X_full)
        match_df['predictedIntHomeScore'] = regression_pred[:, 0]
        match_df['predictedIntAwayScore'] = regression_pred[:, 1]

        # Calculate predicted result (0=away, 1=draw, 2=home)
        match_df['predictedResult'] = match_df.apply(
            lambda row: 2 if row['predictedIntHomeScore'] > row['predictedIntAwayScore']
            else 0 if row['predictedIntHomeScore'] < row['predictedIntAwayScore']
            else 1,
            axis=1
        )

        # Prepare classification features
        classification_features = [
            'homeGD', 'homeWinRate', 'homeClassement', 'predictedIntHomeScore', 'predictedResult',
            'awayGD', 'awayWinRate', 'awayClassement', 'predictedIntAwayScore'
        ]
        X_class = match_df[classification_features]

        # Get final winner prediction
        winner_pred = LEAGUE_MODELS['epl']['winner']['model'].predict(X_class)[0]
        
        # Convert to frontend-friendly format
        home_goals = float(match_df['predictedIntHomeScore'].values[0])
        away_goals = float(match_df['predictedIntAwayScore'].values[0])
        winner = ['away', 'draw', 'home'][winner_pred]  # Map 0,1,2 to strings

        return jsonify({
            'combined_prediction': {
                'score': f"{home_goals:.1f}-{away_goals:.1f}",
                'winner': winner,
                'probability': {
                    'home': 60 if winner == 'home' else 30,
                    'draw': 20 if winner == 'draw' else 10,
                    'away': 60 if winner == 'away' else 30
                },
                'expected_goals': {
                    'home': home_goals,
                    'away': away_goals
                },
                'both_teams_to_score': bool(home_goals >= 1 and away_goals >= 1),
                'over_under': round(home_goals + away_goals, 1)
            }
        })

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500
if __name__ == '__main__':
    app.run(debug=True)