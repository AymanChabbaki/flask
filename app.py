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
            'model': joblib.load('./models/epl_winner.pkl'),
            'categorical_features': ['strHomeTeam', 'strAwayTeam'],
            'numerical_features': [
                'homeGD', 'homeWinRate', 'homeClassement', 'predictedIntHomeScore', 'predictedResult',
                'awayGD', 'awayWinRate', 'awayClassement', 'predictedIntAwayScore'
            ]
        },
        'goals': {
            'model': joblib.load('./models/epl_goals.pkl'),
            'categorical_features': ['strHomeTeam', 'strAwayTeam'],
            'numerical_features': [
                'homeGD', 'homeWinRate', 'homeClassement', 'lastHome5GamesScore',
                'awayGD', 'awayWinRate', 'awayClassement', 'lastAway5GamesScore'
            ]
        }
    },
        'inwi': {
        'winner': {
            'model': joblib.load('./models/inwi_winner.pkl'),
            'categorical_features': ['strHomeTeam', 'strAwayTeam'],
            'numerical_features': [
                'homeGD', 'homeWinRate', 'homeClassement', 'predictedIntHomeScore', 'predictedResult',
                'awayGD', 'awayWinRate', 'awayClassement', 'predictedIntAwayScore'
            ]
        },
        'goals': {
            'model': joblib.load('./models/inwi_goals.pkl'),
            'categorical_features': ['strHomeTeam', 'strAwayTeam'],
            'numerical_features': [
                'homeGD', 'homeWinRate', 'homeClassement', 'lastHome5GamesScore',
                'awayGD', 'awayWinRate', 'awayClassement', 'lastAway5GamesScore'
            ]
        }
    },
        'liga': {
        'winner': {
            'model': joblib.load('./models/liga_winner.pkl'),
            'categorical_features': ['strHomeTeam', 'strAwayTeam'],
            'numerical_features': [
                'homeGD', 'homeWinRate', 'homeClassement', 'predictedIntHomeScore', 'predictedResult',
                'awayGD', 'awayWinRate', 'awayClassement', 'predictedIntAwayScore'
            ]
        },
        'goals': {
            'model': joblib.load('./models/liga_goals.pkl'),
            'categorical_features': ['strHomeTeam', 'strAwayTeam'],
            'numerical_features': [
                'homeGD', 'homeWinRate', 'homeClassement', 'lastHome5GamesScore',
                'awayGD', 'awayWinRate', 'awayClassement', 'lastAway5GamesScore'
            ]
        }
    },
        'bundesliga': {
        'winner': {
            'model': joblib.load('./models/bundesliga_winner.pkl'),
            'categorical_features': ['strHomeTeam', 'strAwayTeam'],
            'numerical_features': [
                'homeGD', 'homeWinRate', 'homeClassement', 'predictedIntHomeScore', 'predictedResult',
                'awayGD', 'awayWinRate', 'awayClassement', 'predictedIntAwayScore'
            ]
        },
        'goals': {
            'model': joblib.load('./models/bundesliga_goals.pkl'),
            'categorical_features': ['strHomeTeam', 'strAwayTeam'],
            'numerical_features': [
                'homeGD', 'homeWinRate', 'homeClassement', 'lastHome5GamesScore',
                'awayGD', 'awayWinRate', 'awayClassement', 'lastAway5GamesScore'
            ]
        }
    },
        'ligue': {
        'winner': {
            'model': joblib.load('./models/ligue_winner.pkl'),
            'categorical_features': ['strHomeTeam', 'strAwayTeam'],
            'numerical_features': [
                'homeGD', 'homeWinRate', 'homeClassement', 'predictedIntHomeScore', 'predictedResult',
                'awayGD', 'awayWinRate', 'awayClassement', 'predictedIntAwayScore'
            ]
        },
        'goals': {
            'model': joblib.load('./models/ligue_goals.pkl'),
            'categorical_features': ['strHomeTeam', 'strAwayTeam'],
            'numerical_features': [
                'homeGD', 'homeWinRate', 'homeClassement', 'lastHome5GamesScore',
                'awayGD', 'awayWinRate', 'awayClassement', 'lastAway5GamesScore'
            ]
        }
    },
        'serie': {
        'winner': {
            'model': joblib.load('./models/serie_winner.pkl'),
            'categorical_features': ['strHomeTeam', 'strAwayTeam'],
            'numerical_features': [
                'homeGD', 'homeWinRate', 'homeClassement', 'predictedIntHomeScore', 'predictedResult',
                'awayGD', 'awayWinRate', 'awayClassement', 'predictedIntAwayScore'
            ]
        },
        'goals': {
            'model': joblib.load('./models/serie_goals.pkl'),
            'categorical_features': ['strHomeTeam', 'strAwayTeam'],
            'numerical_features': [
                'homeGD', 'homeWinRate', 'homeClassement', 'lastHome5GamesScore',
                'awayGD', 'awayWinRate', 'awayClassement', 'lastAway5GamesScore'
            ]
        }
    }
}

# Initialize preprocessors for each model
for league, models in LEAGUE_MODELS.items():
    for model_type, config in models.items():
        config['preprocessor'] = create_preprocessor(
            config['categorical_features'],
            config['numerical_features']
        )

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        match_data = data['match_data']
        
        # Create initial DataFrame with basic features
        features = {
            'homeGD': [match_data['homeGD']],
            'homeWinRate': [match_data['homeWins'] / (match_data['homePlayed'])],
            'homeClassement': [match_data['homeRank']],
            'lastHome5GamesScore': [match_data['lastHome5GamesScore']],
            'strHomeTeam': [match_data['strHomeTeam']],
            'awayGD': [match_data['awayGD']],
            'awayWinRate': [match_data['awayWins'] / (match_data['awayPlayed'])],
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

        
        total_goals = match_df['predictedIntHomeScore'].values[0] + match_df['predictedIntAwayScore'].values[0]
        
        # Calculate initial probabilities based on predicted goals
        home_proba = float((match_df['predictedIntHomeScore'].values[0] / total_goals) * 100) if total_goals > 0 else 33.3
        away_proba = float((match_df['predictedIntAwayScore'].values[0] / total_goals) * 100) if total_goals > 0 else 33.3
        
        # Get final winner prediction (0=away, 1=draw, 2=home)
        winner_pred = LEAGUE_MODELS['epl']['winner']['model'].predict(X_class)[0]
        
        # Convert to frontend-friendly format
        home_goals = float(match_df['predictedIntHomeScore'].values[0])
        away_goals = float(match_df['predictedIntAwayScore'].values[0])
        winner = ['away', 'draw', 'home'][winner_pred]
        
        # Calculate draw probability based on the classifier's prediction probabilities
        # First get the probabilities for each class from the classifier
        if hasattr(LEAGUE_MODELS['epl']['winner']['model'], 'predict_proba'):
            class_probs = LEAGUE_MODELS['epl']['winner']['model'].predict_proba(X_class)[0]
            # Assuming the classes are ordered as [away, draw, home]
            away_proba_class = class_probs[0] * 100
            draw_proba = class_probs[1] * 100
            home_proba_class = class_probs[2] * 100
            
            # Combine the goals-based and classifier probabilities (you can adjust weights)
            home_proba = round((home_proba * 0.3 + home_proba_class * 0.7), 1)
            away_proba = round((away_proba * 0.3 + away_proba_class * 0.7), 1)
            draw_proba = round(draw_proba, 1)
        else:
            # If classifier doesn't provide probabilities, use a simple approach
            draw_proba = 100 - home_proba - away_proba
            if draw_proba < 0:  # Ensure probabilities make sense
                total = home_proba + away_proba
                home_proba = round((home_proba / total) * 100, 1)
                away_proba = round((away_proba / total) * 100, 1)
                draw_proba = 0
        
        # Ensure probabilities sum to 100 (handle any rounding errors)
        total = home_proba + away_proba + draw_proba
        if total != 100:
            # Distribute the difference to the most probable outcome
            diff = 100 - total
            if home_proba >= away_proba and home_proba >= draw_proba:
                home_proba += diff
            elif away_proba >= home_proba and away_proba >= draw_proba:
                away_proba += diff
            else:
                draw_proba += diff
        
        # Final rounding
        home_proba = round(home_proba, 1)
        away_proba = round(away_proba, 1)
        draw_proba = round(draw_proba, 1)

        return jsonify({
            'combined_prediction': {
                'score': f"{home_goals:.1f}-{away_goals:.1f}",
                'winner': winner,
                'probability': {
                    'home': home_proba,
                    'away': away_proba,
                    'draw': draw_proba
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
    app.run()
