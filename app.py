from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import requests
import io
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import uvicorn

app = FastAPI(title="Football Prediction API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# -----------------------------
# Pydantic models for request/response
# -----------------------------
class MatchData(BaseModel):
    homeGD: float
    homeWins: int
    homePlayed: int
    homeRank: int
    lastHome5GamesScore: float
    strHomeTeam: str
    awayGD: float
    awayWins: int
    awayPlayed: int
    awayRank: int
    lastAway5GamesScore: float
    strAwayTeam: str

class PredictionRequest(BaseModel):
    match_data: MatchData

class ProbabilityResponse(BaseModel):
    home: float
    away: float
    draw: float

class ExpectedGoalsResponse(BaseModel):
    home: float
    away: float

class CombinedPredictionResponse(BaseModel):
    score: str
    winner: str
    probability: ProbabilityResponse
    expected_goals: ExpectedGoalsResponse
    both_teams_to_score: bool
    over_under: float

class PredictionResponse(BaseModel):
    combined_prediction: CombinedPredictionResponse

# -----------------------------
# Helper: Load model from GitHub
# -----------------------------
def load_model_from_github(filename):
    base_url = "https://github.com/AymanChabbaki/football-models/raw/refs/heads/main/"
    url = f"{base_url}{filename}"
    response = requests.get(url)
    response.raise_for_status()
    return joblib.load(io.BytesIO(response.content))

# -----------------------------
# Preprocessing helper
# -----------------------------
def create_preprocessor(categorical_features, numerical_features):
    numeric_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    return ColumnTransformer([
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# -----------------------------
# Load all league models in memory
# -----------------------------
LEAGUE_MODELS = {
    'epl': {
        'winner': {
            'filename': 'epl_winner.pkl',
            'categorical_features': ['strHomeTeam', 'strAwayTeam'],
            'numerical_features': [
                'homeGD', 'homeWinRate', 'homeClassement', 'predictedIntHomeScore', 'predictedResult',
                'awayGD', 'awayWinRate', 'awayClassement', 'predictedIntAwayScore'
            ]
        },
        'goals': {
            'filename': 'epl_goals.pkl',
            'categorical_features': ['strHomeTeam', 'strAwayTeam'],
            'numerical_features': [
                'homeGD', 'homeWinRate', 'homeClassement', 'lastHome5GamesScore',
                'awayGD', 'awayWinRate', 'awayClassement', 'lastAway5GamesScore'
            ]
        }
    },
    # Add other leagues similarly...
}

# Load models into memory
for league, models in LEAGUE_MODELS.items():
    for model_type, config in models.items():
        config['model'] = load_model_from_github(config['filename'])
        config['preprocessor'] = create_preprocessor(
            config['categorical_features'],
            config['numerical_features']
        )

# -----------------------------
# FastAPI endpoint
# -----------------------------
@app.post("/api/predict", response_model=PredictionResponse)
async def predict(request_data: PredictionRequest):
    try:
        match_data = request_data.match_data.dict()

        # Build DataFrame
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

        # Regression (goals)
        X_home = match_df[['homeGD','homeWinRate','homeClassement','lastHome5GamesScore','strHomeTeam']]
        X_away = match_df[['awayGD','awayWinRate','awayClassement','lastAway5GamesScore','strAwayTeam']]
        X_full = X_home.join(X_away)
        regression_pred = LEAGUE_MODELS['epl']['goals']['model'].predict(X_full)
        match_df['predictedIntHomeScore'] = regression_pred[:,0]
        match_df['predictedIntAwayScore'] = regression_pred[:,1]

        # Predicted result
        match_df['predictedResult'] = match_df.apply(
            lambda row: 2 if row['predictedIntHomeScore']>row['predictedIntAwayScore'] 
                        else 0 if row['predictedIntHomeScore']<row['predictedIntAwayScore'] 
                        else 1,
            axis=1
        )

        # Classification
        X_class = match_df[['homeGD','homeWinRate','homeClassement','predictedIntHomeScore','predictedResult',
                            'awayGD','awayWinRate','awayClassement','predictedIntAwayScore']]

        total_goals = match_df['predictedIntHomeScore'].values[0] + match_df['predictedIntAwayScore'].values[0]
        home_proba = (match_df['predictedIntHomeScore'].values[0]/total_goals*100) if total_goals>0 else 33.3
        away_proba = (match_df['predictedIntAwayScore'].values[0]/total_goals*100) if total_goals>0 else 33.3

        winner_pred = LEAGUE_MODELS['epl']['winner']['model'].predict(X_class)[0]
        home_goals = float(match_df['predictedIntHomeScore'].values[0])
        away_goals = float(match_df['predictedIntAwayScore'].values[0])
        winner = ['away','draw','home'][winner_pred]

        # Probabilities with classifier
        if hasattr(LEAGUE_MODELS['epl']['winner']['model'],'predict_proba'):
            class_probs = LEAGUE_MODELS['epl']['winner']['model'].predict_proba(X_class)[0]
            away_proba_class, draw_proba, home_proba_class = [p*100 for p in class_probs]
            home_proba = round(home_proba*0.3 + home_proba_class*0.7,1)
            away_proba = round(away_proba*0.3 + away_proba_class*0.7,1)
            draw_proba = round(draw_proba,1)
        else:
            draw_proba = 100 - home_proba - away_proba
            if draw_proba<0:
                total = home_proba + away_proba
                home_proba = round(home_proba/total*100,1)
                away_proba = round(away_proba/total*100,1)
                draw_proba = 0

        # Final rounding
        total = home_proba + away_proba + draw_proba
        if total!=100:
            diff = 100 - total
            if home_proba>=away_proba and home_proba>=draw_proba:
                home_proba += diff
            elif away_proba>=home_proba and away_proba>=draw_proba:
                away_proba += diff
            else:
                draw_proba += diff

        return PredictionResponse(
            combined_prediction=CombinedPredictionResponse(
                score=f"{home_goals:.1f}-{away_goals:.1f}",
                winner=winner,
                probability=ProbabilityResponse(
                    home=home_proba,
                    away=away_proba,
                    draw=draw_proba
                ),
                expected_goals=ExpectedGoalsResponse(
                    home=home_goals,
                    away=away_goals
                ),
                both_teams_to_score=bool(home_goals>=1 and away_goals>=1),
                over_under=round(home_goals + away_goals,1)
            )
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Football Prediction API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
