import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_sample_dataset():
    """
    Generate sample Real Madrid match dataset with realistic data structure
    """
    
    # Sample opponents with different strength levels
    opponents = {
        'Barcelona': 85, 'Atletico Madrid': 82, 'Bayern Munich': 84, 'Manchester City': 86,
        'Liverpool': 83, 'PSG': 81, 'Chelsea': 79, 'Arsenal': 78, 'Juventus': 77,
        'AC Milan': 76, 'Inter Milan': 75, 'Napoli': 74, 'Sevilla': 73, 'Valencia': 70,
        'Real Sociedad': 69, 'Athletic Bilbao': 68, 'Villarreal': 72, 'Real Betis': 67,
        'Celta Vigo': 65, 'Getafe': 64, 'Osasuna': 63, 'Cadiz': 60, 'Almeria': 58,
        'Mallorca': 62, 'Las Palmas': 59, 'Girona': 66, 'Rayo Vallecano': 61
    }
    
    competitions = ['La Liga', 'Champions League', 'Copa del Rey']
    venues = ['Home', 'Away']
    
    # Generate 500 matches over 3 seasons
    matches = []
    start_date = datetime(2021, 8, 1)
    
    for i in range(500):
        match_date = start_date + timedelta(days=random.randint(0, 1095))  # 3 years
        opponent = random.choice(list(opponents.keys()))
        opponent_strength = opponents[opponent]
        competition = random.choice(competitions)
        venue = random.choice(venues)
        
        # Calculate realistic match outcome based on factors
        madrid_strength = 88  # Base Real Madrid strength
        
        # Venue advantage
        if venue == 'Home':
            madrid_strength += 3
        
        # Competition difficulty
        if competition == 'Champions League':
            opponent_strength += 2
        
        # Calculate probabilities
        strength_diff = madrid_strength - opponent_strength
        
        # Base probabilities
        if strength_diff > 10:
            win_prob, draw_prob, loss_prob = 0.7, 0.2, 0.1
        elif strength_diff > 5:
            win_prob, draw_prob, loss_prob = 0.6, 0.25, 0.15
        elif strength_diff > 0:
            win_prob, draw_prob, loss_prob = 0.5, 0.3, 0.2
        elif strength_diff > -5:
            win_prob, draw_prob, loss_prob = 0.4, 0.3, 0.3
        else:
            win_prob, draw_prob, loss_prob = 0.3, 0.25, 0.45
        
        # Generate actual result
        rand = random.random()
        if rand < win_prob:
            result = 'Win'
            madrid_goals = random.randint(1, 4)
            opponent_goals = random.randint(0, 2)
        elif rand < win_prob + draw_prob:
            result = 'Draw'
            goals = random.randint(0, 3)
            madrid_goals = opponent_goals = goals
        else:
            result = 'Loss'
            madrid_goals = random.randint(0, 2)
            opponent_goals = random.randint(1, 4)
        
        # Generate additional statistics
        possession = random.randint(45, 70) if venue == 'Home' else random.randint(40, 65)
        shots = random.randint(8, 20)
        shots_on_target = random.randint(3, min(shots, 12))
        xg_madrid = round(random.uniform(0.5, 3.5), 2)
        xg_opponent = round(random.uniform(0.3, 2.8), 2)
        
        # Rest days (days since last match)
        rest_days = random.randint(2, 14)
        
        # Key players absent (0-3 players)
        key_players_absent = random.randint(0, 3)
        
        matches.append({
            'date': match_date.strftime('%Y-%m-%d'),
            'opponent': opponent,
            'opponent_strength': opponent_strength,
            'competition': competition,
            'venue': venue,
            'result': result,
            'madrid_goals': madrid_goals,
            'opponent_goals': opponent_goals,
            'possession': possession,
            'shots': shots,
            'shots_on_target': shots_on_target,
            'xg_madrid': xg_madrid,
            'xg_opponent': xg_opponent,
            'rest_days': rest_days,
            'key_players_absent': key_players_absent
        })
    
    # Convert to DataFrame and sort by date
    df = pd.DataFrame(matches)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    return df

def add_elo_ratings(df):
    """
    Add Elo ratings for Real Madrid and opponents
    """
    madrid_elo = 2000  # Starting Elo
    opponent_elos = {opponent: 1800 + random.randint(-200, 200) 
                    for opponent in df['opponent'].unique()}
    
    madrid_elo_history = []
    opponent_elo_history = []
    elo_diff_history = []
    
    K = 32  # Elo K-factor
    
    for idx, row in df.iterrows():
        opponent = row['opponent']
        opponent_elo = opponent_elos[opponent]
        
        # Store current Elos
        madrid_elo_history.append(madrid_elo)
        opponent_elo_history.append(opponent_elo)
        elo_diff_history.append(madrid_elo - opponent_elo)
        
        # Calculate expected scores
        expected_madrid = 1 / (1 + 10**((opponent_elo - madrid_elo) / 400))
        expected_opponent = 1 - expected_madrid
        
        # Actual scores
        if row['result'] == 'Win':
            actual_madrid, actual_opponent = 1, 0
        elif row['result'] == 'Draw':
            actual_madrid, actual_opponent = 0.5, 0.5
        else:
            actual_madrid, actual_opponent = 0, 1
        
        # Update Elos
        madrid_elo += K * (actual_madrid - expected_madrid)
        opponent_elos[opponent] += K * (actual_opponent - expected_opponent)
    
    df['madrid_elo'] = madrid_elo_history
    df['opponent_elo'] = opponent_elo_history
    df['elo_diff'] = elo_diff_history
    
    return df

def add_form_features(df):
    """
    Add rolling form features (last 5 matches)
    """
    df['points'] = df['result'].map({'Win': 3, 'Draw': 1, 'Loss': 0})
    
    # Rolling averages (last 5 matches)
    df['form_points'] = df['points'].rolling(window=5, min_periods=1).mean()
    df['form_goals_for'] = df['madrid_goals'].rolling(window=5, min_periods=1).mean()
    df['form_goals_against'] = df['opponent_goals'].rolling(window=5, min_periods=1).mean()
    df['form_xg_for'] = df['xg_madrid'].rolling(window=5, min_periods=1).mean()
    df['form_xg_against'] = df['xg_opponent'].rolling(window=5, min_periods=1).mean()
    
    return df

if __name__ == "__main__":
    print("Generating Real Madrid dataset...")
    
    # Generate base dataset
    df = generate_sample_dataset()
    
    # Add advanced features
    df = add_elo_ratings(df)
    df = add_form_features(df)
    
    # Save to CSV
    df.to_csv('real_madrid_matches.csv', index=False)
    
    print(f"Dataset created with {len(df)} matches")
    print("\nDataset structure:")
    print(df.info())
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nResult distribution:")
    print(df['result'].value_counts())
