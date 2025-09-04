# data_collector.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import os

FOOTBALL_DATA_API_KEY = os.getenv("FOOTBALL_DATA_API_KEY", "9f8afca8102b4593896fc7943b920930")
FOOTBALL_DATA_BASE_URL = "https://api.football-data.org/v4"

def collect_real_madrid_matches(days_back=365):
    """Fetch Real Madrid's recent matches from Football Data API"""
    try:
        headers = {'X-Auth-Token': FOOTBALL_DATA_API_KEY}
        team_id = 86  # Real Madrid's ID
        
        date_to = datetime.now().strftime('%Y-%m-%d')
        date_from = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        # Use the correct endpoint for matches
        url = f"{FOOTBALL_DATA_BASE_URL}/teams/{team_id}/matches"
        params = {
            'dateFrom': date_from,
            'dateTo': date_to,
            'status': 'FINISHED'
        }
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            matches = []
            
            for match in data.get('matches', []):
                is_home = match['homeTeam']['id'] == team_id
                opponent = match['awayTeam']['name'] if is_home else match['homeTeam']['name']
                
                match_info = {
                    'date': match['utcDate'][:10],
                    'opponent': opponent,
                    'competition': match['competition']['name'],
                    'venue': 'Home' if is_home else 'Away',
                    'result': 'Win' if (is_home and match['score']['fullTime']['home'] > match['score']['fullTime']['away']) or \
                                     (not is_home and match['score']['fullTime']['away'] > match['score']['fullTime']['home']) else \
                              'Loss' if (is_home and match['score']['fullTime']['home'] < match['score']['fullTime']['away']) or \
                                      (not is_home and match['score']['fullTime']['away'] < match['score']['fullTime']['home']) else 'Draw',
                    'form_points': np.random.uniform(1.5, 2.8),  # Placeholder (replace with real data if available)
                    'form_goals_for': np.random.uniform(1.5, 3.0),
                    'form_goals_against': np.random.uniform(0.5, 1.5),
                    'form_xg_for': np.random.uniform(1.2, 2.5),
                    'form_xg_against': np.random.uniform(0.8, 1.8),
                    'opponent_form_points': np.random.uniform(0.8, 2.5),
                    'rest_days': np.random.randint(2, 10),
                    'key_players_absent': np.random.randint(0, 3)
                }
                matches.append(match_info)
            
            if matches:
                df = pd.DataFrame(matches)
                print(f"Dataset saved to sample_matches.csv")
                print(f"Total matches: {len(df)}")
                print(f"Date range: {df['date'].min()} to {df['date'].max()}")
                df.to_csv('sample_matches.csv', index=False)
                return df
            else:
                print("No matches found in API response.")
                return pd.DataFrame()
        else:
            print(f"API Error: {response.status_code} - {response.text}")
            return pd.DataFrame()
    
    except Exception as e:
        print(f"Error fetching matches: {str(e)}")
        return pd.DataFrame()

def generate_synthetic_matches(n_matches=300):
    """Generate synthetic match data"""
    start_date = datetime(2022, 1, 1)
    teams = [
        'athletic club', 'atletico madrid', 'barcelona', 'celta vigo',
        'deportivo alaves', 'elche cf', 'espanyol', 'getafe cf',
        'girona', 'levante', 'mallorca', 'osasuna',
        'real betis', 'real sociedad', 'rayo vallecano', 'sevilla',
        'valencia', 'villarreal', 'manchester city', 'arsenal',
        'liverpool', 'bayern munich', 'paris saint-germain'
    ]
    competitions = ['La Liga', 'Champions League', 'Copa del Rey']
    
    matches = []
    for i in range(n_matches):
        date = start_date + timedelta(days=i*5)
        opponent = np.random.choice(teams)
        competition = np.random.choice(competitions, p=[0.6, 0.3, 0.1])
        venue = np.random.choice(['Home', 'Away'])
        
        # Simulate match outcome
        outcome = np.random.choice(['Win', 'Draw', 'Loss'], p=[0.5, 0.3, 0.2])
        
        match_info = {
            'date': date.strftime('%Y-%m-%d'),
            'opponent': opponent,
            'competition': competition,
            'venue': venue,
            'result': outcome,
            'form_points': np.random.uniform(1.5, 2.8),
            'form_goals_for': np.random.uniform(1.5, 3.0),
            'form_goals_against': np.random.uniform(0.5, 1.5),
            'form_xg_for': np.random.uniform(1.2, 2.5),
            'form_xg_against': np.random.uniform(0.8, 1.8),
            'opponent_form_points': np.random.uniform(0.8, 2.5),
            'rest_days': np.random.randint(2, 10),
            'key_players_absent': np.random.randint(0, 3)
        }
        matches.append(match_info)
    
    df = pd.DataFrame(matches)
    print(f"Dataset saved to training_matches.csv")
    print(f"Total matches: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    df.to_csv('training_matches.csv', index=False)
    return df

def main():
    """Main data collection pipeline"""
    # Collect real matches
    real_matches = collect_real_madrid_matches(days_back=365)
    
    # Generate synthetic matches
    synthetic_matches = generate_synthetic_matches(n_matches=300)
    
    # Combine datasets
    combined_df = pd.concat([real_matches, synthetic_matches], ignore_index=True)
    if not combined_df.empty:
        combined_df.to_csv('combined_matches.csv', index=False)
        print(f"Combined dataset saved to combined_matches.csv")
        print(f"Total matches: {len(combined_df)}")
    else:
        print("No data to save in combined_matches.csv")

if __name__ == "__main__":
    main()