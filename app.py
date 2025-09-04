from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import requests
from scripts.feature_engineering import RealMadridFeatureEngineer
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

FOOTBALL_DATA_API_KEY = "9f8afca8102b4593896fc7943b920930"  # Replace with your actual API key
FOOTBALL_DATA_BASE_URL = "https://api.football-data.org/v4"

class AdvancedRealMadridPredictor:
    def __init__(self, model_path='real_madrid_model.pkl'):
        self.model = None
        self.feature_engineer = None
        self.feature_names = None
        self.target_encoder = None
        self.is_trained = False
        
        self.team_strengths = {
            'athletic club': 75, 'atletico madrid': 85, 'barcelona': 95, 'celta vigo': 60,
            'deportivo alaves': 58, 'elche cf': 55, 'espanyol': 58, 'getafe cf': 62,
            'girona': 63, 'levante': 57, 'mallorca': 59, 'osasuna': 60,
            'real betis': 65, 'real madrid': 96, 'real sociedad': 68, 'rayo vallecano': 59,
            'sevilla': 76, 'valencia': 70, 'villarreal': 72,
            'atalanta': 78, 'manchester city': 94, 'arsenal': 82, 'liverpool': 90,
            'aston villa': 75, 'bayer leverkusen': 80, 'vfb stuttgart': 73, 'bayern munich': 92,
            'rb leipzig': 77, 'borussia dortmund': 79, 'inter milan': 82, 'ac milan': 80,
            'juventus': 83, 'bologna': 68, 'paris saint-germain': 88, 'monaco': 72,
            'brest': 65, 'psv eindhoven': 74, 'feyenoord': 73, 'sporting cp': 75,
            'benfica': 76, 'club brugge': 70, 'celtic': 68, 'sturm graz': 60,
            'shakhtar donetsk': 65
        }
        
        self.competition_weights = {
            'Champions League': 1.4,
            'La Liga': 1.0,
            'Copa del Rey': 0.8,
            'Club World Cup': 0.9
        }
        
        self.load_model(model_path)

    def load_model(self, model_path):
        """Load the trained XGBoost model and feature engineer"""
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.feature_engineer = model_data['feature_engineer']
            self.feature_names = model_data['feature_names']
            self.target_encoder = model_data['target_encoder']
            self.is_trained = True
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {str(e)}. Falling back to untrained state.")
            self.is_trained = False

    def predict_match(self, opponent, competition, venue, madrid_form=2.2, 
                     madrid_xg=1.8, madrid_concede=0.7, opponent_form=1.5, 
                     rest_days=4, key_players_out=0):
        """Predict match outcome using the trained XGBoost model"""
        if not self.is_trained:
            return self._fallback_prediction(opponent, competition, venue)

        # Create single match DataFrame
        match_data = {
            'opponent': [opponent],
            'competition': [competition],
            'venue': [venue],
            'madrid_form': [madrid_form],
            'madrid_xg': [madrid_xg],
            'madrid_concede': [madrid_concede],
            'opponent_form': [opponent_form],
            'rest_days': [rest_days],
            'key_players_out': [key_players_out],
            'date': [datetime.now()],
            'madrid_elo': [2000],
            'opponent_elo': [1800],
            'form_points': [madrid_form],
            'form_xg_for': [madrid_xg],
            'form_goals_against': [madrid_concede],
            'opponent_form_points': [opponent_form],
            'key_players_absent': [key_players_out]
        }
        
        match_df = pd.DataFrame(match_data)
        
        # Prepare features
        X_match, _, _ = self.feature_engineer.prepare_features(match_df)
        
        # Predict
        probabilities = self.model.predict_proba(X_match)[0]
        
        # Ensure correct class ordering (Draw, Loss, Win)
        class_mapping = {i: cls for i, cls in enumerate(self.target_encoder.classes_)}
        win_idx = self.target_encoder.transform(['Win'])[0]
        draw_idx = self.target_encoder.transform(['Draw'])[0]
        loss_idx = self.target_encoder.transform(['Loss'])[0]
        
        # Calculate confidence based on max probability
        max_prob = max(probabilities)
        confidence = min(95, 50 + (max_prob - 0.33) * 150)
        
        return {
            'win': round(probabilities[win_idx] * 100, 1),
            'draw': round(probabilities[draw_idx] * 100, 1),
            'loss': round(probabilities[loss_idx] * 100, 1),
            'confidence': round(confidence, 1),
            'model_used': 'XGBoost'
        }

    def _fallback_prediction(self, opponent, competition, venue):
        """Fallback prediction method if model fails"""
        opponent_strength = self.team_strengths.get(opponent.lower(), 65)
        rm_strength = 96
        if venue == 'Home':
            rm_strength += 8
        else:
            rm_strength -= 3
        comp_multiplier = self.competition_weights.get(competition, 1.0)
        effective_opp_strength = opponent_strength * comp_multiplier
        strength_diff = rm_strength - effective_opp_strength
        win_prob = 1 / (1 + np.exp(-strength_diff / 15))
        win_prob = np.clip(win_prob, 0.15, 0.85)
        draw_prob = 0.25 + (0.1 * np.exp(-abs(strength_diff) / 10))
        loss_prob = 1 - win_prob - draw_prob
        if loss_prob < 0:
            loss_prob = 0.05
            win_prob = 0.95 - draw_prob
        total = win_prob + draw_prob + loss_prob
        win_prob /= total
        draw_prob /= total
        loss_prob /= total
        confidence = min(95, 50 + abs(strength_diff) * 2)
        return {
            'win': round(win_prob * 100, 1),
            'draw': round(draw_prob * 100, 1),
            'loss': round(loss_prob * 100, 1),
            'confidence': round(confidence, 1),
            'model_used': 'Statistical'
        }

    def get_real_madrid_matches(self, days_back=30):
        """Fetch Real Madrid's recent matches from Football Data API"""
        try:
            headers = {'X-Auth-Token': FOOTBALL_DATA_API_KEY}
            team_id = 86
            date_to = datetime.now().strftime('%Y-%m-%d')
            date_from = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
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
                    home_score = match['score']['fullTime']['home']
                    away_score = match['score']['fullTime']['away']
                    goal_scorers = []
                    if 'goals' in match:
                        for goal in match['goals']:
                            if goal['team']['id'] == team_id:
                                scorer_info = {
                                    'player': goal['scorer']['name'],
                                    'minute': goal['minute'],
                                    'type': goal.get('type', 'REGULAR')
                                }
                                if goal.get('assist'):
                                    scorer_info['assist'] = goal['assist']['name']
                                goal_scorers.append(scorer_info)
                    match_info = {
                        'date': match['utcDate'][:10],
                        'opponent': opponent,
                        'competition': match['competition']['name'],
                        'venue': 'Home' if is_home else 'Away',
                        'home_score': home_score,
                        'away_score': away_score,
                        'real_madrid_score': home_score if is_home else away_score,
                        'opponent_score': away_score if is_home else home_score,
                        'result': self._determine_result(home_score, away_score, is_home),
                        'goal_scorers': goal_scorers
                    }
                    matches.append(match_info)
                return matches
            else:
                print(f"API Error: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            print(f"Error fetching matches: {str(e)}")
            return []

    def _determine_result(self, home_score, away_score, is_home):
        """Determine match result from Real Madrid's perspective"""
        if is_home:
            if home_score > away_score:
                return 'Win'
            elif home_score < away_score:
                return 'Loss'
            else:
                return 'Draw'
        else:
            if away_score > home_score:
                return 'Win'
            elif away_score < home_score:
                return 'Loss'
            else:
                return 'Draw'

    def get_upcoming_matches(self, days_ahead=30):
        """Fetch Real Madrid's upcoming matches"""
        try:
            headers = {'X-Auth-Token': FOOTBALL_DATA_API_KEY}
            team_id = 86
            date_from = datetime.now().strftime('%Y-%m-%d')
            date_to = (datetime.now() + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
            url = f"{FOOTBALL_DATA_BASE_URL}/teams/{team_id}/matches"
            params = {
                'dateFrom': date_from,
                'dateTo': date_to,
                'status': 'SCHEDULED'
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
                        'time': match['utcDate'][11:16],
                        'opponent': opponent,
                        'competition': match['competition']['name'],
                        'venue': 'Home' if is_home else 'Away'
                    }
                    matches.append(match_info)
                return matches
            else:
                print(f"API Error: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            print(f"Error fetching upcoming matches: {str(e)}")
            return []

# Initialize predictor
predictor = AdvancedRealMadridPredictor(model_path='real_madrid_model.pkl')

@app.route('/')
def serve_index():
    """Serve the main HTML file"""
    return send_from_directory('public', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('public', path)

@app.route('/api/predict', methods=['POST'])
def predict_match():
    """API endpoint for match prediction"""
    try:
        data = request.get_json()
        if not data or not all(key in data for key in ['opponent', 'competition', 'venue']):
            return jsonify({'error': 'Missing required fields'}), 400
        
        opponent = data['opponent'].strip()
        competition = data['competition']
        venue = data['venue']
        if not opponent:
            return jsonify({'error': 'Opponent name cannot be empty'}), 400
        
        madrid_form = float(data.get('madridForm', 2.2))
        madrid_xg = float(data.get('madridXg', 1.8))
        madrid_concede = float(data.get('madridConcede', 0.7))
        opponent_form = float(data.get('opponentForm', 1.5))
        rest_days = int(data.get('restDays', 4))
        key_players_out = int(data.get('keyPlayersOut', 0))
        
        prediction = predictor.predict_match(
            opponent, competition, venue, madrid_form, 
            madrid_xg, madrid_concede, opponent_form, 
            rest_days, key_players_out
        )
        
        prediction['opponent'] = opponent
        prediction['competition'] = competition
        prediction['venue'] = venue
        prediction['timestamp'] = str(int(datetime.now().timestamp()))
        
        return jsonify(prediction)
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Real Madrid Match Predictor',
        'version': '1.0.0'
    })

@app.route('/api/recent-matches', methods=['GET'])
def get_recent_matches():
    """Get Real Madrid's recent matches"""
    try:
        days_back = request.args.get('days', 30, type=int)
        matches = predictor.get_real_madrid_matches(days_back)
        return jsonify({
            'matches': matches,
            'count': len(matches)
        })
    except Exception as e:
        print(f"Recent matches error: {str(e)}")
        return jsonify({'error': 'Failed to fetch recent matches'}), 500

@app.route('/api/upcoming-matches', methods=['GET'])
def get_upcoming_matches():
    """Get Real Madrid's upcoming matches"""
    try:
        days_ahead = request.args.get('days', 30, type=int)
        matches = predictor.get_upcoming_matches(days_ahead)
        return jsonify({
            'matches': matches,
            'count': len(matches)
        })
    except Exception as e:
        print(f"Upcoming matches error: {str(e)}")
        return jsonify({'error': 'Failed to fetch upcoming matches'}), 500

@app.route('/api/team-list', methods=['GET'])
def get_team_list():
    """Get list of available teams for prediction"""
    teams = list(predictor.team_strengths.keys())
    teams.sort()
    return jsonify({
        'teams': teams,
        'count': len(teams)
    })

def print_hala_madrid():
    """Print Hala Madrid ASCII art"""
    print(r"""
 _    _       _          __  __           _      _     _ 
| |  | |     | |        |  \/  |         | |    (_)   | |
| |__| | __ _| | __ _   | \  / | __ _  __| |____ _  __| |
|  __  |/ _` | |/ _` |  | |\/| |/ _` |/ _` |  __| |/ _` |
| |  | | (_| | | (_| |  | |  | | (_| | (_| | |  | | (_| |
|_|  |_|___,_|_|\__,_|  |_|  |_|__,__|__,__|_|  |_|__,__|
                                                                                 
        ⚪ Hala Madrid! ⚪
    """)

if __name__ == '__main__':
    os.makedirs('public/assets/logo', exist_ok=True)
    print_hala_madrid()
    print("🏆 Real Madrid Match Predictor Server")
    print("📁 Letakkan logo Real Madrid di: public/assets/logo/real-madrid-logo.png")
    print("🌐 Server akan berjalan di: http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)

