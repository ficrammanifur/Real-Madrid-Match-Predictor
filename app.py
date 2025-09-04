from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import requests
import logging
import sys

# Menambahkan direktori saat ini dan direktori induk ke sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))

try:
    from feature_engineering import RealMadridFeatureEngineer
except ImportError as e:
    print("Error: Tidak dapat mengimpor RealMadridFeatureEngineer. Pastikan feature_engineering.py ada di direktori scripts.")
    raise

app = Flask(__name__, static_folder='public', static_url_path='/static')
CORS(app)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FOOTBALL_DATA_API_KEY = "9f8afca8102b4593896fc7943b920930"  # Ganti dengan kunci API Anda
FOOTBALL_DATA_BASE_URL = "https://api.football-data.org/v4"

class AdvancedRealMadridPredictor:
    def __init__(self, model_path='real_madrid_model.pkl'):
        self.model = None
        self.feature_engineer = None
        self.feature_names = None
        self.is_trained = False
        self.competition_weights = {
            'Champions League': 1.4,
            'La Liga': 1.0,
            'Copa del Rey': 0.8,
            'Club World Cup': 0.9
        }
        # Mengatur jalur model ke direktori induk
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(base_dir, model_path)
        self.load_model()

    def load_model(self):
        """Memuat model XGBoost yang telah dilatih dan feature engineer"""
        try:
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            self.feature_engineer = model_data['feature_engineer']
            self.feature_names = model_data['feature_names']
            self.is_trained = True
            logger.info(f"Model berhasil dimuat dari {self.model_path}")
        except Exception as e:
            logger.error(f"Error memuat model: {str(e)}. Menggunakan status belum dilatih.")
            self.feature_engineer = RealMadridFeatureEngineer()
            self.is_trained = False

    def predict_score(self, form_xg_for, form_xg_against, outcome, venue):
        """Memperkirakan skor berdasarkan xG dan hasil prediksi"""
        home_goals = round(form_xg_for * (1.1 if venue == 'Home' else 0.9))
        away_goals = round(form_xg_against * (0.9 if venue == 'Home' else 1.1))
        
        if outcome == 'Win':
            home_goals = max(home_goals, away_goals + 1)
        elif outcome == 'Loss':
            away_goals = max(away_goals, home_goals + 1)
        elif outcome == 'Draw':
            home_goals = away_goals = max(home_goals, away_goals)
        
        return home_goals, away_goals

    def predict_match(self, opponent, competition, venue, madrid_form=2.2, 
                     madrid_xg=1.8, madrid_concede=0.7, opponent_form=1.5, 
                     rest_days=4, key_players_out=0):
        """Prediksi hasil pertandingan menggunakan model XGBoost"""
        if not self.is_trained:
            return self._fallback_prediction(opponent, competition, venue)

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
        X_match, _, _ = self.feature_engineer.prepare_features(match_df)
        
        probabilities = self.model.predict_proba(X_match)[0]
        predicted_class = self.model.predict(X_match)[0]
        
        class_mapping = {i: cls for i, cls in enumerate(self.feature_engineer.target_encoder.classes_)}
        win_idx = self.feature_engineer.target_encoder.transform(['Win'])[0]
        draw_idx = self.feature_engineer.target_encoder.transform(['Draw'])[0]
        loss_idx = self.feature_engineer.target_encoder.transform(['Loss'])[0]
        
        outcome = class_mapping[predicted_class]
        home_goals, away_goals = self.predict_score(madrid_xg, madrid_concede, outcome, venue)
        
        max_prob = max(probabilities)
        confidence = min(95, 50 + (max_prob - 0.33) * 150)
        
        return {
            'win': round(probabilities[win_idx] * 100, 1),
            'draw': round(probabilities[draw_idx] * 100, 1),
            'loss': round(probabilities[loss_idx] * 100, 1),
            'confidence': round(confidence, 1),
            'model_used': 'XGBoost',
            'predicted_outcome': outcome,
            'predicted_score': f"{home_goals}-{away_goals}"
        }

    def _fallback_prediction(self, opponent, competition, venue):
        """Prediksi cadangan jika model gagal"""
        opponent_strength = self.feature_engineer.team_strengths.get(opponent.lower(), 65)
        rm_strength = self.feature_engineer.team_strengths.get('real madrid', 96)
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
        
        outcome = 'Win' if win_prob > max(draw_prob, loss_prob) else 'Draw' if draw_prob > loss_prob else 'Loss'
        home_goals = 2 if outcome == 'Win' else 1 if outcome == 'Draw' else 0
        away_goals = 0 if outcome == 'Win' else 1 if outcome == 'Draw' else 2
        
        return {
            'win': round(win_prob * 100, 1),
            'draw': round(draw_prob * 100, 1),
            'loss': round(loss_prob * 100, 1),
            'confidence': round(confidence, 1),
            'model_used': 'Statistical',
            'predicted_outcome': outcome,
            'predicted_score': f"{home_goals}-{away_goals}"
        }

    def get_real_madrid_matches(self, days_back=30):
        """Ambil pertandingan terbaru Real Madrid dari API"""
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
                logger.error(f"API Error: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            logger.error(f"Error mengambil pertandingan: {str(e)}")
            return []

    def _determine_result(self, home_score, away_score, is_home):
        """Menentukan hasil pertandingan dari perspektif Real Madrid"""
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
        """Ambil jadwal pertandingan Real Madrid"""
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
                logger.error(f"API Error: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            logger.error(f"Error mengambil jadwal pertandingan: {str(e)}")
            return []

# Inisialisasi predictor
predictor = AdvancedRealMadridPredictor()

@app.route('/')
def serve_index():
    """Menyajikan file HTML utama"""
    return send_from_directory('public', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Menyajikan file statis"""
    return send_from_directory('public', path)

@app.route('/api/predict', methods=['POST'])
def predict_match():
    """Endpoint API untuk prediksi pertandingan"""
    try:
        data = request.get_json()
        if not data or not all(key in data for key in ['opponent', 'competition', 'venue']):
            return jsonify({'error': 'Field yang diperlukan tidak lengkap'}), 400
        
        opponent = data['opponent'].strip()
        competition = data['competition']
        venue = data['venue']
        if not opponent:
            return jsonify({'error': 'Nama lawan tidak boleh kosong'}), 400
        
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
        
        logger.info(f"Prediksi untuk {opponent}: {prediction}")
        return jsonify(prediction)
        
    except Exception as e:
        logger.error(f"Error prediksi: {str(e)}")
        return jsonify({'error': 'Kesalahan server internal'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Endpoint pemeriksaan kesehatan"""
    return jsonify({
        'status': 'healthy',
        'service': 'Real Madrid Match Predictor',
        'version': '1.0.0'
    })

@app.route('/api/recent-matches', methods=['GET'])
def get_recent_matches():
    """Ambil pertandingan terbaru Real Madrid"""
    try:
        days_back = request.args.get('days', 30, type=int)
        matches = predictor.get_real_madrid_matches(days_back)
        return jsonify({
            'matches': matches,
            'count': len(matches)
        })
    except Exception as e:
        logger.error(f"Error pertandingan terbaru: {str(e)}")
        return jsonify({'error': 'Gagal mengambil pertandingan terbaru'}), 500

@app.route('/api/upcoming-matches', methods=['GET'])
def get_upcoming_matches():
    """Ambil jadwal pertandingan Real Madrid"""
    try:
        days_ahead = request.args.get('days', 30, type=int)
        matches = predictor.get_upcoming_matches(days_ahead)
        return jsonify({
            'matches': matches,
            'count': len(matches)
        })
    except Exception as e:
        logger.error(f"Error jadwal pertandingan: {str(e)}")
        return jsonify({'error': 'Gagal mengambil jadwal pertandingan'}), 500

@app.route('/api/team-list', methods=['GET'])
def get_team_list():
    """Ambil daftar tim yang tersedia untuk prediksi"""
    try:
        teams = list(predictor.feature_engineer.team_strengths.keys())
        teams.sort()
        return jsonify({
            'teams': teams,
            'count': len(teams)
        })
    except Exception as e:
        logger.error(f"Error mengambil daftar tim: {str(e)}")
        return jsonify({'error': 'Gagal mengambil daftar tim'}), 500

def print_hala_madrid():
    """Cetak seni ASCII Hala Madrid"""
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