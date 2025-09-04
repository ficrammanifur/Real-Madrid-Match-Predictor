import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class RealMadridFeatureEngineer:
    def __init__(self):
        self.target_encoder = LabelEncoder()
        self.competition_weights = {
            'Champions League': 1.4,
            'La Liga': 1.0,
            'Copa del Rey': 0.8,
            'Club World Cup': 0.9
        }
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
        # Inisialisasi target_encoder dengan kelas yang diharapkan
        self.target_encoder.fit(['Win', 'Draw', 'Loss'])

    def create_features(self, df):
        """Membuat fitur untuk pelatihan model"""
        df = df.copy()

        # Menangani kolom venue
        venue_col = None
        possible_venue_cols = ['venue', 'Venue', 'home_away', 'Home_Away']
        for col in possible_venue_cols:
            if col in df.columns:
                venue_col = col
                break

        if venue_col is None:
            print("Peringatan: Kolom venue tidak ditemukan. Mengasumsikan 'Away' untuk semua pertandingan.")
            df['venue'] = 'Away'
            venue_col = 'venue'
        
        df['venue_encoded'] = (df[venue_col] == 'Home').astype(int)

        # Menambahkan kekuatan tim
        df['opponent_strength'] = df['opponent'].str.lower().map(self.team_strengths).fillna(65)
        
        # Menambahkan bobot kompetisi
        df['competition_weight'] = df['competition'].map(self.competition_weights).fillna(1.0)
        
        # Memetakan input form dari app.py ke fitur train_model.py
        df['form_points'] = df.get('madrid_form', df.get('form_points', 2.2))
        df['form_xg_for'] = df.get('madrid_xg', df.get('form_xg_for', 1.8))
        df['form_goals_against'] = df.get('madrid_concede', df.get('form_goals_against', 0.7))
        df['opponent_form_points'] = df.get('opponent_form', df.get('opponent_form_points', 1.5))
        df['rest_days'] = df.get('rest_days', 4)
        df['key_players_absent'] = df.get('key_players_out', df.get('key_players_absent', 0))

        # Menghitung selisih ELO (placeholder jika tidak disediakan)
        df['elo_diff'] = df.get('madrid_elo', 2000) - df.get('opponent_elo', 1800)
        
        # Fitur tambahan yang direkayasa
        df['form_diff'] = df['form_points'] - df['opponent_form_points']
        df['xg_ratio'] = df['form_xg_for'] / df['form_goals_against'].clip(lower=0.1)
        df['strength_advantage'] = 96 - df['opponent_strength']
        
        return df

    def prepare_features(self, df):
        """Mempersiapkan matriks fitur dan variabel target"""
        # Membuat fitur
        df = self.create_features(df)
        
        # Mendefinisikan kolom fitur
        feature_cols = [
            'opponent_strength',
            'venue_encoded',
            'competition_weight',
            'elo_diff',
            'form_points',
            'form_goals_for',
            'form_goals_against',
            'form_xg_for',
            'form_xg_against',
            'form_diff',
            'xg_ratio',
            'strength_advantage',
            'rest_days',
            'key_players_absent'
        ]
        
        # Memastikan semua kolom fitur ada
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0  # Nilai default untuk kolom yang hilang
        
        X = df[feature_cols]
        
        # Mempersiapkan variabel target
        y = None
        if 'result' in df.columns:
            y = self.target_encoder.transform(df['result'])
        
        return X, y, feature_cols

    def get_feature_names(self):
        """Mengembalikan nama fitur"""
        return [
            'opponent_strength',
            'venue_encoded',
            'competition_weight',
            'elo_diff',
            'form_points',
            'form_goals_for',
            'form_goals_against',
            'form_xg_for',
            'form_xg_against',
            'form_diff',
            'xg_ratio',
            'strength_advantage',
            'rest_days',
            'key_players_absent'
        ]