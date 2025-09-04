import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import sys
warnings.filterwarnings('ignore')

# Menambahkan direktori saat ini dan direktori induk ke sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from feature_engineering import RealMadridFeatureEngineer
except ImportError as e:
    print("Error: Tidak dapat mengimpor RealMadridFeatureEngineer. Pastikan feature_engineering.py ada di direktori scripts atau direktori proyek.")
    raise

class RealMadridModelTrainer:
    def __init__(self):
        self.model = None
        self.feature_engineer = RealMadridFeatureEngineer()
        self.feature_names = None

    def load_data(self, file_path='combined_matches.csv'):
        """Memuat dan memproses dataset"""
        print("\nMemuat dataset...")
        # Mengatur jalur file ke direktori induk jika dijalankan dari subdirektori scripts
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_path = os.path.join(base_dir, file_path)
        try:
            df = pd.read_csv(full_path)
            print(f"Dataset dimuat: {len(df)} pertandingan dari {full_path}")
            return df
        except Exception as e:
            print(f"Error saat memuat dataset: {str(e)}")
            return None

    def train_model(self, X_train, y_train, X_val=None, y_val=None):
        """Melatih model XGBoost dengan penyetelan hiperparameter"""
        print("\nMelatih model XGBoost...")
        
        # Menghitung bobot kelas untuk menangani ketidakseimbangan
        class_counts = np.bincount(y_train)
        total_samples = len(y_train)
        class_weights = total_samples / (len(class_counts) * class_counts)
        weights = np.array([class_weights[label] for label in y_train])
        
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'random_state': 42,
            'eval_metric': 'mlogloss'
        }
        
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [100, 200, 300],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }
        
        self.model = xgb.XGBClassifier(**params)
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=TimeSeriesSplit(n_splits=5),
            scoring='neg_log_loss',
            n_jobs=-1,
            verbose=1
        )
        
        if X_val is not None and y_val is not None:
            grid_search.fit(X_train, y_train, sample_weight=weights, eval_set=[(X_val, y_val)], early_stopping_rounds=20, verbose=False)
        else:
            grid_search.fit(X_train, y_train, sample_weight=weights)
        
        self.model = grid_search.best_estimator_
        print(f"Parameter terbaik: {grid_search.best_params_}")
        print(f"Skor validasi silang terbaik: {-grid_search.best_score_:.4f}")
        
        return self.model

    def evaluate_model(self, X_test, y_test):
        """Mengevaluasi model pada set pengujian"""
        print("\nMengevaluasi pada set pengujian...")
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        logloss = log_loss(y_test, y_pred_proba)
        
        if not hasattr(self.feature_engineer.target_encoder, 'classes_'):
            print("Error: Target encoder belum di-fit. Pastikan data target sudah diproses.")
            return
        
        brier_scores = []
        for i, class_label in enumerate(self.feature_engineer.target_encoder.classes_):
            y_true_binary = (y_test == i).astype(int)
            brier = brier_score_loss(y_true_binary, y_pred_proba[:, i], pos_label=1)
            brier_scores.append(brier)
        avg_brier = np.mean(brier_scores)
        
        print("\nHasil Pengujian:")
        print(f"Akurasi: {accuracy:.4f}")
        print(f"Log Loss: {logloss:.4f}")
        print(f"Rata-rata Skor Brier (multiclass): {avg_brier:.4f}")
        for i, class_label in enumerate(self.feature_engineer.target_encoder.classes_):
            print(f"Skor Brier ({class_label}): {brier_scores[i]:.4f}")
        
        print("\nLaporan Klasifikasi:")
        print(classification_report(y_test, y_pred, target_names=self.feature_engineer.target_encoder.classes_))

    def plot_feature_importance(self):
        """Memplot pentingnya fitur"""
        importance = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print("\n10 Fitur Terpenting:")
        print(feature_importance_df.head(10))
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance_df.head(10))
        plt.title('10 Fitur Terpenting')
        plt.tight_layout()
        plt.show()

    def plot_calibration_curves(self, X_test, y_test):
        """Menghasilkan kurva kalibrasi untuk setiap kelas"""
        print("\nMenghasilkan kurva kalibrasi...")
        from sklearn.calibration import calibration_curve
        
        if not hasattr(self.feature_engineer.target_encoder, 'classes_'):
            print("Error: Target encoder belum di-fit. Tidak dapat menghasilkan kurva kalibrasi.")
            return
        
        plt.figure(figsize=(10, 8))
        for i, class_name in enumerate(self.feature_engineer.target_encoder.classes_):
            prob_true, prob_pred = calibration_curve(
                y_test == i, self.model.predict_proba(X_test)[:, i], n_bins=10
            )
            plt.plot(prob_pred, prob_true, marker='o', label=f'{class_name}')
        
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel('Probabilitas Prediksi')
        plt.ylabel('Probabilitas Sebenarnya')
        plt.title('Kurva Kalibrasi')
        plt.legend()
        plt.tight_layout()
        plt.show()

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

    def make_example_predictions(self):
        """Menghasilkan prediksi contoh"""
        print("\n=== Prediksi Contoh ===")
        example_matches = [
            {'opponent': 'Barcelona', 'competition': 'La Liga', 'venue': 'Home',
             'form_points': 2.2, 'form_goals_for': 2.0, 'form_goals_against': 0.7,
             'form_xg_for': 1.8, 'form_xg_against': 1.0, 'opponent_form_points': 1.5,
             'rest_days': 4, 'key_players_absent': 0},
            {'opponent': 'Bayern Munich', 'competition': 'Champions League', 'venue': 'Away',
             'form_points': 2.0, 'form_goals_for': 1.8, 'form_goals_against': 0.8,
             'form_xg_for': 1.6, 'form_xg_against': 1.2, 'opponent_form_points': 1.8,
             'rest_days': 3, 'key_players_absent': 1},
            {'opponent': 'Getafe', 'competition': 'La Liga', 'venue': 'Home',
             'form_points': 2.5, 'form_goals_for': 2.5, 'form_goals_against': 0.5,
             'form_xg_for': 2.0, 'form_xg_against': 0.8, 'opponent_form_points': 1.2,
             'rest_days': 5, 'key_players_absent': 0},
            {'opponent': 'Manchester City', 'competition': 'Champions League', 'venue': 'Home',
             'form_points': 2.3, 'form_goals_for': 2.2, 'form_goals_against': 0.6,
             'form_xg_for': 1.9, 'form_xg_against': 1.1, 'opponent_form_points': 1.7,
             'rest_days': 4, 'key_players_absent': 0}
        ]
        
        for match in example_matches:
            match_df = pd.DataFrame([match])
            match_df['madrid_elo'] = 2000
            match_df['opponent_elo'] = 1800
            X_match, _, _ = self.feature_engineer.prepare_features(match_df)
            probs = self.model.predict_proba(X_match)[0]
            predicted_class = self.model.predict(X_match)[0]
            outcome = self.feature_engineer.target_encoder.classes_[predicted_class]
            
            home_goals, away_goals = self.predict_score(
                match['form_xg_for'],
                match['form_xg_against'],
                outcome,
                match['venue']
            )
            
            win_idx = self.feature_engineer.target_encoder.transform(['Win'])[0]
            draw_idx = self.feature_engineer.target_encoder.transform(['Draw'])[0]
            loss_idx = self.feature_engineer.target_encoder.transform(['Loss'])[0]
            print(f"\nReal Madrid vs {match['opponent']} ({match['competition']}, {match['venue']}):")
            print(f"Menang: {probs[win_idx]*100:.1f}%, Seri: {probs[draw_idx]*100:.1f}%, Kalah: {probs[loss_idx]*100:.1f}%")
            print(f"Prediksi Skor: {home_goals}-{away_goals} ({outcome})")

    def save_model(self, file_path='real_madrid_model.pkl'):
        """Menyimpan model dan feature engineer"""
        # Mengatur jalur file ke direktori induk
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_path = os.path.join(base_dir, file_path)
        model_data = {
            'model': self.model,
            'feature_engineer': self.feature_engineer,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, full_path)
        print(f"\nModel disimpan ke {full_path}")

def main():
    print("=== Pelatihan Prediksi Pertandingan Real Madrid ===")
    trainer = RealMadridModelTrainer()
    
    df = trainer.load_data()
    if df is None or df.empty:
        print("Tidak ada data yang tersedia. Keluar.")
        return
    
    print("\nMempersiapkan fitur...")
    X, y, feature_names = trainer.feature_engineer.prepare_features(df)
    trainer.feature_names = feature_names
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    print(f"Set pelatihan: {len(X_train)} pertandingan")
    print(f"Set pengujian: {len(X_test)} pertandingan")
    
    trainer.train_model(X_train, y_train)
    trainer.evaluate_model(X_test, y_test)
    trainer.plot_feature_importance()
    trainer.plot_calibration_curves(X_test, y_test)
    trainer.make_example_predictions()
    trainer.save_model()
    
    print("\nPelatihan model selesai dan disimpan!")

if __name__ == "__main__":
    main()