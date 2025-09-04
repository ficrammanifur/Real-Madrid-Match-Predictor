import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from feature_engineering import RealMadridFeatureEngineer
from advanced_features import enhance_dataset_with_advanced_features

class EnhancedRealMadridPredictor:
    def __init__(self):
        self.model = None
        self.feature_selector = None
        self.selected_features = None
        self.feature_importance = None
        
    def prepare_enhanced_data(self, df_path='real_madrid_matches_enhanced.csv'):
        """
        Prepare enhanced dataset for training
        """
        df = pd.read_csv(df_path)
        df = df.sort_values('date').reset_index(drop=True)
        
        # Define target
        target_map = {'Draw': 0, 'Loss': 1, 'Win': 2}
        y = df['result'].map(target_map).values
        
        # Select numerical features for model
        exclude_cols = ['date', 'opponent', 'competition', 'venue', 'result']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].fillna(0)
        
        # Handle any remaining non-numeric columns
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.Categorical(X[col]).codes
        
        return X, y, feature_cols, df
    
    def select_best_features(self, X, y, k=30):
        """
        Select top k features using statistical tests
        """
        self.feature_selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_mask = self.feature_selector.get_support()
        self.selected_features = X.columns[selected_mask].tolist()
        
        print(f"Selected {len(self.selected_features)} best features:")
        for i, feature in enumerate(self.selected_features, 1):
            score = self.feature_selector.scores_[selected_mask][i-1]
            print(f"{i:2d}. {feature:<30} (score: {score:.2f})")
        
        return X_selected
    
    def train_enhanced_model(self, X, y):
        """
        Train enhanced XGBoost model with optimized hyperparameters
        """
        # Enhanced hyperparameters for complex feature space
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'max_depth': 8,
            'learning_rate': 0.05,
            'n_estimators': 300,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'colsample_bylevel': 0.85,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'eval_metric': 'mlogloss',
            'early_stopping_rounds': 30
        }
        
        # Time-based split
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        self.model = xgb.XGBClassifier(**params)
        
        # Train with validation
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.selected_features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return X_train, X_val, y_train, y_val
    
    def evaluate_enhanced_model(self, X_val, y_val):
        """
        Comprehensive evaluation of enhanced model
        """
        y_pred = self.model.predict(X_val)
        y_pred_proba = self.model.predict_proba(X_val)
        
        accuracy = accuracy_score(y_val, y_pred)
        logloss = log_loss(y_val, y_pred_proba)
        
        print(f"Enhanced Model Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Log Loss: {logloss:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_val, y_pred, target_names=['Draw', 'Loss', 'Win']))
        
        return accuracy, logloss
    
    def plot_enhanced_feature_importance(self, top_n=20):
        """
        Plot feature importance for enhanced model
        """
        plt.figure(figsize=(12, 8))
        
        top_features = self.feature_importance.head(top_n)
        
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importance - Enhanced Real Madrid Model')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
        return top_features
    
    def compare_models(self, basic_model_path='real_madrid_model.pkl'):
        """
        Compare enhanced model with basic model
        """
        # Load basic model results (you would need to save these during basic training)
        print("Model Comparison:")
        print("=" * 50)
        print("Enhanced Model: Uses 30+ advanced features")
        print("Basic Model: Uses 22 standard features")
        print("\nFeature Categories in Enhanced Model:")
        
        feature_categories = {
            'Dynamic Elo': [f for f in self.selected_features if 'elo' in f.lower()],
            'Advanced xG': [f for f in self.selected_features if 'xg' in f.lower()],
            'Player Impact': [f for f in self.selected_features if any(x in f.lower() for x in ['absent', 'rating', 'importance'])],
            'Tactical': [f for f in self.selected_features if any(x in f.lower() for x in ['pressure', 'momentum', 'streak'])],
            'Interactions': [f for f in self.selected_features if any(x in f.lower() for x in ['interaction', 'vs', 'advantage'])]
        }
        
        for category, features in feature_categories.items():
            if features:
                print(f"\n{category} ({len(features)} features):")
                for feature in features[:5]:  # Show top 5
                    print(f"  - {feature}")
                if len(features) > 5:
                    print(f"  ... and {len(features) - 5} more")
    
    def predict_match_enhanced(self, opponent, competition, venue, 
                             opponent_strength=70, rest_days=4, key_players_absent=0,
                             madrid_elo=2000, opponent_elo=1800, form_points=2.0):
        """
        Enhanced match prediction with all advanced features
        """
        if self.model is None:
            print("Enhanced model not trained yet!")
            return None
        
        # Create feature vector (simplified - in practice you'd calculate all features)
        features = np.zeros(len(self.selected_features))
        
        # This is a simplified version - you'd need to calculate all the advanced features
        # For demonstration, we'll use some basic mappings
        feature_dict = {
            'opponent_strength': opponent_strength,
            'venue_encoded': 1 if venue == 'Home' else 0,
            'competition_encoded': {'La Liga': 0, 'Copa del Rey': 1, 'Champions League': 2}[competition],
            'rest_days': rest_days,
            'key_players_absent': key_players_absent,
            'elo_diff_dynamic': madrid_elo - opponent_elo,
            'form_points': form_points
        }
        
        # Fill in available features
        for i, feature_name in enumerate(self.selected_features):
            if feature_name in feature_dict:
                features[i] = feature_dict[feature_name]
        
        # Predict
        probabilities = self.model.predict_proba([features])[0]
        
        result = {
            'Draw': round(probabilities[0] * 100, 1),
            'Loss': round(probabilities[1] * 100, 1),
            'Win': round(probabilities[2] * 100, 1)
        }
        
        return result
    
    def save_enhanced_model(self, filepath='enhanced_real_madrid_model.pkl'):
        """
        Save enhanced model
        """
        model_data = {
            'model': self.model,
            'feature_selector': self.feature_selector,
            'selected_features': self.selected_features,
            'feature_importance': self.feature_importance
        }
        joblib.dump(model_data, filepath)
        print(f"Enhanced model saved to {filepath}")

def main():
    """
    Main training pipeline for enhanced model
    """
    print("=== Enhanced Real Madrid Match Predictor ===\n")
    
    # Create enhanced dataset if it doesn't exist
    try:
        df = pd.read_csv('real_madrid_matches_enhanced.csv')
        print("Enhanced dataset found!")
    except FileNotFoundError:
        print("Creating enhanced dataset...")
        df = enhance_dataset_with_advanced_features()
    
    # Initialize enhanced predictor
    predictor = EnhancedRealMadridPredictor()
    
    # Prepare enhanced data
    print("\nPreparing enhanced features...")
    X, y, feature_cols, df_full = predictor.prepare_enhanced_data()
    print(f"Total features available: {len(feature_cols)}")
    
    # Feature selection
    print("\nSelecting best features...")
    X_selected = predictor.select_best_features(X, y, k=30)
    
    # Train enhanced model
    print("\nTraining enhanced XGBoost model...")
    X_train, X_val, y_train, y_val = predictor.train_enhanced_model(X_selected, y)
    
    # Evaluate
    print("\nEvaluating enhanced model...")
    accuracy, logloss = predictor.evaluate_enhanced_model(X_val, y_val)
    
    # Feature importance
    print("\nTop 15 Most Important Features:")
    top_features = predictor.plot_enhanced_feature_importance(top_n=15)
    print(top_features.head(15))
    
    # Model comparison
    predictor.compare_models()
    
    # Save enhanced model
    predictor.save_enhanced_model()
    
    print(f"\nEnhanced model training completed!")
    print(f"Final accuracy: {accuracy:.4f}")
    print(f"Final log loss: {logloss:.4f}")

if __name__ == "__main__":
    main()
