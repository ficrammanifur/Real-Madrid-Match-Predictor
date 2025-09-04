import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json

class AdvancedFeatureEngine:
    def __init__(self):
        self.player_ratings = self._initialize_player_ratings()
        self.injury_impact_weights = {
            'GK': 0.15,   # Goalkeeper
            'DEF': 0.20,  # Defender  
            'MID': 0.25,  # Midfielder
            'FWD': 0.30   # Forward
        }
        
    def _initialize_player_ratings(self):
        """
        Initialize Real Madrid key players with their ratings and positions
        """
        return {
            'Courtois': {'rating': 89, 'position': 'GK', 'importance': 0.9},
            'Carvajal': {'rating': 84, 'position': 'DEF', 'importance': 0.8},
            'Militao': {'rating': 83, 'position': 'DEF', 'importance': 0.85},
            'Alaba': {'rating': 84, 'position': 'DEF', 'importance': 0.8},
            'Mendy': {'rating': 82, 'position': 'DEF', 'importance': 0.75},
            'Modric': {'rating': 87, 'position': 'MID', 'importance': 0.9},
            'Kroos': {'rating': 86, 'position': 'MID', 'importance': 0.85},
            'Casemiro': {'rating': 85, 'position': 'MID', 'importance': 0.9},
            'Valverde': {'rating': 83, 'position': 'MID', 'importance': 0.8},
            'Vinicius': {'rating': 86, 'position': 'FWD', 'importance': 0.9},
            'Benzema': {'rating': 91, 'position': 'FWD', 'importance': 0.95},
            'Rodrygo': {'rating': 81, 'position': 'FWD', 'importance': 0.75},
            'Bellingham': {'rating': 84, 'position': 'MID', 'importance': 0.85},
            'Mbappe': {'rating': 91, 'position': 'FWD', 'importance': 0.95}
        }
    
    def calculate_dynamic_elo(self, df):
        """
        Enhanced Elo calculation with momentum and form factors
        """
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        madrid_elo = 2000
        opponent_elos = {}
        
        madrid_elo_history = []
        opponent_elo_history = []
        elo_momentum = []
        
        # Track recent performance for momentum
        recent_results = []
        
        for idx, row in df.iterrows():
            opponent = row['opponent']
            
            # Initialize opponent Elo if not seen before
            if opponent not in opponent_elos:
                base_elo = 1800
                # Adjust based on opponent strength
                strength_adjustment = (row['opponent_strength'] - 70) * 4
                opponent_elos[opponent] = base_elo + strength_adjustment
            
            opponent_elo = opponent_elos[opponent]
            
            # Calculate momentum factor from last 5 games
            momentum_factor = 0
            if len(recent_results) >= 3:
                recent_points = [3 if r == 'Win' else 1 if r == 'Draw' else 0 
                               for r in recent_results[-5:]]
                momentum_factor = (np.mean(recent_points) - 1.5) * 20  # -30 to +30
            
            # Adjust Elo with momentum
            adjusted_madrid_elo = madrid_elo + momentum_factor
            
            # Store values
            madrid_elo_history.append(madrid_elo)
            opponent_elo_history.append(opponent_elo)
            elo_momentum.append(momentum_factor)
            
            # Calculate K-factor based on match importance
            base_k = 32
            if row['competition'] == 'Champions League':
                k_factor = base_k * 1.5
            elif row['competition'] == 'Copa del Rey':
                k_factor = base_k * 0.8
            else:
                k_factor = base_k
            
            # Expected scores
            expected_madrid = 1 / (1 + 10**((opponent_elo - adjusted_madrid_elo) / 400))
            expected_opponent = 1 - expected_madrid
            
            # Actual scores
            if row['result'] == 'Win':
                actual_madrid, actual_opponent = 1, 0
            elif row['result'] == 'Draw':
                actual_madrid, actual_opponent = 0.5, 0.5
            else:
                actual_madrid, actual_opponent = 0, 1
            
            # Update Elos
            madrid_elo += k_factor * (actual_madrid - expected_madrid)
            opponent_elos[opponent] += k_factor * (actual_opponent - expected_opponent)
            
            # Update recent results
            recent_results.append(row['result'])
            if len(recent_results) > 10:
                recent_results.pop(0)
        
        df['madrid_elo_dynamic'] = madrid_elo_history
        df['opponent_elo_dynamic'] = opponent_elo_history
        df['elo_momentum'] = elo_momentum
        df['elo_diff_dynamic'] = df['madrid_elo_dynamic'] - df['opponent_elo_dynamic']
        
        return df
    
    def calculate_advanced_xg_features(self, df):
        """
        Calculate advanced xG-based features
        """
        df = df.copy()
        
        # Rolling xG efficiency over different windows
        for window in [3, 5, 10]:
            # Goals vs xG efficiency
            df[f'xg_efficiency_{window}'] = (
                df['madrid_goals'].rolling(window=window, min_periods=1).sum() /
                (df['xg_madrid'].rolling(window=window, min_periods=1).sum() + 0.1)
            )
            
            # Defensive xG efficiency
            df[f'xg_defensive_{window}'] = (
                df['opponent_goals'].rolling(window=window, min_periods=1).sum() /
                (df['xg_opponent'].rolling(window=window, min_periods=1).sum() + 0.1)
            )
            
            # xG difference trend
            df[f'xg_diff_{window}'] = (
                df['xg_madrid'].rolling(window=window, min_periods=1).mean() -
                df['xg_opponent'].rolling(window=window, min_periods=1).mean()
            )
        
        # xG variance (consistency measure)
        df['xg_variance_5'] = df['xg_madrid'].rolling(window=5, min_periods=1).std().fillna(0)
        
        # Over/under-performance streaks
        df['xg_overperformance'] = df['madrid_goals'] - df['xg_madrid']
        df['xg_overperf_streak'] = df['xg_overperformance'].rolling(window=3, min_periods=1).mean()
        
        return df
    
    def calculate_player_absence_impact(self, df):
        """
        Calculate detailed impact of player absences
        """
        df = df.copy()
        
        # Simulate realistic injury patterns
        np.random.seed(42)  # For reproducible results
        
        absence_features = []
        
        for idx, row in df.iterrows():
            # Base absence count from original data
            base_absences = row['key_players_absent']
            
            # Simulate which specific players are absent
            absent_players = []
            if base_absences > 0:
                # Higher probability for certain players to be injured
                injury_probabilities = {
                    'Courtois': 0.1, 'Carvajal': 0.15, 'Militao': 0.12,
                    'Alaba': 0.18, 'Mendy': 0.14, 'Modric': 0.20,
                    'Kroos': 0.08, 'Casemiro': 0.12, 'Valverde': 0.10,
                    'Vinicius': 0.12, 'Benzema': 0.15, 'Rodrygo': 0.10,
                    'Bellingham': 0.08, 'Mbappe': 0.10
                }
                
                # Select absent players based on probabilities
                for player, prob in injury_probabilities.items():
                    if np.random.random() < prob and len(absent_players) < base_absences:
                        absent_players.append(player)
                
                # Fill remaining absences with random players
                remaining_players = [p for p in self.player_ratings.keys() 
                                   if p not in absent_players]
                while len(absent_players) < base_absences and remaining_players:
                    absent_players.append(np.random.choice(remaining_players))
                    remaining_players.remove(absent_players[-1])
            
            # Calculate impact scores
            total_rating_loss = 0
            positional_impact = {'GK': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
            importance_loss = 0
            
            for player in absent_players:
                if player in self.player_ratings:
                    player_data = self.player_ratings[player]
                    total_rating_loss += player_data['rating']
                    positional_impact[player_data['position']] += 1
                    importance_loss += player_data['importance']
            
            # Calculate weighted impact
            weighted_impact = 0
            for pos, count in positional_impact.items():
                weighted_impact += count * self.injury_impact_weights[pos]
            
            absence_features.append({
                'absent_players_count': len(absent_players),
                'total_rating_loss': total_rating_loss,
                'importance_loss': importance_loss,
                'weighted_positional_impact': weighted_impact,
                'gk_absent': positional_impact['GK'],
                'def_absent': positional_impact['DEF'],
                'mid_absent': positional_impact['MID'],
                'fwd_absent': positional_impact['FWD']
            })
        
        # Add features to dataframe
        absence_df = pd.DataFrame(absence_features)
        for col in absence_df.columns:
            df[col] = absence_df[col]
        
        return df
    
    def calculate_tactical_features(self, df):
        """
        Calculate tactical and contextual features
        """
        df = df.copy()
        
        # Match context features
        df['days_since_last_match'] = df['rest_days']
        df['fixture_congestion'] = (df['rest_days'] < 4).astype(int)
        
        # Season timing features
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['season_start'] = ((df['month'] >= 8) | (df['month'] <= 1)).astype(int)
        df['season_end'] = (df['month'] >= 4).astype(int)
        df['winter_break'] = (df['month'] == 1).astype(int)
        
        # Competition pressure
        competition_pressure = {
            'La Liga': 1.0,
            'Champions League': 1.5,
            'Copa del Rey': 0.8
        }
        df['competition_pressure'] = df['competition'].map(competition_pressure)
        
        # Opponent familiarity (how often played recently)
        opponent_frequency = df.groupby('opponent').cumcount() + 1
        df['opponent_familiarity'] = opponent_frequency
        
        # Venue streak (consecutive home/away games)
        df['venue_streak'] = df.groupby((df['venue'] != df['venue'].shift()).cumsum()).cumcount() + 1
        
        # Goal difference momentum
        df['goal_diff'] = df['madrid_goals'] - df['opponent_goals']
        df['goal_diff_momentum'] = df['goal_diff'].rolling(window=3, min_periods=1).mean()
        
        # Performance vs similar opponents
        df['opponent_tier'] = pd.cut(df['opponent_strength'], 
                                   bins=[0, 65, 75, 85, 100], 
                                   labels=[0, 1, 2, 3])  # Weak, Medium, Strong, Elite
        
        # Calculate performance vs each tier
        for tier in range(4):
            tier_mask = df['opponent_tier'] == tier
            if tier_mask.sum() > 0:
                tier_performance = df[tier_mask]['goal_diff'].expanding().mean()
                df[f'performance_vs_tier_{tier}'] = 0
                df.loc[tier_mask, f'performance_vs_tier_{tier}'] = tier_performance
        
        return df
    
    def create_interaction_features(self, df):
        """
        Create interaction features between different variables
        """
        df = df.copy()
        
        # Elo and venue interaction
        df['elo_home_advantage'] = df['elo_diff_dynamic'] * df['venue_encoded']
        
        # Form and competition interaction
        df['form_ucl_pressure'] = df['form_points'] * (df['competition'] == 'Champions League').astype(int)
        
        # Rest and opponent strength interaction
        df['rest_vs_strong_opponent'] = df['rest_days'] * (df['opponent_strength'] > 80).astype(int)
        
        # Injury impact and competition
        df['injury_ucl_impact'] = df['weighted_positional_impact'] * df['competition_pressure']
        
        # xG efficiency and venue
        df['xg_efficiency_home'] = df['xg_efficiency_5'] * df['venue_encoded']
        
        # Momentum and pressure situations
        df['momentum_pressure'] = df['elo_momentum'] * df['competition_pressure']
        
        return df

def enhance_dataset_with_advanced_features(csv_path='real_madrid_matches.csv'):
    """
    Main function to enhance dataset with all advanced features
    """
    print("Loading dataset...")
    df = pd.read_csv(csv_path)
    
    print("Initializing advanced feature engine...")
    afe = AdvancedFeatureEngine()
    
    print("Calculating dynamic Elo ratings...")
    df = afe.calculate_dynamic_elo(df)
    
    print("Adding advanced xG features...")
    df = afe.calculate_advanced_xg_features(df)
    
    print("Calculating player absence impact...")
    df = afe.calculate_player_absence_impact(df)
    
    print("Adding tactical features...")
    df = afe.calculate_tactical_features(df)
    
    print("Creating interaction features...")
    df = afe.create_interaction_features(df)
    
    # Save enhanced dataset
    enhanced_path = 'real_madrid_matches_enhanced.csv'
    df.to_csv(enhanced_path, index=False)
    
    print(f"Enhanced dataset saved to {enhanced_path}")
    print(f"New features added: {len(df.columns) - 15}")  # Original had ~15 columns
    print(f"Total features: {len(df.columns)}")
    
    # Display new feature summary
    new_features = [col for col in df.columns if col not in [
        'date', 'opponent', 'opponent_strength', 'competition', 'venue',
        'result', 'madrid_goals', 'opponent_goals', 'possession', 'shots',
        'shots_on_target', 'xg_madrid', 'xg_opponent', 'rest_days', 'key_players_absent'
    ]]
    
    print(f"\nNew advanced features ({len(new_features)}):")
    for i, feature in enumerate(new_features, 1):
        print(f"{i:2d}. {feature}")
    
    return df

if __name__ == "__main__":
    enhanced_df = enhance_dataset_with_advanced_features()
    
    # Show sample of enhanced data
    print("\nSample of enhanced dataset:")
    print(enhanced_df[['date', 'opponent', 'result', 'elo_diff_dynamic', 
                      'xg_efficiency_5', 'weighted_positional_impact', 
                      'competition_pressure']].head(10))
