import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import joblib
from collections import defaultdict
import random

class UCLMonteCarloSimulator:
    def __init__(self, model_path='enhanced_real_madrid_model.pkl'):
        """
        Initialize UCL Monte Carlo simulator with trained model
        """
        try:
            self.model_data = joblib.load(model_path)
            self.model = self.model_data['model']
            self.selected_features = self.model_data['selected_features']
            print(f"Loaded enhanced model from {model_path}")
        except FileNotFoundError:
            print(f"Model file {model_path} not found. Using basic probabilities.")
            self.model = None
            
        # UCL 2024-25 format: 36 teams, Swiss system
        self.ucl_teams = self._initialize_ucl_teams()
        
    def _initialize_ucl_teams(self):
        """
        Initialize UCL teams with their strengths for 2024-25 season
        """
        teams = {
            # Pot 1 (Top seeds)
            'Real Madrid': 88,
            'Manchester City': 86,
            'Bayern Munich': 84,
            'PSG': 81,
            'Liverpool': 83,
            'Inter Milan': 75,
            'Borussia Dortmund': 78,
            'RB Leipzig': 76,
            'Barcelona': 85,
            
            # Pot 2 (Strong teams)
            'Atletico Madrid': 82,
            'Atalanta': 74,
            'Juventus': 77,
            'Benfica': 73,
            'Arsenal': 78,
            'Club Brugge': 68,
            'Shakhtar Donetsk': 70,
            'AC Milan': 76,
            'Feyenoord': 69,
            
            # Pot 3 (Medium strength)
            'Sporting CP': 71,
            'PSV Eindhoven': 70,
            'Dinamo Zagreb': 65,
            'Red Bull Salzburg': 67,
            'Lille': 68,
            'Red Star Belgrade': 64,
            'Young Boys': 62,
            'Celtic': 66,
            'Slovan Bratislava': 58,
            
            # Pot 4 (Emerging teams)
            'Monaco': 72,
            'Aston Villa': 69,
            'Bologna': 67,
            'Girona': 66,
            'Stuttgart': 68,
            'Sturm Graz': 61,
            'Brest': 63,
            'Sparta Prague': 60
        }
        return teams
    
    def generate_swiss_fixtures(self, madrid_opponents=None):
        """
        Generate realistic Swiss system fixtures for Real Madrid
        """
        if madrid_opponents is None:
            # Select 8 opponents from different pots for realistic draw
            pot1 = ['Manchester City', 'Bayern Munich', 'PSG', 'Liverpool', 'Barcelona']
            pot2 = ['Atletico Madrid', 'Atalanta', 'Juventus', 'Arsenal', 'AC Milan']
            pot3 = ['Sporting CP', 'PSV Eindhoven', 'Lille', 'Celtic']
            pot4 = ['Monaco', 'Aston Villa', 'Bologna', 'Girona', 'Stuttgart']
            
            # Select 2 from each pot (realistic UCL draw)
            opponents = (
                random.sample(pot1, 2) + 
                random.sample(pot2, 2) + 
                random.sample(pot3, 2) + 
                random.sample(pot4, 2)
            )
        else:
            opponents = madrid_opponents
        
        # Generate home/away split (4 home, 4 away)
        venues = ['Home'] * 4 + ['Away'] * 4
        random.shuffle(venues)
        
        fixtures = []
        for i, opponent in enumerate(opponents):
            fixtures.append({
                'opponent': opponent,
                'opponent_strength': self.ucl_teams[opponent],
                'venue': venues[i],
                'competition': 'Champions League'
            })
        
        return fixtures
    
    def predict_match_probability(self, opponent, venue, opponent_strength):
        """
        Predict match outcome probabilities
        """
        if self.model is not None:
            # Use trained model (simplified feature vector)
            features = np.zeros(len(self.selected_features))
            
            # Basic feature mapping
            feature_dict = {
                'opponent_strength': opponent_strength,
                'venue_encoded': 1 if venue == 'Home' else 0,
                'competition_encoded': 2,  # Champions League
                'rest_days': 4,  # Average rest
                'key_players_absent': 0,  # Assume full squad
                'elo_diff_dynamic': 88 - opponent_strength,  # Madrid base strength
                'form_points': 2.0,  # Good form
                'competition_pressure': 1.5  # UCL pressure
            }
            
            # Fill available features
            for i, feature_name in enumerate(self.selected_features):
                if feature_name in feature_dict:
                    features[i] = feature_dict[feature_name]
            
            probabilities = self.model.predict_proba([features])[0]
            return {
                'Draw': probabilities[0],
                'Loss': probabilities[1], 
                'Win': probabilities[2]
            }
        else:
            # Fallback to rule-based probabilities
            madrid_strength = 88
            if venue == 'Home':
                madrid_strength += 3
            
            strength_diff = madrid_strength - opponent_strength
            
            if strength_diff > 10:
                return {'Win': 0.65, 'Draw': 0.25, 'Loss': 0.10}
            elif strength_diff > 5:
                return {'Win': 0.55, 'Draw': 0.30, 'Loss': 0.15}
            elif strength_diff > 0:
                return {'Win': 0.45, 'Draw': 0.35, 'Loss': 0.20}
            elif strength_diff > -5:
                return {'Win': 0.35, 'Draw': 0.35, 'Loss': 0.30}
            else:
                return {'Win': 0.25, 'Draw': 0.30, 'Loss': 0.45}
    
    def simulate_single_match(self, opponent, venue, opponent_strength):
        """
        Simulate a single match outcome
        """
        probabilities = self.predict_match_probability(opponent, venue, opponent_strength)
        
        rand = np.random.random()
        if rand < probabilities['Win']:
            return 'Win', 3
        elif rand < probabilities['Win'] + probabilities['Draw']:
            return 'Draw', 1
        else:
            return 'Loss', 0
    
    def simulate_swiss_phase(self, fixtures, n_simulations=10000):
        """
        Simulate Swiss phase multiple times
        """
        results = []
        
        for sim in range(n_simulations):
            total_points = 0
            wins = 0
            draws = 0
            losses = 0
            match_results = []
            
            for fixture in fixtures:
                result, points = self.simulate_single_match(
                    fixture['opponent'], 
                    fixture['venue'], 
                    fixture['opponent_strength']
                )
                
                total_points += points
                if result == 'Win':
                    wins += 1
                elif result == 'Draw':
                    draws += 1
                else:
                    losses += 1
                
                match_results.append({
                    'opponent': fixture['opponent'],
                    'venue': fixture['venue'],
                    'result': result,
                    'points': points
                })
            
            results.append({
                'simulation': sim + 1,
                'total_points': total_points,
                'wins': wins,
                'draws': draws,
                'losses': losses,
                'matches': match_results
            })
        
        return results
    
    def analyze_qualification_chances(self, simulation_results):
        """
        Analyze qualification chances based on points distribution
        """
        points_distribution = [result['total_points'] for result in simulation_results]
        
        # UCL Swiss system qualification thresholds (estimated)
        thresholds = {
            'Top 8 (Direct R16)': 16,  # ~5-6 wins
            'Top 24 (Playoffs)': 9,    # ~3 wins
            'Elimination': 8           # Below 9 points
        }
        
        analysis = {}
        total_sims = len(simulation_results)
        
        for category, threshold in thresholds.items():
            if category == 'Elimination':
                count = sum(1 for points in points_distribution if points <= threshold)
            else:
                count = sum(1 for points in points_distribution if points >= threshold)
            
            percentage = (count / total_sims) * 100
            analysis[category] = {
                'count': count,
                'percentage': percentage
            }
        
        return analysis, points_distribution
    
    def plot_simulation_results(self, simulation_results, fixtures):
        """
        Create comprehensive visualization of simulation results
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Points distribution
        points = [result['total_points'] for result in simulation_results]
        axes[0, 0].hist(points, bins=range(0, 25), alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(np.mean(points), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(points):.1f}')
        axes[0, 0].axvline(16, color='green', linestyle='-', alpha=0.7, 
                          label='Top 8 Threshold')
        axes[0, 0].axvline(9, color='orange', linestyle='-', alpha=0.7, 
                          label='Playoff Threshold')
        axes[0, 0].set_xlabel('Total Points')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Points Distribution (10,000 simulations)')
        axes[0, 0].legend()
        
        # 2. Win distribution
        wins = [result['wins'] for result in simulation_results]
        axes[0, 1].hist(wins, bins=range(0, 9), alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(np.mean(wins), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(wins):.1f}')
        axes[0, 1].set_xlabel('Number of Wins')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Wins Distribution')
        axes[0, 1].legend()
        
        # 3. Results by opponent strength
        opponent_results = defaultdict(list)
        for result in simulation_results:
            for match in result['matches']:
                opponent_strength = next(f['opponent_strength'] for f in fixtures 
                                       if f['opponent'] == match['opponent'])
                opponent_results[match['opponent']].append(match['result'])
        
        # Calculate win percentage by opponent
        opponent_win_pct = {}
        for opponent, results in opponent_results.items():
            win_pct = (results.count('Win') / len(results)) * 100
            opponent_strength = next(f['opponent_strength'] for f in fixtures 
                                   if f['opponent'] == opponent)
            opponent_win_pct[opponent] = {'win_pct': win_pct, 'strength': opponent_strength}
        
        opponents = list(opponent_win_pct.keys())
        win_pcts = [opponent_win_pct[opp]['win_pct'] for opp in opponents]
        strengths = [opponent_win_pct[opp]['strength'] for opp in opponents]
        
        scatter = axes[0, 2].scatter(strengths, win_pcts, alpha=0.7, s=100)
        axes[0, 2].set_xlabel('Opponent Strength')
        axes[0, 2].set_ylabel('Win Percentage (%)')
        axes[0, 2].set_title('Win % vs Opponent Strength')
        
        # Add opponent labels
        for i, opponent in enumerate(opponents):
            axes[0, 2].annotate(opponent.split()[-1], (strengths[i], win_pcts[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 4. Venue performance
        home_results = []
        away_results = []
        
        for result in simulation_results:
            home_points = sum(match['points'] for match in result['matches'] 
                            if next(f['venue'] for f in fixtures 
                                  if f['opponent'] == match['opponent']) == 'Home')
            away_points = sum(match['points'] for match in result['matches'] 
                            if next(f['venue'] for f in fixtures 
                                  if f['opponent'] == match['opponent']) == 'Away')
            home_results.append(home_points)
            away_results.append(away_points)
        
        axes[1, 0].hist([home_results, away_results], bins=range(0, 13), 
                       alpha=0.7, label=['Home', 'Away'], edgecolor='black')
        axes[1, 0].set_xlabel('Points')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Points Distribution by Venue')
        axes[1, 0].legend()
        
        # 5. Qualification probability pie chart
        analysis, _ = self.analyze_qualification_chances(simulation_results)
        
        labels = []
        sizes = []
        colors = ['gold', 'lightblue', 'lightcoral']
        
        for i, (category, data) in enumerate(analysis.items()):
            labels.append(f"{category}\n{data['percentage']:.1f}%")
            sizes.append(data['percentage'])
        
        axes[1, 1].pie(sizes, labels=labels, colors=colors, autopct='', startangle=90)
        axes[1, 1].set_title('Qualification Chances')
        
        # 6. Expected results table
        axes[1, 2].axis('off')
        
        # Create summary table
        table_data = []
        for fixture in fixtures:
            probs = self.predict_match_probability(
                fixture['opponent'], fixture['venue'], fixture['opponent_strength']
            )
            table_data.append([
                fixture['opponent'][:12],  # Truncate long names
                fixture['venue'][0],  # H/A
                f"{probs['Win']*100:.0f}%",
                f"{probs['Draw']*100:.0f}%", 
                f"{probs['Loss']*100:.0f}%"
            ])
        
        table = axes[1, 2].table(cellText=table_data,
                                colLabels=['Opponent', 'V', 'Win%', 'Draw%', 'Loss%'],
                                cellLoc='center',
                                loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        axes[1, 2].set_title('Match Probabilities')
        
        plt.tight_layout()
        plt.show()
    
    def run_full_simulation(self, madrid_opponents=None, n_simulations=10000):
        """
        Run complete UCL simulation for Real Madrid
        """
        print("=== Real Madrid UCL Swiss Phase Simulation ===\n")
        
        # Generate fixtures
        fixtures = self.generate_swiss_fixtures(madrid_opponents)
        
        print("Generated fixtures:")
        for i, fixture in enumerate(fixtures, 1):
            print(f"{i}. vs {fixture['opponent']} ({fixture['venue']}) - Strength: {fixture['opponent_strength']}")
        
        print(f"\nRunning {n_simulations:,} simulations...")
        
        # Run simulations
        results = self.simulate_swiss_phase(fixtures, n_simulations)
        
        # Analyze results
        analysis, points_dist = self.analyze_qualification_chances(results)
        
        print(f"\n=== Simulation Results ===")
        print(f"Average points: {np.mean(points_dist):.1f}")
        print(f"Median points: {np.median(points_dist):.1f}")
        print(f"Standard deviation: {np.std(points_dist):.1f}")
        
        print(f"\n=== Qualification Chances ===")
        for category, data in analysis.items():
            print(f"{category}: {data['percentage']:.1f}% ({data['count']:,} simulations)")
        
        # Plot results
        self.plot_simulation_results(results, fixtures)
        
        return results, analysis, fixtures

def main():
    """
    Main simulation runner
    """
    # Initialize simulator
    simulator = UCLMonteCarloSimulator()
    
    # Option 1: Random draw
    print("Option 1: Random realistic draw")
    results1, analysis1, fixtures1 = simulator.run_full_simulation(n_simulations=10000)
    
    # Option 2: Tough draw
    tough_opponents = [
        'Manchester City', 'Bayern Munich', 'Barcelona', 'Liverpool',
        'Arsenal', 'Atletico Madrid', 'AC Milan', 'Juventus'
    ]
    
    print("\n" + "="*60)
    print("Option 2: Tough draw scenario")
    results2, analysis2, fixtures2 = simulator.run_full_simulation(
        madrid_opponents=tough_opponents, n_simulations=10000
    )
    
    # Option 3: Favorable draw
    favorable_opponents = [
        'PSG', 'Inter Milan', 'Atalanta', 'Benfica',
        'Sporting CP', 'Celtic', 'Monaco', 'Aston Villa'
    ]
    
    print("\n" + "="*60)
    print("Option 3: Favorable draw scenario")
    results3, analysis3, fixtures3 = simulator.run_full_simulation(
        madrid_opponents=favorable_opponents, n_simulations=10000
    )
    
    # Compare scenarios
    print("\n" + "="*60)
    print("=== SCENARIO COMPARISON ===")
    
    scenarios = [
        ("Random Draw", analysis1),
        ("Tough Draw", analysis2), 
        ("Favorable Draw", analysis3)
    ]
    
    print(f"{'Scenario':<15} {'Top 8':<10} {'Playoffs':<10} {'Elimination':<12}")
    print("-" * 50)
    
    for name, analysis in scenarios:
        top8 = analysis['Top 8 (Direct R16)']['percentage']
        playoffs = analysis['Top 24 (Playoffs)']['percentage']
        elimination = analysis['Elimination']['percentage']
        print(f"{name:<15} {top8:>6.1f}%   {playoffs:>6.1f}%   {elimination:>8.1f}%")

if __name__ == "__main__":
    main()
