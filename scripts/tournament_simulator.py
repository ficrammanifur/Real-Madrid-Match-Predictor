import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ucl_simulation import UCLMonteCarloSimulator
import random

class FullTournamentSimulator(UCLMonteCarloSimulator):
    def __init__(self, model_path='enhanced_real_madrid_model.pkl'):
        super().__init__(model_path)
        self.knockout_teams = []
        
    def simulate_knockout_match(self, opponent, venue, opponent_strength, is_final=False):
        """
        Simulate knockout match (with extra time/penalties if needed)
        """
        # Regular time
        result, _ = self.simulate_single_match(opponent, venue, opponent_strength)
        
        if result != 'Draw':
            return result
        
        # Extra time (slightly favor stronger team)
        madrid_strength = 88 + (3 if venue == 'Home' else 0)
        strength_diff = madrid_strength - opponent_strength
        
        # Extra time probabilities (more decisive)
        if strength_diff > 5:
            et_win_prob = 0.6
        elif strength_diff > 0:
            et_win_prob = 0.55
        elif strength_diff > -5:
            et_win_prob = 0.45
        else:
            et_win_prob = 0.4
        
        rand = np.random.random()
        if rand < et_win_prob:
            return 'Win'
        elif rand < et_win_prob + 0.2:  # 20% still draw after ET
            # Penalties (50-50 with slight experience bonus for Madrid)
            penalty_prob = 0.55 if not is_final else 0.6  # Experience in finals
            return 'Win' if np.random.random() < penalty_prob else 'Loss'
        else:
            return 'Loss'
    
    def simulate_two_leg_tie(self, opponent, opponent_strength, madrid_home_first=True):
        """
        Simulate two-leg knockout tie
        """
        if madrid_home_first:
            leg1_venue, leg2_venue = 'Home', 'Away'
        else:
            leg1_venue, leg2_venue = 'Away', 'Home'
        
        # Leg 1
        leg1_result, leg1_points = self.simulate_single_match(opponent, leg1_venue, opponent_strength)
        
        # Leg 2 
        leg2_result, leg2_points = self.simulate_single_match(opponent, leg2_venue, opponent_strength)
        
        # Calculate aggregate
        total_points = leg1_points + leg2_points
        
        if total_points > 3:  # More wins than losses
            return 'Win', (leg1_result, leg2_result)
        elif total_points < 3:
            return 'Loss', (leg1_result, leg2_result)
        else:
            # Tied on aggregate - simulate decisive match
            # Use away goals or go to extra time
            decisive_venue = leg2_venue  # Second leg venue
            decisive_result = self.simulate_knockout_match(opponent, decisive_venue, opponent_strength)
            return decisive_result, (leg1_result, leg2_result)
    
    def simulate_full_tournament(self, swiss_fixtures=None, n_simulations=1000):
        """
        Simulate complete UCL tournament from Swiss phase to final
        """
        tournament_results = []
        
        for sim in range(n_simulations):
            # Swiss phase
            if swiss_fixtures is None:
                fixtures = self.generate_swiss_fixtures()
            else:
                fixtures = swiss_fixtures
            
            swiss_results = self.simulate_swiss_phase(fixtures, n_simulations=1)
            swiss_points = swiss_results[0]['total_points']
            
            # Determine Swiss phase outcome
            if swiss_points >= 16:
                swiss_position = 'Top 8'
                next_round = 'Round of 16'
            elif swiss_points >= 9:
                swiss_position = 'Playoffs'
                next_round = 'Playoff Round'
            else:
                swiss_position = 'Eliminated'
                next_round = None
            
            tournament_result = {
                'simulation': sim + 1,
                'swiss_points': swiss_points,
                'swiss_position': swiss_position,
                'furthest_round': swiss_position if next_round is None else next_round
            }
            
            # Continue if qualified
            if next_round is not None:
                current_round = next_round
                
                # Knockout phase simulation
                while current_round is not None:
                    # Determine opponent strength based on round
                    if current_round == 'Playoff Round':
                        opponent_strength = random.randint(70, 80)  # Medium teams
                    elif current_round == 'Round of 16':
                        opponent_strength = random.randint(75, 85)  # Strong teams
                    elif current_round == 'Quarter-finals':
                        opponent_strength = random.randint(80, 87)  # Very strong
                    elif current_round == 'Semi-finals':
                        opponent_strength = random.randint(82, 88)  # Elite
                    else:  # Final
                        opponent_strength = random.randint(84, 88)  # Best teams
                    
                    # Simulate knockout tie
                    if current_round == 'Final':
                        # Single match final
                        venue = 'Neutral'  # Treat as slight away disadvantage
                        result = self.simulate_knockout_match('Final Opponent', venue, opponent_strength, is_final=True)
                    else:
                        # Two-leg tie
                        result, _ = self.simulate_two_leg_tie('Knockout Opponent', opponent_strength)
                    
                    tournament_result['furthest_round'] = current_round
                    
                    if result == 'Win':
                        # Advance to next round
                        if current_round == 'Playoff Round':
                            current_round = 'Round of 16'
                        elif current_round == 'Round of 16':
                            current_round = 'Quarter-finals'
                        elif current_round == 'Quarter-finals':
                            current_round = 'Semi-finals'
                        elif current_round == 'Semi-finals':
                            current_round = 'Final'
                        elif current_round == 'Final':
                            tournament_result['furthest_round'] = 'Winner'
                            current_round = None
                    else:
                        # Eliminated
                        current_round = None
            
            tournament_results.append(tournament_result)
        
        return tournament_results
    
    def analyze_tournament_outcomes(self, tournament_results):
        """
        Analyze full tournament simulation results
        """
        total_sims = len(tournament_results)
        
        # Count outcomes
        outcome_counts = {}
        for result in tournament_results:
            outcome = result['furthest_round']
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        
        # Calculate percentages
        outcome_analysis = {}
        for outcome, count in outcome_counts.items():
            percentage = (count / total_sims) * 100
            outcome_analysis[outcome] = {
                'count': count,
                'percentage': percentage
            }
        
        return outcome_analysis
    
    def plot_tournament_analysis(self, tournament_results):
        """
        Create comprehensive tournament analysis plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Swiss phase points distribution
        swiss_points = [result['swiss_points'] for result in tournament_results]
        axes[0, 0].hist(swiss_points, bins=range(0, 25), alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(np.mean(swiss_points), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(swiss_points):.1f}')
        axes[0, 0].set_xlabel('Swiss Phase Points')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Swiss Phase Performance')
        axes[0, 0].legend()
        
        # 2. Tournament outcomes pie chart
        outcome_analysis = self.analyze_tournament_outcomes(tournament_results)
        
        # Order outcomes by tournament progression
        outcome_order = ['Eliminated', 'Playoffs', 'Round of 16', 'Quarter-finals', 
                        'Semi-finals', 'Final', 'Winner']
        
        labels = []
        sizes = []
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(outcome_order)))
        
        for outcome in outcome_order:
            if outcome in outcome_analysis:
                labels.append(f"{outcome}\n{outcome_analysis[outcome]['percentage']:.1f}%")
                sizes.append(outcome_analysis[outcome]['percentage'])
        
        axes[0, 1].pie(sizes, labels=labels, colors=colors[:len(sizes)], 
                      autopct='', startangle=90)
        axes[0, 1].set_title('Tournament Outcomes Distribution')
        
        # 3. Success rate by Swiss performance
        swiss_categories = {
            'Top 8 (16+ pts)': [],
            'Playoffs (9-15 pts)': [],
            'Eliminated (0-8 pts)': []
        }
        
        for result in tournament_results:
            points = result['swiss_points']
            furthest = result['furthest_round']
            
            if points >= 16:
                swiss_categories['Top 8 (16+ pts)'].append(furthest)
            elif points >= 9:
                swiss_categories['Playoffs (9-15 pts)'].append(furthest)
            else:
                swiss_categories['Eliminated (0-8 pts)'].append(furthest)
        
        # Calculate success rates (reaching at least QF)
        success_rates = {}
        for category, outcomes in swiss_categories.items():
            if outcomes:
                successful = sum(1 for outcome in outcomes 
                               if outcome in ['Quarter-finals', 'Semi-finals', 'Final', 'Winner'])
                success_rates[category] = (successful / len(outcomes)) * 100
            else:
                success_rates[category] = 0
        
        categories = list(success_rates.keys())
        rates = list(success_rates.values())
        
        bars = axes[1, 0].bar(categories, rates, color=['gold', 'silver', 'lightcoral'])
        axes[1, 0].set_ylabel('Success Rate (%)')
        axes[1, 0].set_title('Quarter-final+ Rate by Swiss Performance')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{rate:.1f}%', ha='center', va='bottom')
        
        # 4. Detailed outcome breakdown
        axes[1, 1].axis('off')
        
        # Create summary table
        table_data = []
        total_sims = len(tournament_results)
        
        for outcome in outcome_order:
            if outcome in outcome_analysis:
                count = outcome_analysis[outcome]['count']
                percentage = outcome_analysis[outcome]['percentage']
                table_data.append([outcome, f"{count:,}", f"{percentage:.1f}%"])
        
        table = axes[1, 1].table(cellText=table_data,
                                colLabels=['Outcome', 'Count', 'Percentage'],
                                cellLoc='center',
                                loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        axes[1, 1].set_title('Detailed Tournament Outcomes')
        
        plt.tight_layout()
        plt.show()
        
        return outcome_analysis

def main():
    """
    Run full tournament simulation
    """
    print("=== Real Madrid Full UCL Tournament Simulation ===\n")
    
    # Initialize simulator
    simulator = FullTournamentSimulator()
    
    # Run tournament simulation
    print("Running 5,000 full tournament simulations...")
    print("(This includes Swiss phase + knockout rounds)")
    
    tournament_results = simulator.simulate_full_tournament(n_simulations=5000)
    
    # Analyze results
    outcome_analysis = simulator.analyze_tournament_outcomes(tournament_results)
    
    print(f"\n=== Tournament Simulation Results ===")
    print(f"Total simulations: {len(tournament_results):,}")
    
    print(f"\n=== Outcome Probabilities ===")
    outcome_order = ['Winner', 'Final', 'Semi-finals', 'Quarter-finals', 
                    'Round of 16', 'Playoffs', 'Eliminated']
    
    for outcome in outcome_order:
        if outcome in outcome_analysis:
            data = outcome_analysis[outcome]
            print(f"{outcome:<15}: {data['percentage']:>6.1f}% ({data['count']:,} times)")
    
    # Key statistics
    advanced_count = sum(outcome_analysis.get(outcome, {'count': 0})['count'] 
                        for outcome in ['Quarter-finals', 'Semi-finals', 'Final', 'Winner'])
    advanced_pct = (advanced_count / len(tournament_results)) * 100
    
    final_count = sum(outcome_analysis.get(outcome, {'count': 0})['count'] 
                     for outcome in ['Final', 'Winner'])
    final_pct = (final_count / len(tournament_results)) * 100
    
    winner_count = outcome_analysis.get('Winner', {'count': 0})['count']
    winner_pct = (winner_count / len(tournament_results)) * 100
    
    print(f"\n=== Key Statistics ===")
    print(f"Reach Quarter-finals or better: {advanced_pct:.1f}%")
    print(f"Reach Final: {final_pct:.1f}%")
    print(f"Win Tournament: {winner_pct:.1f}%")
    
    # Plot analysis
    simulator.plot_tournament_analysis(tournament_results)
    
    return tournament_results, outcome_analysis

if __name__ == "__main__":
    main()
