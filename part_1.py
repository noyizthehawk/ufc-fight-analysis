import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency


TAKEDOWN_THRESHOLD = 1.0  # Fighters with td_avg >= 1.0 are classified as wrestlers

def reach_advantage(ufc_dataset, fighter_dataset):
    ufc_dataset['reach_advantage'] = ufc_dataset['r_reach'] - ufc_dataset['b_reach']
    ufc_dataset['red_win'] = (ufc_dataset['winner'] == ufc_dataset['r_name']).astype(int)
    
    bins = [-100, -20, -15, -10, -5, 0, 5, 10, 15, 20, 100]
    ufc_dataset['reach_bin'] = pd.cut(ufc_dataset['reach_advantage'], bins=bins)
    
    win_rate_by_bin = ufc_dataset.groupby('reach_bin')['red_win'].mean()
    
    print("\n=== MYTH #1: Reach Advantage ===")
    print("Win Rate by Reach Advantage:")
    print((win_rate_by_bin * 100).round(2))
    
    # statistical test
    contingency = pd.crosstab(ufc_dataset['reach_bin'], ufc_dataset['red_win'])
    chi2, p_value, dof, _ = chi2_contingency(contingency)
    print(f"\nChi-square test: p-value = {p_value:.4f}")
    if p_value < 0.05:
        print("✓ Reach advantage is statistically significant")
    else:
        print("✗ No significant effect detected")
    print("=" * 50)
    

    plt.figure(figsize=(12, 6))
    (win_rate_by_bin * 100).plot(kind='bar', color='steelblue', edgecolor='black')
    plt.axhline(50, color='red', linestyle='--', linewidth=2, label='50% baseline')
    plt.xlabel("Reach Advantage (cm)", fontsize=12)
    plt.ylabel("Red Win Percentage (%)", fontsize=12)
    plt.title("Myth #1: Does Reach Advantage Predict Wins?", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('myth1_reach_advantage.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return ufc_dataset, win_rate_by_bin


def youth_beat_experience(ufc_dataset, fighter_dataset):
    ufc_dataset['r_dob'] = pd.to_datetime(ufc_dataset['r_dob'], errors='coerce')
    ufc_dataset['b_dob'] = pd.to_datetime(ufc_dataset['b_dob'], errors='coerce')
    ufc_dataset['age_diff'] = (ufc_dataset['r_dob'] - ufc_dataset['b_dob']).dt.days/365.25
    ufc_dataset['red_win'] = (ufc_dataset['winner'] == ufc_dataset['r_name']).astype(int)
    
    bins = [-10, -5, -3, -2, -1, 0, 1, 2, 3, 5, 10]
    ufc_dataset['age_bin'] = pd.cut(ufc_dataset['age_diff'], bins=bins)
    
    win_rate_by_age_bin = ufc_dataset.groupby('age_bin')['red_win'].mean()
    
    print("\n=== MYTH #2: Youth Beats Experience ===")
    print("Win Rate by Age Difference:")
    print((win_rate_by_age_bin * 100).round(2))
    
    # Add statistical test
    contingency = pd.crosstab(ufc_dataset['age_bin'], ufc_dataset['red_win'])
    chi2, p_value, dof, _ = chi2_contingency(contingency)
    print(f"\nChi-square test: p-value = {p_value:.4f}")
    if p_value < 0.05:
        print("✓ Age difference is statistically significant")
    else:
        print("✗ No significant effect detected")
    print("=" * 50)
    
    # visualization
    plt.figure(figsize=(12, 6))
    (win_rate_by_age_bin * 100).plot(kind='bar', color='steelblue', edgecolor='black')
    plt.axhline(50, color='red', linestyle='--', linewidth=2, label='50% baseline')
    plt.xlabel("Age Difference (Red - Blue) in Years", fontsize=12)
    plt.ylabel("Red Win Percentage (%)", fontsize=12)
    plt.title("Myth #2: Does Youth Beat Experience?", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('myth2_youth_vs_experience.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return ufc_dataset, win_rate_by_age_bin


def wrestlers_vs_strikers(ufc_dataset, fighter_dataset):
    
    ufc_merged = ufc_dataset.copy()
    
    # Classify both fighters based on takedown average
    ufc_merged['r_style'] = ufc_merged['r_td_avg'].apply(
        lambda x: 'Wrestler' if pd.notna(x) and x >= TAKEDOWN_THRESHOLD else 'Striker'
    )
    ufc_merged['b_style'] = ufc_merged['b_td_avg'].apply(
        lambda x: 'Wrestler' if pd.notna(x) and x >= TAKEDOWN_THRESHOLD else 'Striker'
    )
    ufc_merged['matchup'] = ufc_merged['r_style'] + ' vs ' + ufc_merged['b_style']
    ufc_merged['red_win'] = (ufc_merged['winner'] == ufc_merged['r_name']).astype(int)
    
    print("\n=== MYTH #3: Wrestlers vs Strikers ===")
    
    # Win rates by all matchup types
    matchup_summary = ufc_merged.groupby('matchup').agg({
        'red_win': ['mean', 'count']
    })
    matchup_summary.columns = ['red_win_rate', 'n_fights']
    matchup_summary['red_win_pct'] = (matchup_summary['red_win_rate'] * 100).round(2)
    
    print("\nWin rates by matchup type:")
    print(matchup_summary[['red_win_pct', 'n_fights']])
    
    # Head-to-head: Wrestler vs Striker only
    wrestler_vs_striker = ufc_merged[
        ((ufc_merged['r_style'] == 'Wrestler') & (ufc_merged['b_style'] == 'Striker')) |
        ((ufc_merged['r_style'] == 'Striker') & (ufc_merged['b_style'] == 'Wrestler'))
    ].copy()
    
    # Who actually won?
    wrestler_vs_striker['wrestler_won'] = (
        ((wrestler_vs_striker['r_style'] == 'Wrestler') & (wrestler_vs_striker['red_win'] == 1)) |
        ((wrestler_vs_striker['b_style'] == 'Wrestler') & (wrestler_vs_striker['red_win'] == 0))
    ).astype(int)
    
    wrestler_win_rate = wrestler_vs_striker['wrestler_won'].mean()
    n_matchups = len(wrestler_vs_striker)
    
    print(f"\nHead-to-Head Results:")
    print(f"  Wrestler win rate: {wrestler_win_rate*100:.2f}%")
    print(f"  Striker win rate: {(1-wrestler_win_rate)*100:.2f}%")
    print(f"  Total matchups: {n_matchups}")
    
    # Statistical test
    from scipy.stats import binomtest
    result = binomtest(wrestler_vs_striker['wrestler_won'].sum(), n_matchups, 0.5)
    print(f"\nBinomial test: p-value = {result.pvalue:.4f}")
    if result.pvalue < 0.05:
        winner = "Wrestlers" if wrestler_win_rate > 0.5 else "Strikers"
        print(f"{winner} have a statistically significant advantage")
    else:
        print(" No significant difference between styles")
    print("=" * 50)
    
    # Visualization
    plt.figure(figsize=(10, 6))
    styles = ['Wrestler', 'Striker']
    win_rates = [wrestler_win_rate * 100, (1 - wrestler_win_rate) * 100]
    plt.bar(styles, win_rates, color=['#d62728', '#1f77b4'], 
            edgecolor='black', linewidth=2, alpha=0.7)
    plt.axhline(50, color='black', linestyle='--', linewidth=2, label='50% baseline')
    plt.ylabel('Win Rate (%) in Cross-Style Matchups', fontsize=12)
    plt.title('Myth #3: Wrestlers vs Strikers Head-to-Head', fontsize=14, fontweight='bold')
    plt.ylim(40, 60)
    plt.legend()
    plt.tight_layout()
    plt.savefig('myth3_wrestlers_vs_strikers.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return matchup_summary


def size_matters(ufc_dataset, fighter_dataset):
    # Impact of height and reach by division
    mens_division = ['flyweight', 'bantamweight', 'featherweight', 
                     'lightweight', 'welterweight', 'middleweight', 'light heavyweight', 'heavyweight']
    womens_divison = ["women's flyweight", "women's strawweight", "women's bantamweight"]
    
    print("\n=== MYTH #4: Size Matters by Division ===")
    
    results = []
    
    for division in mens_division + womens_divison:
        div_data = ufc_dataset[ufc_dataset['division'].str.lower() == division.lower()].copy()
        
        if len(div_data) < 30:  # Skip if too few fights
            continue
        
        div_data['height_advantage'] = div_data['r_height'] - div_data['b_height']
        div_data['reach_advantage'] = div_data['r_reach'] - div_data['b_reach']
        div_data['red_win'] = (div_data['winner'] == div_data['r_name']).astype(int)
        
        # Calculate correlation
        height_corr = div_data[['height_advantage', 'red_win']].corr().iloc[0, 1]
        reach_corr = div_data[['reach_advantage', 'red_win']].corr().iloc[0, 1]
        
        results.append({
            'Division': division.title(),
            'N_Fights': len(div_data),
            'Height_Corr': round(height_corr, 3),
            'Reach_Corr': round(reach_corr, 3)
        })
    
    results_df = pd.DataFrame(results)
    print("\nCorrelation: Physical Advantage → Win")
    print(results_df.to_string(index=False))
    print("=" * 50)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(results_df))
    width = 0.35
    
    ax.bar([i - width/2 for i in x], results_df['Height_Corr'], 
           width, label='Height', alpha=0.7, color='#2ecc71', edgecolor='black')
    ax.bar([i + width/2 for i in x], results_df['Reach_Corr'], 
           width, label='Reach', alpha=0.7, color='#3498db', edgecolor='black')
    ax.axhline(0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('Division', fontsize=12)
    ax.set_ylabel('Correlation with Winning', fontsize=12)
    ax.set_title('Myth #4: Does Size Impact Change by Division?', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['Division'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('myth4_size_by_division.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results_df


# Main
if __name__ == "__main__":
    print("=" * 50)
    print("UFC MYTH-BUSTING ANALYSIS")
    print("=" * 50)
    
    ufc_dataset = pd.read_csv("UFC_clean.csv")
    fighter_dataset = pd.read_csv("fighter_clean.csv")
    
    print(f"\nLoaded {len(ufc_dataset):,} fights")
    print(f"Loaded {len(fighter_dataset):,} fighters\n")
    
    # Run all analyses
    reach_advantage(ufc_dataset, fighter_dataset)
    youth_beat_experience(ufc_dataset, fighter_dataset)
    wrestlers_vs_strikers(ufc_dataset, fighter_dataset)
    size_matters(ufc_dataset, fighter_dataset)
    
    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE")
    print("=" * 50)
    print("\nGenerated files:")
    print("  - myth1_reach_advantage.png")
    print("  - myth2_youth_vs_experience.png")
    print("  - myth3_wrestlers_vs_strikers.png")
    print("  - myth4_size_by_division.png")