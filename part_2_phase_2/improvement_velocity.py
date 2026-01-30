import pandas as pd

fighters_df = pd.read_csv("csv/fighter_level_data.csv")  # Load dataset

# Compute fight time
fighters_df["fight_time_sec"] = (fighters_df["finish_round"] - 1) * 300 + fighters_df["match_time_sec"]
fighters_df["fight_time_min"] = fighters_df["fight_time_sec"] / 60

wweight_for_strikers = [0.35, 0.35, 0.05, 0.05, 0.20]
weight_for_grapplers = [0.05, 0.05, 0.35, 0.35, 0.20]
weight_for_balanced = [0.20, 0.20, 0.20, 0.20, 0.20]

# Performance metrics used:
# 1. strike_diff_per_min      -> net striking dominance per minute
# 2. sig_str_acc              -> striking efficiency
# 3. td_acc                   -> takedown efficiency
# 4. control_fraction         -> fraction of fight spent in control

fighters_df["sig_str_landed_per_min"] = fighters_df["sig_str_landed"] / fighters_df["fight_time_min"]  
fighters_df["sig_str_absorbed_per_min"] = fighters_df["sig_str_absorbed"] / fighters_df["fight_time_min"]  
fighters_df["td_landed_per_min"] = fighters_df["td_landed"] / fighters_df["fight_time_min"] 
fighters_df["control_fraction"] = fighters_df["ctrl"] / fighters_df["fight_time_sec"]
fighters_df["strike_diff_per_min"] = fighters_df["sig_str_landed_per_min"] - fighters_df["sig_str_absorbed_per_min"]
fighters_df["td_acc_fight"] = (
    fighters_df["td_landed"] / fighters_df["td_atmpted"]
)
fighters_df["td_acc_fight"] = fighters_df["td_acc_fight"].fillna(0)

#zscores for all metric
fighters_df["strike_diff_z"] = ((fighters_df["strike_diff_per_min"] - fighters_df["strike_diff_per_min"].mean())
                                 / fighters_df["strike_diff_per_min"].std())
fighters_df["strike_acc_z"] = ((fighters_df["sig_str_acc"] - fighters_df["sig_str_acc"].mean())
                                / fighters_df["sig_str_acc"].std())
fighters_df["td_acc_fight_z"] = (
    fighters_df["td_acc_fight"] - fighters_df["td_acc_fight"].mean()
) / fighters_df["td_acc_fight"].std()
fighters_df["control_fraction_z"] = (
    fighters_df["control_fraction"] - fighters_df["control_fraction"].mean()
) / fighters_df["control_fraction"].std()

def group_fight_style(row):
    # Thresholds (interpretable, not learned)
    TD_ATTEMPTS_PM = 0.4      
    STR_LANDED_PM = 3.5       

    td_attempts_pm = row['td_atmpted'] / row['fight_time_min']
    sig_landed_pm = row['sig_str_landed'] / row['fight_time_min']

    if td_attempts_pm >= TD_ATTEMPTS_PM and sig_landed_pm < STR_LANDED_PM:
        return 'Grappler'

    
    if sig_landed_pm >= STR_LANDED_PM and td_attempts_pm < TD_ATTEMPTS_PM:
        return 'Striker'

    return 'Balanced'

fighters_df['style'] = fighters_df.apply(group_fight_style, axis=1)

# Compute style-weighted performance score
def compute_style_performance_score(row):
    win_flag = row.get('win_flag', 0)
    if row['style'] == 'Striker':
        weights = wweight_for_strikers
    elif row['style'] == 'Grappler':
        weights = weight_for_grapplers
    else:
        weights = weight_for_balanced

    score = (
        row['strike_diff_z'] * weights[0] +
        row['strike_acc_z'] * weights[1] +
        row['td_acc_fight_z'] * weights[2] +
        row['control_fraction_z'] * weights[3] +
        win_flag * weights[4]
    )
    return score

fighters_df['style_performance_score'] = fighters_df.apply(compute_style_performance_score, axis=1)

# Min-max scaling to 0-100
min_score = fighters_df['style_performance_score'].min()
max_score = fighters_df['style_performance_score'].max()
fighters_df['performance_0_100'] = 100 * (fighters_df['style_performance_score'] - min_score) / (max_score - min_score)

mean_perf = fighters_df['performance_0_100'].mean()
std_perf = fighters_df['performance_0_100'].std()

def performance_label(performance):
    if performance >= mean_perf + 1.5*std_perf:
        return "Exceptional dominance"
    elif performance >= mean_perf + 1.0*std_perf:
        return "Elite dominance"
    elif performance >= mean_perf + 0.5*std_perf:
        return "Good dominance"
    elif performance > mean_perf - 0.5*std_perf:
        return "Okay dominance"
    elif performance > mean_perf - 1.0*std_perf:
        return "Below Average dominance"
    else:
        return "Poor dominance"

fighters_df['performance_category'] = fighters_df['performance_0_100'].apply(performance_label)

print(fighters_df[['name', 'style', 'performance_0_100', 'performance_category']].head())

fighter_name = "Ilia Topuria"
adesanya_fights = fighters_df[fighters_df["name"] == fighter_name].sort_values("fight_number")
print(
    adesanya_fights[
        ["fight_number", "event_name", "style", "performance_0_100", "performance_category"]
    ]
)