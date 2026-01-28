import pandas as pd
# HOW FIGHTERS CHANGE OVER TIME

def red_corner_fighters(ufc_dataset):
    columns = [col for col in ufc_dataset.columns if not col.startswith('b_') and col != 'referee']  # Get all columns that don't contain b_ and aren't referee
    red_df = ufc_dataset[columns].copy()  # Filter to those columns
    red_df['won'] = (ufc_dataset['winner_id'] == ufc_dataset['r_id']).astype(int)  # Create win indicator - compare winner_id with r_id
    red_df['opponent_name'] = ufc_dataset['b_name']  # Add opponent info (from blue corner)
    red_df['opponent_id'] = ufc_dataset['b_id']
    red_df['sig_str_absorbed'] = ufc_dataset['b_sig_str_landed']  # Add opponent's strikes (becomes "absorbed" for red fighter)
    rename_dict = {col: col.replace('r_', '') for col in red_df.columns if col.startswith('r_')}  # Rename columns to remove r_ prefix
    red_df = red_df.rename(columns=rename_dict)
    return red_df

def blue_corner_fighters(ufc_dataset):
    columns = [col for col in ufc_dataset.columns if not col.startswith('r_') and col != 'referee']  # Get all columns that don't contain r_ and aren't referee
    blue_df = ufc_dataset[columns].copy()  # Filter to those columns
    blue_df['won'] = (ufc_dataset['winner_id'] == ufc_dataset['b_id']).astype(int)  # Create win indicator - compare winner_id with b_id
    blue_df['opponent_name'] = ufc_dataset['r_name']  # Add opponent info (from red corner)
    blue_df['opponent_id'] = ufc_dataset['r_id']
    blue_df['sig_str_absorbed'] = ufc_dataset['r_sig_str_landed']  # Add opponent's strikes (becomes "absorbed" for blue fighter)
    rename_dict = {col: col.replace('b_', '') for col in blue_df.columns if col.startswith('b_')}  # Rename columns to remove b_ prefix
    blue_df = blue_df.rename(columns=rename_dict)
    return blue_df

def merge_red_blue(red_df, blue_df):
    combined_df = pd.concat([red_df, blue_df], ignore_index=True)  # Concatenate red and blue dataframes
    columns_to_remove = ['sig_str_landed','sig_str_atmpted', 'sig_str_acc', 'total_str_landed', 'total_str_atmpted', 'total_str_acc', 'suatt', 'str_acc', 'str_def', 'suavg']
    combined_df = combined_df.drop(columns=columns_to_remove)
    return combined_df

if __name__ == "__main__":
    ufc_dataset = pd.read_csv("csv/UFC_clean.csv")
    
    red_df = red_corner_fighters(ufc_dataset)  # Create red corner dataframe
    blue_df = blue_corner_fighters(ufc_dataset)  # Create blue corner dataframe
    fighter_df = merge_red_blue(red_df, blue_df)  # Merge them
    
    fighter_df['date'] = pd.to_datetime(fighter_df['date'])  # Sort by fighter and date
    fighter_df = fighter_df.sort_values(['name', 'date']).reset_index(drop=True)
    
    fighter_df['fight_number'] = fighter_df.groupby('name').cumcount() + 1  # Add fight number for each fighter
    fighter_df['days_since_last_fight'] = fighter_df.groupby('name')['date'].diff().dt.days  # Days since last fight
    fighter_df['dob'] = pd.to_datetime(fighter_df['dob'], errors='coerce')  # Convert DOB to datetime
    fighter_df['age_at_fight'] = ((fighter_df['date'] - fighter_df['dob']).dt.days / 365.25).round(1)  # Calculate age at fight
    
    fighter_df['rolling_win_rate_3'] = (  # Rolling win rate last 3 fights (excluding current fight)
        fighter_df.groupby('name')['won']
        .shift(1)
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    
    fighter_df['rolling_win_rate_5'] = (  # Rolling win rate last 5 fights (excluding current fight)
        fighter_df.groupby('name')['won']
        .shift(1)
        .rolling(window=5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    
    fighter_df['rolling_sig_str_landed'] = (  # Rolling average strikes landed (last 3 fights)
        fighter_df.groupby('name')['sig_stlanded']
        .shift(1)
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    
    fighter_df['rolling_sig_str_absorbed'] = (  # Rolling average strikes absorbed (last 3 fights)
        fighter_df.groupby('name')['sig_str_absorbed']
        .shift(1)
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    
    # Check the result
    print(f"\n{'='*60}")
    print(f"DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Original fights: {len(ufc_dataset)}")
    print(f"Fighter-level rows: {len(fighter_df)} (should be ~2x fights)")
    print(f"Unique fighters: {fighter_df['name'].nunique()}")
    print(f"\n{'='*60}")
    print(f"SAMPLE DATA - Fighter Career View")
    print(f"{'='*60}")
    
    sample_fighter = fighter_df[fighter_df['name'] == fighter_df['name'].value_counts().index[0]]  # Show one fighter's full career progression
    print(f"\nFighter: {sample_fighter['name'].iloc[0]}")
    print(f"Total UFC Fights: {len(sample_fighter)}")
    print(sample_fighter[['date', 'fight_number', 'won', 'days_since_last_fight', 'age_at_fight', 'rolling_win_rate_3', 'rolling_sig_str_landed', 'rolling_sig_str_absorbed']].head(10))
    
    print(f"\n{'='*60}")
    print(f"COLUMNS IN DATASET")
    print(f"{'='*60}")
    print(fighter_df.columns.tolist())
    
    fighter_df.to_csv('csv/fighter_level_data.csv', index=False)  # Save it
    print(f"\n{'='*60}")
    print(f"Saved to 'fighter_level_data.csv'")
    print(f"{'='*60}\n")