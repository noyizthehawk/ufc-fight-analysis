import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load prepared data
fighter_df = pd.read_csv("csv/fighter_level_data.csv")

print(f"\n{'='*60}")
print(f"PRIME WINDOW DETECTION ANALYSIS")
print(f"{'='*60}")
print(f"Loaded {len(fighter_df)} fighter-fight records")
print(f"Unique fighters: {fighter_df['name'].nunique()}")
