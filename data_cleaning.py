import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import seaborn as sns

f1_laps = pd.read_csv("data/f1_laps_simple.csv")
f1_results = pd.read_csv("data/f1_results_simple.csv")

try:
    f1_qualifying = pd.read_csv("data/f1_qualifying_simple.csv")
    has_qualifying = True
    print("✓ Qualifying data loaded successfully")
except FileNotFoundError:
    f1_qualifying = pd.DataFrame()
    has_qualifying = False
    print("⚠ No qualifying data found. Run data_collection.py with qualifying first.")

f1_laps_filled = f1_laps.copy()
f1_results_filled = f1_results.copy()
f1_laps_filled['LapTime_seconds'] = f1_laps_filled.groupby(['Driver', 'Race'])['LapTime_seconds'].transform(
    lambda x: x.fillna(x.median())
)
f1_laps_filled['LapTime_seconds'].fillna(f1_laps_filled['LapTime_seconds'].median(), inplace=True)
f1_laps_filled['TireCompound'] = f1_laps_filled.groupby('Race')['TireCompound'].transform(
    lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'MEDIUM')
)
f1_laps_filled['TireAge'].fillna(0, inplace=True)
f1_laps_filled['Position'] = f1_laps_filled.groupby(['Driver', 'Race'])['Position'].fillna(method='ffill')
f1_laps_filled['Position'].fillna(15, inplace=True)

lap_times = f1_laps_filled['LapTime_seconds'].dropna()
Q1 = lap_times.quantile(0.25)  # 25th percentile
Q2 = lap_times.quantile(0.50)  # 50th percentile (median)
Q3 = lap_times.quantile(0.75)  # 75th percentile
IQR = Q3 - Q1

print("=== IQR EXPLAINED ===")
print(f"Q1 (25th percentile): {Q1:.2f} seconds")
print(f"Q2 (median):          {Q2:.2f} seconds") 
print(f"Q3 (75th percentile): {Q3:.2f} seconds")
print(f"IQR (Q3 - Q1):       {IQR:.2f} seconds")

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"\n=== OUTLIER BOUNDS ===")
print(f"Lower bound: {lower_bound:.2f} seconds")
print(f"Upper bound: {upper_bound:.2f} seconds")

#Find outliers
outliers = lap_times[(lap_times < lower_bound) | (lap_times > upper_bound)]
print(f"\nOutliers found: {len(outliers)} out of {len(lap_times)} lap times")
print(f"Percentage: {len(outliers)/len(lap_times)*100:.1f}%")

f1_laps_filled['Driver'] = f1_laps_filled['Driver'].astype('category')
f1_laps_filled['Team'] = f1_laps_filled['Team'].astype('category')

f1_results_filled['Status'] = f1_results_filled['Status'].astype('category')
f1_results_filled['Driver'] = f1_results_filled['Driver'].astype('category')
f1_results_filled['Team'] = f1_results_filled['Team'].astype('category')

if has_qualifying:
    f1_qualifying_filled = f1_qualifying.copy()
    
    for q_col in ['Q1', 'Q2', 'Q3']:
        if q_col in f1_qualifying_filled.columns:
            f1_qualifying_filled[q_col] = f1_qualifying_filled.groupby('Race')[q_col].transform(
                lambda x: x.fillna(x.median())
            )
    
    quali_times = f1_qualifying_filled[['Q1', 'Q2', 'Q3']].apply(pd.to_numeric, errors='coerce')
    f1_qualifying_filled['BestQualifyingTime'] = quali_times.min(axis=1)
    
    for race in f1_qualifying_filled['Race'].unique():
        race_mask = f1_qualifying_filled['Race'] == race
        pole_time = f1_qualifying_filled.loc[race_mask, 'BestQualifyingTime'].min()
        f1_qualifying_filled.loc[race_mask, 'GapToPole'] = f1_qualifying_filled.loc[race_mask, 'BestQualifyingTime'] - pole_time
    
    f1_qualifying_filled['QualifyingPerformance'] = (f1_qualifying_filled['QualifyingPosition'] / 20) * 100
    
    f1_qualifying_filled['Driver'] = f1_qualifying_filled['Driver'].astype('category')
    f1_qualifying_filled['Team'] = f1_qualifying_filled['Team'].astype('category')
    
    f1_qualifying_filled.to_csv("data/f1_qualifying_cleaned.csv", index=False)
    print("✓ Qualifying data cleaned and saved")
    
    quali_features = f1_qualifying_filled[['Year', 'Race', 'Driver', 'BestQualifyingTime', 'GapToPole', 'QualifyingPerformance']]
    f1_results_filled = f1_results_filled.merge(
        quali_features,
        on=['Year', 'Race', 'Driver'],
        how='left'
    )
    print("✓ Qualifying features merged into race results")

f1_laps_filled.to_csv("data/f1_laps_cleaned.csv", index=False)
f1_results_filled.to_csv("data/f1_results_cleaned.csv", index=False)

print("\n=== CLEANING SUMMARY ===")
print(f"Laps cleaned: {len(f1_laps_filled)} records")
print(f"Results cleaned: {len(f1_results_filled)} records")
if has_qualifying:
    print(f"Qualifying cleaned: {len(f1_qualifying_filled)} records")
    print(f"Qualifying features added to results: BestQualifyingTime, GapToPole, QualifyingPerformance")