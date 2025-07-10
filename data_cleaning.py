import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import seaborn as sns



f1_laps = pd.read_csv("data/f1_laps_simple.csv")
f1_results = pd.read_csv("data/f1_results_simple.csv")


# Fill missing values with appropriate strategies
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


#Default grid postion
f1_laps_filled['Position'].fillna(15, inplace=True)



#Outlier detection
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

#save 
f1_laps_filled.to_csv("data/f1_laps_cleaned.csv", index=False)
f1_results_filled.to_csv("data/f1_results_cleaned.csv", index=False)