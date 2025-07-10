import fastf1
import pandas as pd
from pathlib import Path
import time

# Setup cache for faster loading
cache_dir = Path("./fastf1_cache")
cache_dir.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(cache_dir))

def collect_single_race(year, race_name):
    """Collect data for one race - simple version"""
    print(f"Getting data for {year} {race_name}...")
    
    try:
        # Load the race session
        session = fastf1.get_session(year, race_name, 'R')  # 'R' = Race
        session.load()
        
        # Get basic lap data
        laps = session.laps
        
        # Create simple lap data
        lap_data = []
        for index, lap in laps.iterrows():
            lap_info = {
                'Year': year,
                'Race': race_name,
                'Driver': lap['Driver'],
                'Team': lap['Team'],
                'LapNumber': lap['LapNumber'],
                'LapTime_seconds': lap['LapTime'].total_seconds() if pd.notna(lap['LapTime']) else None,
                'Position': lap['Position'],
                'TireCompound': lap['Compound'],
                'TireAge': lap['TyreLife']
            }
            lap_data.append(lap_info)
        
        # Get race results
        results = session.results
        
        # Create simple results data
        result_data = []
        for index, result in results.iterrows():
            result_info = {
                'Year': year,
                'Race': race_name,
                'Driver': result['BroadcastName'],
                'Team': result['TeamName'],
                'Position': result['Position'],
                'GridPosition': result['GridPosition'],
                'Points': result['Points'],
                'Status': result['Status']  # Finished, DNF, etc.
            }
            result_data.append(result_info)
        
        print(f"✓ Got {len(lap_data)} laps and {len(result_data)} results")
        return lap_data, result_data
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return [], []

def collect_multiple_races():
    """Collect data for multiple races"""
    
    races_to_collect = [
        (2024, 1),
        (2024, 2), 
        (2024, 3),
        (2024, 4),
        (2024, 5),
        (2024, 6),
        (2024, 7),
        (2024, 8),
        (2024, 9),
        (2024, 10),
        (2024, 11),
        (2024, 12),
        (2024, 13),
        (2024, 14),
        (2024, 15),
        (2024, 16),
        (2024, 17),
        (2024, 18),
        (2024, 19),
        (2024, 20),
        (2024, 21),
        (2024, 22),
        (2024, 23),
        (2024, 24),
        (2025, 1),
        (2025, 2),
        (2025, 3),
        (2025, 4),
        (2025, 5),
        (2025, 6),
        (2025, 7),
        (2025, 8),
        (2025, 9),
        (2025, 10),
        (2025, 11),
        (2025, 12)
    ]
    
    all_laps = []
    all_results = []
    
    for year, race in races_to_collect:
        print(f"\nCollecting {race} {year}...")
        
        lap_data, result_data = collect_single_race(year, race)
        
        # Add to our collections
        all_laps.extend(lap_data)
        all_results.extend(result_data)
        
        # Small delay to be nice to the API
        time.sleep(2)
    
    return all_laps, all_results

def save_data(lap_data, result_data):
    """Save data to CSV files"""
    
    # Create data folder
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    
    # Convert to DataFrames
    laps_df = pd.DataFrame(lap_data)
    results_df = pd.DataFrame(result_data)
    
    # Save as CSV files
    laps_df.to_csv(data_dir / "f1_laps_simple.csv", index=False)
    results_df.to_csv(data_dir / "f1_results_simple.csv", index=False)
    
    print(f"\n✓ Saved data:")
    print(f"  - Laps: {len(laps_df)} records")
    print(f"  - Results: {len(results_df)} records")
    
    return laps_df, results_df



if __name__ == "__main__":
    print("Simple F1 Data Collection")
    print("========================")
    
    # Collect the data
    lap_data, result_data = collect_multiple_races()
    
    # Save to files
    laps_df, results_df = save_data(lap_data, result_data)
    
    print("\nDone!")