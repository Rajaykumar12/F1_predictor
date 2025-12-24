import fastf1
import pandas as pd
from pathlib import Path
import time

# Setup cache for faster loading
cache_dir = Path("./fastf1_cache")
cache_dir.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(cache_dir))

def collect_qualifying_data(year, race_name):
    """Collect qualifying data for one race"""
    print(f"  Getting qualifying data...")
    
    try:
        session = fastf1.get_session(year, race_name, 'Q')
        session.load()
        
        # Get qualifying results
        results = session.results
        
        # Create qualifying data
        qualifying_data = []
        for index, result in results.iterrows():
            quali_info = {
                'Year': year,
                'Race': race_name,
                'Driver': result['BroadcastName'],
                'Team': result['TeamName'],
                'QualifyingPosition': result['Position'],
                'Q1': result['Q1'].total_seconds() if pd.notna(result['Q1']) else None,
                'Q2': result['Q2'].total_seconds() if pd.notna(result['Q2']) else None,
                'Q3': result['Q3'].total_seconds() if pd.notna(result['Q3']) else None,
                'Status': result['Status']
            }
            qualifying_data.append(quali_info)
        
        print(f"  ✓ Got {len(qualifying_data)} qualifying results")
        return qualifying_data
        
    except Exception as e:
        print(f"  ✗ Error getting qualifying: {e}")
        return []

def collect_single_race(year, race_name):
    """Collect race, lap and qualifying data"""
    print(f"Getting data for {year} {race_name}...")
    
    try:
        session = fastf1.get_session(year, race_name, 'R')
        session.load()
        
        # Get basic lap data
        laps = session.laps
        
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
        
        results = session.results
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
                'Status': result['Status']
            }
            result_data.append(result_info)
        
        print(f"  ✓ Got {len(lap_data)} laps and {len(result_data)} results")
        
        qualifying_data = collect_qualifying_data(year, race_name)
        
        return lap_data, result_data, qualifying_data
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return [], [], []

def collect_multiple_races():
    """Collect 2025 season data (24 races)"""
    
    races_to_collect = [
        (2025, 1), (2025, 2), (2025, 3), (2025, 4), (2025, 5), (2025, 6),
        (2025, 7), (2025, 8), (2025, 9), (2025, 10), (2025, 11), (2025, 12),
        (2025, 13), (2025, 14), (2025, 15), (2025, 16), (2025, 17), (2025, 18),
        (2025, 19), (2025, 20), (2025, 21), (2025, 22), (2025, 23), (2025, 24)
    ]
    
    all_laps = []
    all_results = []
    all_qualifying = []
    
    for year, race in races_to_collect:
        print(f"\nCollecting {race} {year}...")
        
        lap_data, result_data, qualifying_data = collect_single_race(year, race)
        
        all_laps.extend(lap_data)
        all_results.extend(result_data)
        all_qualifying.extend(qualifying_data)
        time.sleep(2)
    
    return all_laps, all_results, all_qualifying

def save_data(lap_data, result_data, qualifying_data):
    """Save data to CSV files"""
    
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    
    laps_df = pd.DataFrame(lap_data)
    results_df = pd.DataFrame(result_data)
    qualifying_df = pd.DataFrame(qualifying_data)
    
    laps_df.to_csv(data_dir / "f1_laps_simple.csv", index=False)
    results_df.to_csv(data_dir / "f1_results_simple.csv", index=False)
    qualifying_df.to_csv(data_dir / "f1_qualifying_simple.csv", index=False)
    
    print(f"\n✓ Saved data:")
    print(f"  - Laps: {len(laps_df)} records")
    print(f"  - Results: {len(results_df)} records")
    print(f"  - Qualifying: {len(qualifying_df)} records")
    
    return laps_df, results_df, qualifying_df



if __name__ == "__main__":
    print("F1 Data Collection with Qualifying")
    print("===================================")
    print("Collecting race, lap, and qualifying data...")
    
    # Collect the data
    lap_data, result_data, qualifying_data = collect_multiple_races()
    
    # Save to files
    laps_df, results_df, qualifying_df = save_data(lap_data, result_data, qualifying_data)
    
    print("\n✓ Data collection complete!")
    print(f"  Total races processed: {len(results_df) // 20}")
    print(f"  Files saved to: data/")
    print("\nNext step: Run 'python data_cleaning.py' to process the data")
    print("\nDone!")