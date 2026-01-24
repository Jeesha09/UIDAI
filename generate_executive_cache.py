import pandas as pd
import numpy as np
import glob
import os
import json
from data_preprocessing import preprocess_dataframe
from backend_logic import ExecutiveSummaryEngine

def read_all_csvs_in_folder(folder_name):
    """
    Finds all .csv files in a folder and merges them into one DataFrame.
    """
    file_paths = glob.glob(os.path.join(folder_name, "*.csv"))
    
    if not file_paths:
        return pd.DataFrame()
    
    df_list = []
    for file in file_paths:
        try:
            temp_df = pd.read_csv(file, low_memory=False)
            df_list.append(temp_df)
        except Exception as e:
            print(f"Skipping bad file: {file}")
    
    if df_list:
        full_df = pd.concat(df_list, ignore_index=True)
        return full_df
    else:
        return pd.DataFrame()

print("Loading enrollment data...")
df_enrol = read_all_csvs_in_folder("api_data_aadhar_enrolment")
df_bio = read_all_csvs_in_folder("api_data_aadhar_biometric")
df_demo = read_all_csvs_in_folder("api_data_aadhar_demographic")

print("Preprocessing data...")
if not df_enrol.empty:
    df_enrol = preprocess_dataframe(df_enrol, "Enrollment")
if not df_bio.empty:
    df_bio = preprocess_dataframe(df_bio, "Biometric")
if not df_demo.empty:
    df_demo = preprocess_dataframe(df_demo, "Demographic")

print("Initializing Executive Summary Engine...")
exec_engine = ExecutiveSummaryEngine(df_enrol, df_bio, df_demo)

print("Generating Executive Summary data...")

# 1. Early Warning System
print("  → Calculating early warning metrics...")
early_warning = exec_engine.get_early_warning_system()

# 2. Stagnation Detection
print("  → Detecting stagnant pincodes...")
stagnation = exec_engine.get_stagnation_detection()

# 3. Peer Benchmarking
print("  → Computing peer benchmarks...")
benchmarks = exec_engine.get_peer_benchmarking()

# 4. Location Clusters
print("  → Clustering locations...")
clusters = exec_engine.get_location_clusters()

# 5. Calculate total enrollment volume
age_cols = [col for col in df_enrol.columns if col.startswith('age_')]
total_vol = float(df_enrol[age_cols].sum().sum())

# 6. Store basic stats
total_enrollments = len(df_enrol)
total_districts = int(df_enrol['district'].nunique())
total_states = int(df_enrol['state'].nunique())

print("\nPreparing data for JSON export...")

# Convert DataFrames to JSON-serializable format
executive_cache = {
    "early_warning": {
        "metric_value": int(early_warning["metric_value"]),
        "details_df": early_warning["details_df"].reset_index().to_dict(orient='records')
    },
    "stagnation": {
        "total_stagnant": int(stagnation["total_stagnant"]),
        "pincode_list": [str(p) for p in stagnation["pincode_list"][:100]]  # Limit to first 100
    },
    "benchmarks": {
        "top_performers": benchmarks["top_performers"].to_dict(orient='records'),
        "bottom_performers": benchmarks["bottom_performers"].to_dict(orient='records')
    },
    "clusters": clusters.to_dict(orient='records') if not clusters.empty else [],
    "total_volume": total_vol,
    "stats": {
        "total_enrollments": total_enrollments,
        "total_districts": total_districts,
        "total_states": total_states
    }
}

print("Saving to executive_cache.json...")
with open('executive_cache.json', 'w') as f:
    json.dump(executive_cache, f, indent=2)

print("✅ Executive Summary cache generated successfully!")
print(f"   - Critical decline districts: {executive_cache['early_warning']['metric_value']}")
print(f"   - Stagnant pincodes: {executive_cache['stagnation']['total_stagnant']}")
print(f"   - Total enrollment volume: {executive_cache['total_volume']:,.0f}")
print(f"   - Districts analyzed: {total_districts}")
