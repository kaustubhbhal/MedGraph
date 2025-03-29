import pandas as pd
from pathlib import Path
from scipy.stats import zscore
import numpy as np

# Configuration
input_dir = Path("./Maryland")  # Update this path
output_dir = Path("cleaned_data")
output_dir.mkdir(exist_ok=True)

# Define columns to keep for each file type
columns_to_keep = {
    "conditions": ["START", "STOP", "PATIENT", "ENCOUNTER", "CODE", "SYSTEM"],
    "encounters": ["Id", "START", "STOP", "PATIENT", "CODE"],
    "medications": ["START", "STOP", "PATIENT", "ENCOUNTER", "CODE"],
    "observations": ["DATE", "PATIENT", "ENCOUNTER", "CODE", "VALUE"],  # Will be specially processed
    "patients": ["Id", "BIRTHDATE", "DEATHDATE", "FIRST", "LAST", "GENDER", "RACE", "ADDRESS"],
    "procedures": ["START", "STOP", "PATIENT", "ENCOUNTER", "SYSTEM", "CODE", "DESCRIPTION"]
}

def clean_observations():
    """Special cleaning for observations data with robust value normalization"""
    file_prefix = "observations"
    input_file = input_dir / f"{file_prefix}.csv"
    output_file = output_dir / f"{file_prefix}_cleaned.csv"
    
    try:
        # Load the observations data
        df = pd.read_csv(input_file)
        
        # Keep only required columns
        columns = columns_to_keep[file_prefix]
        available_columns = [col for col in columns if col in df.columns]
        missing_columns = set(columns) - set(available_columns)
        
        if missing_columns:
            print(f"Warning: {file_prefix}.csv is missing columns: {missing_columns}")
        
        df = df[available_columns]
        
        # Remove rows with missing ENCOUNTER values
        original_count = len(df)
        df = df.dropna(subset=['ENCOUNTER'])
        removed_encounter = original_count - len(df)
        print(f"Removed {removed_encounter} rows with missing ENCOUNTER values.")
        
        # Convert VALUE to numeric and remove invalid/missing
        df['VALUE'] = pd.to_numeric(df['VALUE'], errors='coerce')
        original_count = len(df)
        df = df.dropna(subset=['VALUE'])
        removed_value = original_count - len(df)
        print(f"Removed {removed_value} rows with non-numeric or missing VALUEs. {len(df)} rows remaining.")
        
        # Robust z-score calculation with edge case handling
        def safe_zscore(group):
            """Calculate z-score with edge case handling"""
            if len(group) < 2:
                return np.nan  # Not enough data for z-score
            std = group.std(ddof=1)
            if std == 0:  # All values identical
                return 0.0  # Consider identical values as "average" (z=0)
            return (group - group.mean()) / std
        
        # Calculate z-scores
        df['VALUE_ZSCORE'] = df.groupby('CODE')['VALUE'].transform(safe_zscore)
        
        # Generate diagnostic report
        diagnostic_report = df.groupby('CODE').agg(
            observation_count=('VALUE', 'size'),
            unique_values=('VALUE', 'nunique'),
            zscore_missing=('VALUE_ZSCORE', lambda x: x.isna().sum())
        ).sort_values('zscore_missing', ascending=False)
        
        # Print summary statistics
        total_codes = len(diagnostic_report)
        codes_with_missing = len(diagnostic_report[diagnostic_report['zscore_missing'] > 0])
        single_obs_codes = len(diagnostic_report[diagnostic_report['observation_count'] == 1])
        identical_value_codes = len(diagnostic_report[
            (diagnostic_report['unique_values'] == 1) & 
            (diagnostic_report['observation_count'] > 1)
        ])
        
        print("\nZ-score Normalization Report:")
        print(f"Total unique codes: {total_codes}")
        print(f"Codes with missing z-scores: {codes_with_missing}")
        print(f"  - Due to single observation: {single_obs_codes}")
        print(f"  - Due to identical values: {identical_value_codes}")
        print(f"Valid codes with complete z-scores: {total_codes - codes_with_missing}")
        
        # Optionally: Save diagnostic report
        diagnostic_report.to_csv(output_dir / "zscore_diagnostics.csv")
        print(f"Saved detailed z-score diagnostics to zscore_diagnostics.csv")
        
        # Save cleaned file
        df.to_csv(output_file, index=False)
        print(f"Created {output_file} with robust z-score normalization.")
        
    except FileNotFoundError:
        print(f"Error: {input_file} not found")
    except Exception as e:
        print(f"Error processing {file_prefix}: {str(e)}")

def clean_standard_csv(file_prefix, columns):
    """Clean and save a standard CSV file keeping only specified columns"""
    input_file = input_dir / f"{file_prefix}.csv"
    output_file = output_dir / f"{file_prefix}_cleaned.csv"
    
    try:
        df = pd.read_csv(input_file)
        
        # Find which requested columns actually exist in the file
        available_columns = [col for col in columns if col in df.columns]
        missing_columns = set(columns) - set(available_columns)
        
        if missing_columns:
            print(f"Warning: {file_prefix}.csv is missing columns: {missing_columns}")
        
        # Keep only available requested columns
        df = df[available_columns]
        
        # Save cleaned file
        df.to_csv(output_file, index=False)
        print(f"Created {output_file} with columns: {available_columns}")
        
    except FileNotFoundError:
        print(f"Error: {input_file} not found")
    except Exception as e:
        print(f"Error processing {file_prefix}: {str(e)}")

# Process all files
for file_type, columns in columns_to_keep.items():
    if file_type == "observations":
        clean_observations()
    else:
        clean_standard_csv(file_type, columns)

print("\nCleaning complete. Check the 'cleaned_data' directory.")