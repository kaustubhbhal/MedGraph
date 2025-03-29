import pandas as pd
from pathlib import Path

# Configuration
input_dir = Path("./Maryland")  # Update this path
output_dir = Path("cleaned_data")
output_dir.mkdir(exist_ok=True)

# Define columns to keep for each file type
columns_to_keep = {
    "conditions": ["START", "STOP", "PATIENT", "ENCOUNTER", "CODE", "SYSTEM"],
    "encounters": ["Id", "START", "STOP", "PATIENT", "CODE"],
    "medications": ["START", "STOP", "PATIENT", "ENCOUNTER", "CODE"],
    "observations": ["DATE", "PATIENT", "ENCOUNTER", "CODE", "VALUE"],
    "patients": ["Id", "BIRTHDATE", "DEATHDATE", "FIRST", "LAST", "GENDER", "RACE", "ADDRESS"],
    "procedures": ["START", "STOP", "PATIENT", "ENCOUNTER", "SYSTEM", "CODE", "DESCRIPTION"]
}

def clean_csv(file_prefix, columns):
    """Clean and save a CSV file keeping only specified columns"""
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
    clean_csv(file_type, columns)

print("\nCleaning complete. Check the 'cleaned_data' directory.")