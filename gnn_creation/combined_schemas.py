import pandas as pd

def combine_schemas():
    # Load the observation data from each state
    # Assuming the CSV files are in the same directory as this script
    # Adjust the file paths as necessary

    co_observation_df = pd.read_csv("co_data/observations.csv")
    ca_observation_df = pd.read_csv("ca_data/observations.csv")
    tx_observation_df = pd.read_csv("tx_data/observations.csv")
    ma_observation_df = pd.read_csv("ma_data/observations.csv")


    co_observation_codes = co_observation_df["CODE"].unique()
    ca_observation_codes = ca_observation_df["CODE"].unique()
    tx_observation_codes = tx_observation_df["CODE"].unique()
    ma_observation_codes = ma_observation_df["CODE"].unique()

    # Combine all observation codes into a single set
    all_observation_codes = set(
        co_observation_codes.tolist()
        + ca_observation_codes.tolist()
        + tx_observation_codes.tolist()
        + ma_observation_codes.tolist()
    )
    # Create a mapping from each observation code to a unique index
    observation_code_to_index = {
        code: idx for idx, code in enumerate(all_observation_codes)
    }
    # Create a reverse mapping from index to observation code
    index_to_observation_code = {
        idx: code for code, idx in observation_code_to_index.items()
    }

    return observation_code_to_index, index_to_observation_code