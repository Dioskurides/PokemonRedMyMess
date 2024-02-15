from pathlib import Path
import pandas as pd
import numpy as np
# from red_gym_env import RedGymEnv  # Uncomment this if RedGymEnv is needed later in the script

def print_available_run_indices():
    sess_id = input("Enter session ID: ")
    instance_id = input("Enter instance ID: ")

    sess_path = Path(f'session_{sess_id}')
    tdf = pd.read_csv(f"{sess_path}/agent_stats_{instance_id}.csv.gz", compression='gzip', low_memory=False)
    
    # Remove rows where the 'map' column might be incorrectly filled or used as placeholder
    tdf = tdf[tdf['map'] != 'map']
    
    # Find the indices where a new run starts
    new_run_indices = tdf.index[tdf["step"].astype(int) == 0].tolist()
    
    # Print the run indices
    print("Available run indices:")
    for i, _ in enumerate(new_run_indices):
        print(i)

    # Optionally, return the list of indices if needed for further processing
    return new_run_indices

# Now call the function directly
if __name__ == "__main__":
    print_available_run_indices()
