from pathlib import Path
import pandas as pd
import numpy as np
from red_gym_env import RedGymEnv
from tqdm import tqdm  # Import tqdm

def run_recorded_actions_on_emulator_and_save_video():
    # Prompt for session ID, instance ID, and run index
    sess_id = input("Enter session ID: ")
    instance_id = input("Enter instance ID: ")
    run_index = int(input("Enter run index: "))  # Convert to integer

    sess_path = Path(f'session_{sess_id}')
    tdf = pd.read_csv(f"session_{sess_id}/agent_stats_{instance_id}.csv.gz", compression='gzip', low_memory=False)
    tdf = tdf[tdf['map'] != 'map']  # remove unused 
    action_arrays = np.array_split(tdf, np.array((tdf["step"].astype(int) == 0).sum()))
    action_list = [int(x) for x in list(action_arrays[run_index]["last_action"])]
    max_steps = len(action_list) - 1

    env_config = {
        'headless': True, 'save_final_state': True, 'early_stop': False,
        'action_freq': 24, 'init_state': '../has_pokedex_nballs.state', 'max_steps': max_steps,
        'print_rewards': False, 'extra_buttons': True, 'save_video': True, 'fast_video': True, 'session_path': sess_path,
        'gb_path': '../PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0, 'instance_id': f'{instance_id}_recorded'
    }
    env = RedGymEnv(env_config)
    env.reset_count = run_index

    obs = env.reset()
    for action in tqdm(action_list, desc="Uh, Ah: Bald ist dein episches Video da!"):
        obs, rewards, term, trunc, info = env.step(action)
        env.render()

# Now call the function directly
if __name__ == "__main__":
    run_recorded_actions_on_emulator_and_save_video()

